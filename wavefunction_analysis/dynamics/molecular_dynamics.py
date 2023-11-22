import os, sys
import numpy as np
import scipy

from pyscf import scf, gto, df, lib, grad

from wavefunction_analysis.utils import print_matrix, convert_units
from wavefunction_analysis.utils.ortho_ao_basis import get_ortho_basis
from wavefunction_analysis.plot import plt


AU_TIME_IN_SEC = 2.0 * 6.6260693e-34 * 8.854187817e-12 * 5.291772108e-11 \
                    / 1.60217653e-19 / 1.60217653e-19
kT_AU_to_Kelvin = 0.25 * 9.1093826e-31 * (1.60217653e-19*1.60217653e-19 * 8.854187817e-12 * 6.6260693e-34)**2 / 1.3806505e-23

FS = 1.0e15
BOHR = 0.52917721067121
ELECTRON_MASS_IN_AMU = 5.4857990945e-04

def get_boltzmann_beta(temperature):
    from pyscf.data.nist import HARTREE2J, BOLTZMANN
    return HARTREE2J / (BOLTZMANN * temperature)


class NuclearDynamicsStep():
    """
    update_method has options
    euler, leapforg, velocity_verlet
    """

    def __init__(self, atmsym, init_coords,
            nuclear_dt, nuclear_update_method, nuclear_save_nframe,
            init_velocity=None, init_kick=None):

        #key.setdefault('temperature', 298) # system temperature
        #key.setdefault('nuclear_dt', 10)
        #key.setdefault('nuclear_update_method', 'velocity_verlet')
        #key.setdefault('nuclear_nuclear_save_nframe', 0)
        #key.setdefault('init_velocity', None)
        #key.setdefault('init_kick', None)

        #for name, value in key.items(): # put all the variables in the class
        #    setattr(self, name, value)

        self.nuclear_dt = nuclear_dt
        #self.nuclear_t_total = nuclear_t_total
        #self.nuclear_ntimes = int(self.nuclear_t_total/self.nuclear_dt)
        self.nuclear_update_method = nuclear_update_method

        self.atmsym = atmsym
        self.natoms = len(atmsym)

        self.nuclear_mass = np.zeros(self.natoms)
        # use pyscf's
        from pyscf.data import elements
        for i in range(self.natoms):
            self.nuclear_mass[i] = elements.MASSES[elements.charge(self.atmsym[i])] / ELECTRON_MASS_IN_AMU
        #print_matrix('nuclear_mass:\n', self.nuclear_mass)

        self.nuclear_save_nframe = nuclear_save_nframe
        #self.nuclear_kinetic = np.zeros((self.nuclear_save_nframe))
        #self.nuclear_energies_total = np.zeros((self.nuclear_save_nframe))

        #self.nuclear_coordinates = np.zeros((self.nuclear_save_nframe, self.natoms, 3))
        self.nuclear_coordinates = np.reshape(init_coords, (self.natoms, 3))
        # change to A.U.
        self.nuclear_coordinates /= BOHR
        #self.nuclear_velocities = np.zeros((self.nuclear_save_nframe, self.natoms, 3))
        #self.nuclear_velocities = np.zeros((self.natoms, 3))
        #self.nuclear_forces = np.zeros((self.nuclear_save_nframe, self.natoms, 3))
        if isinstance(init_velocity, list):
            init_velocity = np.reshape(init_velocity, (self.natoms, 3))
        elif init_velocity == 'random':
            init_velocity = self.init_velocity_random(2e-3)
        elif 'thermo' in init_velocity:
            init_velocity = self.init_velocity_thermo(float(init_velocity.split('_')[1]))

        if isinstance(init_velocity, np.ndarray):
            self.nuclear_velocities = init_velocity
            self.remove_trans_rotat_velocity()
            self.get_nuclear_kinetic_energy(self.nuclear_velocities)
        else:
            self.nuclear_velocities = np.zeros((self.natoms, 3))
            self.nuclear_kinetic = 0.
            self.nuclear_temperature = 0.

        if init_kick:
            self.nuclear_forces = np.reshape(init_kick, (self.natoms, 3))
        else: self.nuclear_forces = np.zeros((self.natoms, 3))


    def init_velocity_random(self, etrans, sigma=1e-4, scale=.1):
        """
        random kinetic energy for atoms at three directions
        """
        size = 3 * self.natoms

        #etrans = convert_units(etrans*scale, 'eh', 'kcal')
        mean = etrans / float(size)

        rng = np.random.default_rng()
        # mean is the center
        # sigma is the standard deviation whose square is variance
        ek = rng.normal(loc=mean, scale=sigma, size=size)
        ek = np.abs(ek) * etrans / np.sum(ek) # scale by the generated kinetic energy

        sign = rng.random((self.natoms, 3))
        sign = np.where(sign>.5, 1, -1)
        velocity = 2.*np.einsum('ix,i->ix', ek.reshape(self.natoms, 3), 1./self.nuclear_mass)
        velocity = np.einsum('ix,ix->ix', sign, np.sqrt(velocity))
        return velocity


    def init_velocity_thermo(self, temp):
        """
        random velocity following Boltzmann distribution
        """
        #etrans = unit_conversion(temp, 'k', 'eh')
        #sigma = np.sqrt(etrans/self.nuclear_mass)
        beta_b = get_boltzmann_beta(temp)
        sigma = np.sqrt(1./beta_b/self.nuclear_mass)
        velocity = np.zeros((self.natoms, 3))

        rng = np.random.default_rng()
        for i in range(self.natoms):
            velocity[i] = rng.normal(loc=0., scale=sigma[i], size=3)

        return velocity


    def update_nuclear_coords_velocity(self, nuclear_force):
        if self.nuclear_update_method == 'euler':
            self.euler_step(nuclear_force)
        elif self.nuclear_update_method == 'leapfrog':
            self.leapfrog_step(nuclear_force)
        elif self.nuclear_update_method == 'velocity_verlet':
            self.velocity_verlet_step(nuclear_force, 1)
            # we will finish the last falf after electronic step


    def update_nuclear_coords_velocity2(self, nuclear_force):
        # velocity_verlet is cumbersome
        if self.nuclear_update_method == 'velocity_verlet':
            self.velocity_verlet_step(nuclear_force, 2)

        self.remove_trans_rotat_velocity()


    def euler_step(self, nuclear_force):
        self.nuclear_velocities += self.nuclear_dt * np.einsum('ix,i->ix', nuclear_force, 1./self.nuclear_mass)
        self.nuclear_coordinates += self.nuclear_dt * self.nuclear_velocities
        self.get_nuclear_kinetic_energy(self.nuclear_velocities)


    def leapfrog_step(self, nuclear_force):
        old_nuclear_velocities = np.copy(self.nuclear_velocities)
        self.nuclear_velocities += self.nuclear_dt * np.einsum('ix,i->ix', nuclear_force, 1./self.nuclear_mass)
        self.nuclear_coordinates += self.nuclear_dt * self.nuclear_velocities

        average_nuclear_velocities = 0.5 * (old_nuclear_velocities + self.nuclear_velocities)
        self.get_nuclear_kinetic_energy(average_nuclear_velocities)


    def velocity_verlet_step(self, nuclear_force, half):
        self.nuclear_velocities += 0.5 * self.nuclear_dt * np.einsum('ix,i->ix', nuclear_force, 1./self.nuclear_mass)
        if half == 1:
            self.nuclear_coordinates += self.nuclear_dt * self.nuclear_velocities
        if half == 2:
            self.get_nuclear_kinetic_energy(self.nuclear_velocities)


    def get_nuclear_kinetic_energy(self, velocities, mass=None):
        if mass is None: mass = self.nuclear_mass

        v2 = np.einsum('ix,ix->i', velocities, velocities)
        self.nuclear_kinetic = 0.5 * np.einsum('i,i', mass, v2)
        self.nuclear_temperature = self.nuclear_kinetic * 2 / (velocities.size)


    def remove_trans_rotat_velocity(self):
        return
        # translational part
        v0 = np.einsum('i,ix->x', self.nuclear_mass, self.nuclear_velocities)
        #v0 /= np.sum(self.nuclear_mass)
        #self.nuclear_velocities[:] -= v0
        self.nuclear_velocities -= np.einsum('x,i->ix', v0/self.natoms, 1./self.nuclear_mass)

        # rotational part
        # total angular momentum
        L = np.einsum('i,ix->x', self.nuclear_mass, np.cross(self.nuclear_coordinates, self.nuclear_velocities))
        # total moment of inertia tensor
        x2 = np.einsum('i,ix,iy->xy', self.nuclear_mass, self.nuclear_coordinates, self.nuclear_coordinates)
        I = -np.copy(x2) # off-diagonal
        for i in range(3): # diagonal
            j, k = (i+1)%3, (i+2)%3
            I[i,i] = (x2[j,j]+x2[k,k])

        print_matrix('I', I)
        print_matrix('L', L)
        # total angular velocity I^-1 * L
        U, s, Vt = np.linalg.svd(I)
        idx = np.where(s[np.abs(s)>1e-10])[0]
        s, U, Vt = 1./s[idx], U[:,idx], Vt[:,idx]
        W = np.einsum('ij,j,kj,k->i', Vt, s, U, L)
        #L = np.zeros((self.natoms, 3)) # debug
        #for i in range(self.natoms):
        #    L[i] = np.cross(W, self.nuclear_coordinates[i])
        self.nuclear_velocities -= np.cross(W, self.nuclear_coordinates)



def cal_idempotency(P, S=None):
    if S is None:
        PSP = np.einsum('ij,jk->ik', P, P)
        print_matrix('idempotency in orthogonal basis:\n', PSP-P)
    else:
        PSP = np.einsum('ij,jk,kl->il', P, S, P)
        print_matrix('idempotency in atomic basis:\n', PSP-P)

    return PSP


def density_purification(P, S=None):
    # McWeeny formula
    if S is None:
        P2 = np.einsum('ij,jk->ik', P, P)
        P = 3.*P2 - 2.*np.einsum('ij,jk->ik', P, P2)
    else:
        P2 = np.einsum('ij,jk,kl->il', P, S, P)
        P = 3.*P2 - 2.*np.einsum('ij,jk,kl->il', P, S, P2)
    return P


def cal_electronic_energy1(mf, P1, Z=None, method='lowdin'):
    """normal energy"""
    if Z is None: Z = get_ortho_basis(mf.get_ovlp(), method)[1]

    # build fock
    hcore = mf.get_hcore()
    vjk = mf.get_veff(mf.mol, P1)
    F = hcore + vjk

    energy_tot = np.einsum('ij,ji->', hcore, P1)
    energy_tot += 0.5 * np.einsum('ij,ji->', vjk, P1)
    energy_tot += mf.energy_nuc()

    return energy_tot, F


def cal_electronic_energy2(mf, P1, Z=None, method='lowdin'):
    """extended lagrange energy"""
    if Z is None: Z = get_ortho_basis(mf.get_ovlp(), method)[1]

    # build fock
    hcore = mf.get_hcore()
    vjk = mf.get_veff(mf.mol, (P1+P1.T)*.5) # symmetrize density matrix
    F = hcore + vjk

    #print('elec: ', mf.mol.nelectron//2)
    F_ortho = np.einsum('ji,jk,kl->il', Z, F, Z)
    _, Q = np.linalg.eigh(F_ortho)
    Q = Q[:,:mf.mol.nelectron//2]
    P2 = np.einsum('ij,kj->ik', Q, Q)
    P2 = 2.* np.einsum('ij,jk,lk->il', Z, P2, Z) # total D matrix
    P3 = 2.* P2 - P1 # total


    energy_tot = np.einsum('ij,ji->', hcore, P2)
    energy_tot += .5* np.einsum('ij,ji->', vjk, P3)
    energy_tot += mf.energy_nuc()

    energy_tot2 = 0.

    return energy_tot, P2, P3, F, energy_tot2


def cal_electronic_force(mf, P1, P2=None, P3=None, F=None, Z=None, method='lowdin'):
    if P2 is None: P2 = P1
    if P3 is None: P3 = P1
    if F is None: F = mf.get_fock(dm=P1)
    if Z is None: Z = get_ortho_basis(mf.get_ovlp(), method)[1]

    g = grad.rhf.Gradients(mf)
    hcore_deriv = g.hcore_generator(mf.mol)
    vjk_deriv = g.get_veff(mf.mol, (P1+P1.T)*.5) # symmetrize density matrix
    ovlp_deriv = g.get_ovlp(mf.mol)

    ZZtFP = np.einsum('ij,kj->ik', Z, Z)
    ZZtFP = np.einsum('ij,jk,kl->il', ZZtFP, F, P2)
    ZZtFP += ZZtFP.T


    de = np.zeros((mf.mol.natm, 3))
    aoslices = mf.mol.aoslice_by_atom()
    atmlst = range(mf.mol.natm)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        h1ao = hcore_deriv(ia)
        de[k] += np.einsum('xij,ij->x', h1ao, P2)
        de[k] += 2 * np.einsum('xij,ij->x', vjk_deriv[:,p0:p1], P3[p0:p1])
        de[k] -= np.einsum('xij,ij->x', ovlp_deriv[:,p0:p1], ZZtFP[p0:p1])
        #for m in range(p0, p1):
        #    de[k] -= ovlp_deriv[:,m,m] * FP[m,m]
        #    for n in range(p0, m):
        #        de[k] -= 2 * ovlp_deriv[:,m,n] * FP[m,n]

    de += grad.rhf.grad_nuc(mf.mol)
    electronic_forces = -de

    #print_matrix('electronic_forces:\n', electronic_forces)
    return electronic_forces


def cal_pulay_force(mf, F, P, Fao, Pao, Z, method='lowdin'):
    g = grad.rhf.Gradients(mf)
    hcore_deriv = g.hcore_generator(mf.mol)
    vjk_deriv = g.get_veff(mf.mol, Pao)
    ovlp_deriv = g.get_ovlp(mf.mol)

    PFP = np.einsum('ij,jk,kl->il', Pao, Fao, Pao)

    de = np.zeros((mf.mol.natm, 3))
    aoslices = mf.mol.aoslice_by_atom()
    atmlst = range(mf.mol.natm)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        h1ao = hcore_deriv(ia)
        de[k] += np.einsum('xij,ij->x', h1ao, Pao)
        de[k] += 2 * np.einsum('xij,ij->x', vjk_deriv[:,p0:p1], Pao[p0:p1])
        de[k] -= 2 * np.einsum('xij,ij->x', ovlp_deriv[:,p0:p1], PFP[p0:p1])

    de += grad.rhf.grad_nuc(mf.mol)

    nbas = Z.shape[0]
    ZxL = np.zeros((mf.mol.natm, 3, nbas, nbas))
    FP = np.einsum('ij,jk->ik', F, P)

    if method == 'lowdin':
        e, v = np.linalg.eigh(Z)

        n = len(e)
        einv = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                einv[i,j] = (e[i] * e[j]) / (e[i] + e[j])

        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia,2:]
            usu = np.einsum('ki,xkl,lj,ij->xij', v[p0:p1], ovlp_deriv[:,p0:p1], v, einv)
            usu += np.einsum('ki,xlk,lj,ij->xij', v, ovlp_deriv[:,p0:p1], v[p0:p1], einv)
            ZxL[k] += np.einsum('xij,ki,lj->xkl', usu, v, v)

        FP = np.einsum('ij,jk->ik', FP, Z)

    elif method == 'cholesky':
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia,2:]
            ZxL[k] += np.einsum('ip,xpq,jq->xij', Z[:,p0:p1], ovlp_deriv[:,p0:p1], Z)
            ZxL[k] += np.einsum('ip,xqp,jq->xij', Z, ovlp_deriv[:,p0:p1], Z[:,p0:p1])

        for i in range(nbas):
            ZxL[:,:,i,:i] = 0.
            ZxL[:,:,i,i] *= 0.5

    de += 2.*np.einsum('axij,ji->ax', ZxL, FP)

    return (-de)


def run_pyscf_gs(atom, functional, basis, charge, unit='Bohr', efield=None, max_memory=60000, verbose=1):
    mol = gto.M(
            verbose = verbose,
            atom = atom,
            unit = unit,
            basis = basis,
            charge = charge,
            max_memory = max_memory)

    # ground-state
    mf = scf.RKS(mol)
    mf.xc = functional

    if efield:
        mol.set_common_orig([0, 0, 0])
        h = (mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
            + np.einsum('x,xij->ij', efield, mol.intor('cint1e_r_sph', comp=3)))
        mf.get_hcore = lambda *args: h

    return mf



class ElectronicDynamicsStep():
    def __init__(self, key):
        # default values
        key.setdefault('electron_software', 'pyscf')
        key.setdefault('functional', 'pbe0')
        key.setdefault('basis', 'sto-3g')
        key.setdefault('atmsym', None)
        key.setdefault('charge', 0)

        key.setdefault('electronic_dt', 0)
        key.setdefault('electronic_update_method', 'velocity_verlet')
        key.setdefault('electronic_save_nframe', 0)
        key.setdefault('ortho_method', 'lowdin')
        key.setdefault('init_density', None)
        key.setdefault('init_efield', None)
        key.setdefault('max_memory', 60000)
        key.setdefault('verbose', 1)
        key.setdefault('unit', 'Bohr') # for coordinates

        for name, value in key.items(): # put all the variables in the class
            setattr(self, name, value)

        self.electronic_kinetic = 0


    def get_ortho_basis(self, ovlp=None):
        if ovlp is None: ovlp = self.mf.get_ovlp()
        L, Z, ovlp_inv = get_ortho_basis(ovlp, self.ortho_method)

        return L, Z, ovlp_inv


    def init_electronic_density_static(self, coords):
        self.setup_electronic_basis(coords)
        #self.mf.kernel()
        self.energy_tot = self.mf.scf()
        self.energy_tot2 = self.energy_tot
        self.cal_electronic_force()

        return self.energy_tot, self.electronic_forces


    def update_electronic_density_static(self, coords):
        self.setup_electronic_basis(coords)

        #self.mf.kernel()
        self.energy_tot = self.mf.scf()
        self.cal_electronic_force()

        return self.energy_tot, self.electronic_forces


    def setup_electronic_basis(self, coords):
        # construct atom
        atom = ''
        for i in range(len(self.atmsym)):
            atom += str(self.atmsym[i]) + ' '
            for x in range(3):
                atom += str(coords[i,x]) + ' '
            atom += ';  '
        #print('current coords in ED:\n', atom)

        if self.electron_software == 'pyscf':
            self.mf = run_pyscf_gs(atom, self.functional, self.basis, self.charge,
                    self.unit, self.init_efield, self.max_memory, self.verbose)


    def cal_electronic_force(self):
        if self.electron_software == 'pyscf':
            #g = self.mf.Gradients()
            g = grad.rhf.Gradients(self.mf)
            self.electronic_forces = - g.grad()

        return self.electronic_forces


    # electronic dynamics
    def update_electronic_density_dynamics(self, coords):
        return




class ExtendedLagElectronicDynamicsStep(ElectronicDynamicsStep):
    """
    refer: Niklasson 2009 JCP 10.1063/1.3148075
           Niklasson 2017 JCP 10.1063/1.4985893
           Niklasson 2020 JCP 10.1063/1.5143270
    """
    def __init__(self, key):
        super().__init__(key)

        # default values
        if not hasattr(self, 'xl_nk'): self.xl_nk = 6
        if not hasattr(self, 'xl_chem_potential'): self.xl_chem_potential = -0.166401
        if not hasattr(self, 'xl_inv_temp'): self.xl_inv_temp = 1500
        self.xl_inv_temp = kT_AU_to_Kelvin / self.xl_inv_temp

        self.ncall = 1


    def init_xl_variables(self):
        if self.xl_nk == 3:
            self.xl_kappa, self.xl_alpha = 1.69, 0.150
            self.xl_cs = [-2, 3, 0, -1]
        elif self.xl_nk == 4:
            self.xl_kappa, self.xl_alpha = 1.75, 0.057
            self.xl_cs = [-3, 6, -2, -2, 1]
        elif self.xl_nk == 5:
            self.xl_kappa, self.xl_alpha = 1.82, 0.018
            self.xl_cs = [-6, 14, -8, -3, 4, -1]
        elif self.xl_nk == 6:
            self.xl_kappa, self.xl_alpha = 1.84, 0.0055
            self.xl_cs = [-14, 36, -27, -2, 12, -6, 1]
        elif self.xl_nk == 7:
            self.xl_kappa, self.xl_alpha = 1.86, 0.0016
            self.xl_cs = [-36, 99, -88, 11, 32, -25, 8, -1]
        elif self.xl_nk == 9:
            self.xl_kappa, self.xl_alpha = 1.89, 0.00012
            self.xl_cs = [-286, 858, -936, 364, 168, -300, 184, -63, 12, -1]
        self.xl_cs = np.flip(self.xl_cs) * self.xl_alpha

        # init with converged density, after SCF!
        self.Pao = self.mf.make_rdm1()
        self.nao = self.Pao.shape[0]
        DS = np.einsum('ij,jk->ik', self.Pao, self.mf.get_ovlp())

        self.xl_Xdotdot = np.zeros((self.nao, self.nao))
        self.xl_auxiliary_density = np.zeros((self.xl_nk+2, self.nao, self.nao))
        #for k in range(self.xl_nk+1):
        #    self.xl_auxiliary_density[k] = DS
        self.xl_auxiliary_density[0] = DS


    def init_electronic_density_static(self, coords):
        super().init_electronic_density_static(coords)
        self.init_xl_variables()

        return self.energy_tot, self.electronic_forces


    def update_electronic_density_static(self, coords):
        if self.ncall > self.xl_nk:
            self.static_extended_lagrange()
            self.xl_build_fock(coords)
        else: # do scf for first xl_nk steps
            self.xl_build_fock_init(coords, self.ncall)
            self.ncall += 1

        return self.energy_tot, self.electronic_forces


    def static_extended_lagrange(self):
        self.xl_auxiliary_density[self.xl_nk+1] = 2 * self.xl_auxiliary_density[self.xl_nk]
        self.xl_auxiliary_density[self.xl_nk+1] -= self.xl_auxiliary_density[self.xl_nk-1]
        self.xl_auxiliary_density[self.xl_nk+1] += self.xl_kappa * self.xl_Xdotdot
        for k in range(self.xl_nk+1):
            self.xl_auxiliary_density[self.xl_nk+1] += self.xl_cs[k] * self.xl_auxiliary_density[k]

        # move these forward
        for k in range(self.xl_nk+1):
            self.xl_auxiliary_density[k] = self.xl_auxiliary_density[k+1]


    def xl_build_fock(self, coords):
        # set new coordinates
        self.setup_electronic_basis(coords)
        ovlp = self.mf.get_ovlp() # need it later
        L, Z, ovlp_inv = self.get_ortho_basis(ovlp)

        # density
        self.Pao = np.einsum('ij,jk->ik', self.xl_auxiliary_density[self.xl_nk], ovlp_inv)
        #self.Pao = np.einsum('ij,jk,lk->ik', Z, self.xl_auxiliary_density[self.xl_nk], Z)

        # energy and force
        self.energy_tot, P2, P3, F, self.energy_tot2 = cal_electronic_energy2(self.mf, self.Pao, Z)
        self.electronic_forces = cal_electronic_force(self.mf, self.Pao, P2, P3, F, Z)

        # calculate the second derivatives of xl_auxiliary_density
        self.xl_Xdotdot = np.einsum('ij,jk->ik', P2, ovlp) - self.xl_auxiliary_density[self.xl_nk]
        #self.xl_Xdotdot = np.einsum('ji,jk,kl->il', L, P2, L) - self.xl_auxiliary_density[self.xl_nk]

        return self.energy_tot, self.electronic_forces


    def xl_build_fock_init(self, coords, k):
        super().init_electronic_density_static(coords)

        self.Pao = self.mf.make_rdm1()
        DS = np.einsum('ij,jk->ik', self.Pao, self.mf.get_ovlp())
        self.xl_auxiliary_density[k] = DS

        return self.energy_tot, self.electronic_forces


    def xl_auxiliary_density_dotdot(self):
        print('xl_auxiliary continue')
        L, Z, ovlp_inv = self.get_ortho_basis()

        # used for the \delta D / \delta \lambda
        F0 = self.mf.get_hcore()
        P = np.einsum('ij,jk->ik', self.xl_auxiliary_density[self.xl_nk], ovlp_inv)
        F0 += self.mf.get_veff(self.mf.mol, P)
        F0 = np.einsum('ji,jk,kl->il', Z, F0, Z)
        e, Q = np.linalg.eigh(F0)
        F0 = np.einsum('ji,jk,kl->il', Q, F0, Q)
        print_matrix('F0 should be diagonal 2\n', F0)
        ###

        # krylov space
        W = np.zeros((50, self.nao, self.nao))
        V = np.zeros((50, self.nao, self.nao))

        W[0] = np.einsum('ij,jk->ik', self.Pao, ovlp) - self.xl_auxiliary_density[self.xl_nk]
        W[0] = np.einsum('ij,jk->ik', W[0], self.xl_preconditioner_k)
        print_matrix('W0:\n', W[0])

        thresh = 1.0e-8
        m, error = 0, 1.0
        while error > thresh:
            m += 1
            V[m] = W[m-1]
            if m > 1:
                for j in range(1, m-1):
                    V[m] -= np.einsum('ij,ij->', V[m], V[j]) * V[j]
            V[m] /= np.linalg.norm(V[m])
            print_matrix('vm:\n', V[m])

            F1 = self.mf.get_veff(self.mf.mol, np.einsum('ij,jk->ik', V[m], ovlp_inv))
            F1 = np.einsum('ji,jk,kl->il', Z, F1, Z)
            F1 = np.einsum('ji,jk,kl->il', Q, F1, Q)

            n = 8
            c = 1.0 / np.power(2, 2+n)
            X0 = (0.5 - self.xl_chem_potential * c) * self.xl_identity - c * F0
            X1 = - c * F1
            for i in range(n):
                Xtmp = np.einsum('ij,jk->ik', X0, X1) - np.einsum('ij,jk->ik', X1, X0)
                Ytmp = 2 * (np.einsum('ij,jk->ik', X0, X0) - X0) + self.xl_identity
                Ytmp = np.linalg.inv(Ytmp)
                X0 = np.einsum('ij,jk,kl->il', Ytmp, X0, X0)
                X1 = Xtmp + 2 * np.einsum('ij,jk->ik', (X1 - Xtmp), X0)
                X1 = np.einsum('ij,jk->ik', Ytmp, X1)

            print_matrix('X0\n', X0)
            print_matrix('X1\n', X1)
            D0mu = self.xl_inv_temp * np.einsum('ij,jk->ik', X0, (self.xl_identity - X0))
            print_matrix('D0mu\n', D0mu)
            D1 = X1 - np.trace(X1) / np.trace(D0mu) * D0mu
            D1 = np.einsum('ij,jk,lk->il', Q, D1, Q)
            D1 = np.einsum('ij,jk,kl->il', Z, D1, L)
            W[m] = np.einsum('ij,jk->ik', self.xl_preconditioner_k, (D1 - V[m]))

            W0_res = np.zeros((self.nao, self.nao))
            for k in range(1, m):
                for l in range(1, k):
                    O = np.einsum('ij,ij->', W[k], W[l])
                    W0_res += W[k] * np.einsum('ij,ij->', W[l], W[0]) / O
                    W0_res += W[l] * np.einsum('ij,ij->', W[k], W[0]) / O

            W0_res -= W[0]
            print_matrix('W0_res:\n', W0_res)
            error = np.einsum('ij,ij->', W0_res, W0_res) / np.einsum('ij,ij->', W[0], W[0])
            print('error:', error)

        Xdotdot = np.zeros((self.nao, self.nao))
        for k in range(1, m):
            for l in range(1, k):
                O = np.einsum('ij,ij->', W[k], W[l])
                Xdotdot -= V[k] * np.einsum('ij,ij->', W[l], W[0]) / O
                Xdotdot -= V[l] * np.einsum('ij,ij->', W[k], W[0]) / O

        self.xl_Xdotdot = Xdotdot




class CurvyElectronicDynamicsStep(ElectronicDynamicsStep):
    """
    refer: Herbert 2004 JCP 10.1063/1.1814934
    """
    def __init__(self, key):
        super().__init__(key)

        # default values
        if not hasattr(self, 'cy_core_e_thresh'): self.cy_core_e_thresh = -10.0
        if not hasattr(self, 'cy_fic_mass'): self.cy_fic_mass = 360
        if not hasattr(self, 'core_energy'): self.core_energy = 0.
        self.cy_fic_mass = np.sqrt(self.cy_fic_mass)


    def init_electronic_density_static(self, coords):
        super().init_electronic_density_static(coords)
        self.Pao = self.mf.make_rdm1()
        #print_matrix('init P:\n', self.Pao)
        self.nao = self.Pao.shape[0]
        self.cy_delta_dot = np.zeros((self.nao, self.nao))
        self.cy_delta = np.zeros((self.nao, self.nao))

        return self.energy_tot, self.electronic_forces


    def update_electronic_density_static(self, coords):
        self.setup_electronic_basis(coords)

        G, F, P, Z = self.cy_orbital_response()
        mass_bas = self.cy_get_mass_on_basis(F)
        #if self.electronic_update_method == 'velocity_verlet':
        #    self.velocity_verlet_step(G, mass_bas, 2)

        self.cy_update_delta(G, mass_bas)
        self.cal_electronic_energy_force(F, P, Z)
        P = self.cy_update_density(P, Z)

        return self.energy_tot, self.electronic_forces


    def update_electronic_density_static2(self, coords):
        if self.electronic_update_method == 'velocity_verlet':
#            self.setup_electronic_basis(coords)
            G, F, P, Z = self.cy_orbital_response()
            mass_bas = self.cy_get_mass_on_basis(F)
            self.velocity_verlet_step(G, mass_bas, 2)
            cal_idempotency(self.Pao*.5, self.mf.get_ovlp())
            cal_idempotency(P*.5)

        self.energy_tot += self.electronic_kinetic
        return self.energy_tot, self.electronic_forces


    def cy_get_mass_on_basis(self, F):
        if self.core_energy >= 0:
            sqrt_mass = np.ones(self.nao) * self.cy_fic_mass
        else:
            sqrt_mass = np.zeros(self.nao)
            for i in range(self.nao):
                factor = 1
                if F[i,i] < self.cy_core_e_thresh:
                    factor = 1 + 2 * np.sqrt(np.abs(F[i,i]-self.cy_core_e_thresh))
                sqrt_mass[i] = self.cy_fic_mass * factor

        mass_bas = np.einsum('i,j->ij', sqrt_mass, sqrt_mass)
        #print_matrix('mass_ba\n', mass_bas)
        return mass_bas


    def cal_electronic_energy_force(self, F, P, Z):
        self.energy_tot, Fao = cal_electronic_energy1(self.mf, self.Pao, Z)
        #self.electronic_forces = cal_electronic_force(self.mf, self.Pao, Z=Z)
        #self.cal_electronic_force()
        self.electronic_forces = cal_pulay_force(self.mf, F, P, Fao, self.Pao, Z, self.ortho_method)

        return self.energy_tot, self.electronic_forces


    def cy_orbital_response(self):
        F = self.mf.get_fock(dm=self.Pao) # AO
        #hcore = self.mf.get_hcore()
        #vjk = self.mf.get_veff(self.mf.mol, self.Pao)
        #F = hcore + vjk

        L, Z = self.get_ortho_basis()[:2]
        P = np.einsum('ji,jk,kl->il', L, self.Pao, L)
        F = np.einsum('ij,jk,lk->il', Z, F, Z) # orthogonal basis

        # we use the minus version
        G = np.einsum('ij,jk->ik', F, P)
        G -= G.T

        return G, F, P, Z


    def cy_update_delta(self, G, mass_bas):

        if self.electronic_update_method == 'euler':
            self.euler_step(G, mass_bas)
        elif self.electronic_update_method == 'leapfrog':
            self.leapfrog_step(G, mass_bas)
        elif self.electronic_update_method == 'velocity_verlet':
            self.velocity_verlet_step(G, mass_bas, 1)
            # we will finish the last falf after electronic step


    def euler_step(self, G, mass_bas):
        self.cy_delta_dot += self.electronic_dt * np.einsum('ij,ij->ij', G, 1/mass_bas)
        self.cy_delta = self.electronic_dt * self.cy_delta_dot
        self.get_electronic_kinetic_energy(self.cy_delta_dot, mass_bas)


    def leapfrog_step(self, G, mass_bas):
        old_cy_delta_dot = 0.5 * self.cy_delta_dot
        self.cy_delta_dot += self.electronic_dt * np.einsum('ij,ij->ij', G, 1/mass_bas)
        self.cy_delta = self.electronic_dt * self.cy_delta_dot
        old_cy_delta_dot += 0.5 * self.cy_delta_dot
        self.get_electronic_kinetic_energy(old_cy_delta_dot, mass_bas)


    def velocity_verlet_step(self, G, mass_bas, half):
        self.cy_delta_dot -= 0.5 * self.electronic_dt * np.einsum('ij,ij->ij', G, 1/mass_bas)
        if half == 1:
            self.cy_delta = self.electronic_dt * self.cy_delta_dot
        if half == 2:
            self.get_electronic_kinetic_energy(self.cy_delta_dot, mass_bas)
            print('electronic_kinetic:', self.electronic_kinetic)
            #expdelta = scipy.linalg.expm(self.cy_delta)
            #self.cy_delta_dot = np.einsum('ij,jk->ik', expdelta, self.cy_delta_dot)
            #self.cy_delta = np.zeros((self.nao, self.nao)) # reset to zero # important!


    def cy_update_density(self, P, Z):
        expdelta = scipy.linalg.expm(self.cy_delta)
        P = np.einsum('ij,jk,lk->il', expdelta, P, expdelta)
        self.Pao = np.einsum('ji,jk,kl->il', Z, P, Z)

        return P


    def get_electronic_kinetic_energy(self, velocities, mass):
        v2 = np.einsum('ij,ij->ij', velocities, velocities)
        self.electronic_kinetic = 0.5 * np.einsum('ij,ij', mass, v2)
        self.electronic_temperature = self.electronic_kinetic * 2 / (velocities.size)




class GrassmannElectronicDynamicsStep(ElectronicDynamicsStep):

    def __init__(self, key):
        super().__init__(key)

        # default values
        if not hasattr(self, 'npoints'): self.npoints = 11#14

        self.ncall = -1


    def init_electronic_density_static(self, coords):
        super().init_electronic_density_static(coords)
        self.Pao = self.mf.make_rdm1()
        self.nao = self.Pao.shape[0]
        self.nocc = self.mf.mol.nelectron // 2 # closed-shell restricted

        if coords.shape[0] > 3: self.npoints = coords.shape[0] * 3 + 1

        self.lagrange = np.zeros((self.npoints))
        self.gamma = np.zeros((self.npoints, self.nao, self.nocc))
        self.coords_array = np.zeros((self.npoints, self.mf.mol.natm*3))

        self.update_gamma_lagrange(coords)

        return self.energy_tot, self.electronic_forces


    def update_electronic_density_static(self, coords):
        if self.ncall < self.npoints-1:
            super().update_electronic_density_static(coords)
            self.update_gamma_lagrange(coords)

            self.Pao = self.mf.make_rdm1()
            cal_idempotency(self.Pao*.5, self.mf.get_ovlp())
        else:
            P, Z = self.interpolation(coords)
            self.energy_tot, Fao = cal_electronic_energy1(self.mf, self.Pao, Z)
            F = np.einsum('ij,jk,lk->il', Z, Fao, Z)
            self.electronic_forces = cal_pulay_force(self.mf, F, P, Fao, self.Pao, Z, self.ortho_method)

            energy0 = self.mf.scf()
            Pao0 = self.mf.make_rdm1()
            print('energy:', energy0, self.energy_tot, energy0-self.energy_tot)
            print('Pao:', Pao0, self.Pao, np.linalg.norm(Pao0-self.Pao))

        return self.energy_tot, self.electronic_forces


    def update_gamma_lagrange(self, coords):
        self.ncall += 1
        L = self.get_ortho_basis()[0]

        mo_coeff = self.mf.mo_coeff
        occidx = np.where(self.mf.mo_occ>0)[0]
        orbo = mo_coeff[:,occidx]
        C = np.einsum('ji,jk->ik', L, orbo)
        if self.ncall == 0:
            self.C_ref = np.copy(C)
            self.coords_array[self.ncall] = np.copy(coords.flatten())

            self.C_ref_v = np.einsum('ji,jk->ik', L, mo_coeff[:,1:]) # tmp
            self.F_prev = self.mf.get_fock()
            _, Z, _ = self.get_ortho_basis()
            self.F_prev = np.einsum('ij,jk,lk->il', Z, self.F_prev, Z)

        else:
            L1 = np.einsum('ij,jk->ik', C, np.linalg.inv(np.einsum('ji,jk->ik', self.C_ref, C)))
            L1 -= self.C_ref
            u, s, vh = np.linalg.svd(L1, full_matrices=False)
            print('L1:', L1, u, s, vh)
            theta = np.arccos(np.einsum('ji,jk->ik', self.C_ref, C))
            print('theta:', theta)
            print('Cv*tan theta:', np.einsum('ij,jk->ik', self.C_ref_v, np.tan(theta)))
            self.gamma[self.ncall] = np.einsum('ij,j,jl->il', u, np.arctan(s), vh)
            self.coords_array[self.ncall] = np.copy(coords.flatten())
            print('orbital0:', C)

            Pao = self.mf.make_rdm1()
            P = np.einsum('ji,jk,kl->il', L, Pao, L)
            G_elec = np.einsum('ij,jk->ik', self.F_prev, P)
            G_elec -= np.einsum('ij,jk->ik', P, self.F_prev)
            print('electronic G:', G_elec)
            self.F_prev = self.mf.get_fock()
            _, Z, _ = self.get_ortho_basis()
            self.F_prev = np.einsum('ij,jk,lk->il', Z, self.F_prev, Z)

        #self.ncall += 1


    def interpolation(self, coords):
        self.setup_electronic_basis(coords)

        if coords.shape[0] == 2:
            d0 = (coords[1,2]-coords[0,2])
            dk = self.coords_array[:,5] - self.coords_array[:,2]
            for i in range(self.npoints):
                l0, l1 = 1., 1.
                for k in range(self.npoints):
                    if k != i:
                        l0 *= d0 - dk[k]
                        l1 *= dk[i] - dk[k]
                self.lagrange[i] = l0 / l1
            print('d:', d0, dk)

        elif coords.shape[0] > 3:
            array = np.concatenate((np.ones(self.npoints), self.coords_array), axis=1)

            det = 1. / np.linalg.det(array)
            for i in range(self.npoints):
                m = np.copy(array)
                m[i,1:] = np.copy(coords.flatten())
                self.lagrange[i] = np.linalg.det(m) * det

        print('lagrange:', self.lagrange, np.sum(self.lagrange))
#        gamma = np.einsum('ipq,i->pq', self.gamma, self.lagrange)
        gamma = self.gamma[1]
        print('gamma2: ', self.gamma, gamma)

        u, s, vh = np.linalg.svd(gamma, full_matrices=False)
        C = np.einsum('ij,kj,k->ik', self.C_ref, vh, np.cos(s))
        C += np.einsum('ij,j->ij', u, np.sin(s))
        C = np.einsum('ij,jk->ik', C, vh)
        print('orbital:', C)

        P = np.einsum('ij,kj->ik', C, C) * 2. # alpha + beta
        L, Z = self.get_ortho_basis()[:2]
        self.Pao = np.einsum('ji,jk,kl->il', Z, P, Z)

        cal_idempotency(self.Pao*.5, self.mf.get_ovlp())
        cal_idempotency(P*.5)

        return P, Z




class MolecularDynamics():
    def __init__(self, key):
        atmsym                = key.get('atmsym', None)
        if atmsym == None:
            raise AttributeError('no molecule symbols given')
        init_coords           = key.get('init_coords', None)
        if init_coords is None:
            raise ValueError('no initial molecule coordinates given')

        self.ed_method        = key.get('ed_method', 'normal')

        total_time            = key.get('total_time', 4000)
        nuclear_dt            = key.get('nuclear_dt', 10)
        nuclear_update_method = key.get('nuclear_update_method', 'velocity_verlet')
        nuclear_save_nframe   = key.get('nuclear_save_nframe', 0)
        init_velocity         = key.get('init_velocity', None)
        init_kick             = key.get('init_kick', None)


        electronic_dt         = key.get('electronic_dt', 0)

        self.nuclear_nsteps = int(total_time/nuclear_dt) + 1
        self.total_time = total_time * AU_TIME_IN_SEC
        self.nuclear_dt = nuclear_dt * AU_TIME_IN_SEC
        self.electronic_dt = electronic_dt * AU_TIME_IN_SEC
        print('running molecular dynamics in\n%4d steps, total time %6.3f fs\n' % (self.nuclear_nsteps, self.total_time*FS))

        self.ndstep = NuclearDynamicsStep(atmsym, init_coords, nuclear_dt,
                                          nuclear_update_method, nuclear_save_nframe,
                                          init_velocity, init_kick)

        if self.ed_method == 'extended_lag':
            self.edstep = ExtendedLagElectronicDynamicsStep(key)
        elif self.ed_method == 'curvy':
            key['electronic_dt'] = nuclear_dt
            key['electronic_update_method'] = nuclear_update_method
            self.edstep = CurvyElectronicDynamicsStep(key)
        elif self.ed_method == 'grassmann':
            self.edstep = GrassmannElectronicDynamicsStep(key)
        else:
            self.edstep = ElectronicDynamicsStep(key)

        self.md_time_total_energies = np.zeros(self.nuclear_nsteps)
        self.md_time_total_energies2 = np.zeros(self.nuclear_nsteps)
        self.md_time_coordinates = np.zeros((self.nuclear_nsteps, self.ndstep.natoms, 3))


    def run_dynamics(self):
        print('current time:%7.3f fs' % 0.0)
        coords = self.ndstep.nuclear_coordinates # equal sign used here as a pointer
        # coords will change when self.ndstep.nuclear_coordinates changes!

        et, electronic_force = self.edstep.init_electronic_density_static(coords)
        #et, electronic_force = self.edstep.update_electronic_density_static(coords)
        print('potential:', et)
        print_matrix('forces:\n', self.edstep.electronic_forces)
        print('kinetic:', self.ndstep.nuclear_kinetic)
        print('temperature: %4.2f K' % float(self.ndstep.nuclear_temperature * kT_AU_to_Kelvin))
        print_matrix('velocities:\n', self.ndstep.nuclear_velocities)
        print_matrix('current nuclear coordinates:\n', coords*BOHR)

        self.md_time_coordinates[0] = coords
        self.md_time_total_energies[0] = et + self.ndstep.nuclear_kinetic
        self.md_time_total_energies2[0] = self.edstep.energy_tot2 + self.ndstep.nuclear_kinetic

        # loop times
        for ti in range(1, self.nuclear_nsteps):
            self.ndstep.update_nuclear_coords_velocity(electronic_force)
            et, electronic_force = self.edstep.update_electronic_density_static(coords)

            print('current time:%7.3f fs' % float(ti*self.nuclear_dt*FS))
            #coords = self.ndstep.nuclear_coordinates # dont need to reassign
            print_matrix('current nuclear coordinates:\n', coords*BOHR)

            ## velocity_verlet is cumbersome
            #if self.ndstep.nuclear_update_method == 'velocity_verlet':
            #    self.ndstep.velocity_verlet_step(electronic_force, 2)
            self.ndstep.update_nuclear_coords_velocity2(electronic_force)

            if self.ed_method == 'curvy':
                et, electronic_force = self.edstep.update_electronic_density_static2(coords)

            print('potential:', et)
            print_matrix('forces:\n', self.edstep.electronic_forces)
            print('kinetic:', self.ndstep.nuclear_kinetic)
            print('temperature: %4.2f K' % float(self.ndstep.nuclear_temperature * kT_AU_to_Kelvin))
            print_matrix('velocities:\n', self.ndstep.nuclear_velocities)

            self.md_time_coordinates[ti] = coords
            self.md_time_total_energies[ti] = et + self.ndstep.nuclear_kinetic #+ self.edstep.electronic_kinetic
            self.md_time_total_energies2[ti] = self.edstep.energy_tot2 + self.ndstep.nuclear_kinetic #+ self.edstep.electronic_kinetic


    def plot_time_variables(self, fig_name=None):
        time_line = np.linspace(0, self.total_time, self.nuclear_nsteps) * FS
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11,6), sharex=True)

        coords = self.md_time_coordinates * BOHR
        ax[0].plot(time_line, coords[:,1,-1]-coords[:,0,-1])
        ax[0].set_ylabel('He--H$^+$ ($\AA$)')
        ax[1].plot(time_line, self.md_time_total_energies-self.md_time_total_energies[0])
        ax[1].set_xlabel('Time (fs)')
        ax[1].set_ylabel('Energy (a.u.)')

        plt.tight_layout()
        if fig_name:
            plt.savefig(fig_name)
        else:
            plt.show()


def plot_time_variables(total_time, nuclear_nsteps, dists, energies):
    time_line = np.linspace(0, total_time, nuclear_nsteps) * FS
    method1 = ['BO', 'XL-3', 'XL-6', 'XL-9', 'Curvy']
    method2 = ['BO', 'XL-3', 'XL-3r', 'XL-6', 'XL-6r', 'XL-9', 'XL-9r', 'Curvy']

    dists = np.array(dists) * BOHR
    energies = np.array(energies)
    energies -= energies[0,0]

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11,6), sharex=True)
    for i in range(dists.shape[0]):
        ax[0].plot(time_line, dists[i], label=method1[i])
        ax[0].set_ylabel('He--H$^+$ Length ($\AA$)')
        ax[0].legend()

    for i in range(energies.shape[0]):
        ax[1].plot(time_line, energies[i], label=method2[i])
        ax[1].set_xlabel('Time (fs)')
        ax[1].set_ylabel('Energy (a.u.)')
        ax[1].legend()

    plt.tight_layout()
    plt.savefig('dynamics')



if __name__ == '__main__':
    mdtype = int(sys.argv[1])

    key = {}
    key['functional'] = 'hf'
    #key['basis'] = '3-21g'
    #key['charge'] = 0
    #key['atmsym'] = [1, 1]
    #key['init_coords'] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]
    key['basis'] = 'sto-3g'
    key['charge'] = 1
    key['atmsym'] = [1, 2]
    key['init_coords'] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.929352]]
    key['init_velocity'] = [[0.0, 0.0, 0.0008], [0.0, 0.0, -0.0008]]

    key['nuclear_dt'] = 10
    #key['total_time'] = 120
    key['total_time'] = 4000
    key['nuclear_update_method'] = 'velocity_verlet'
    key['ortho_method'] = 'lowdin' # 'cholesky'

    if mdtype == 0:
        md = MolecularDynamics(key)
        md.run_dynamics()
        md.plot_time_variables(fig_name='normal_time_coords')

    elif mdtype == 1:
        print('run extended_lag')
        key['ed_method'] = 'extended_lag'
        md = MolecularDynamics(key)
        md.run_dynamics()
        md.plot_time_variables(fig_name='extended_lag_time_coords')

    elif mdtype == 2:
        print('run curvy')
        key['ed_method'] = 'curvy'
        key['ortho_method'] = 'cholesky'
        md = MolecularDynamics(key)
        md.run_dynamics()
        md.plot_time_variables(fig_name='curvy_time_coords')

    elif mdtype == 3:
        key['ed_method'] = 'grassmann'
        md = MolecularDynamics(key)
        md.run_dynamics()
        md.plot_time_variables(fig_name='grassmann_time_coords')

    elif mdtype == 4:
        dists = []
        energies = []

        md = MolecularDynamics(key)
        md.run_dynamics()

        dists.append( md.md_time_coordinates[:,1,-1] - md.md_time_coordinates[:,0,-1])
        energies.append( md.md_time_total_energies)

        key['ed_method'] = 'extended_lag'
        for xl_nk in [3, 6, 9]:

            key['xl_nk'] = xl_nk
            md = MolecularDynamics(key)
            md.run_dynamics()

            dists.append( md.md_time_coordinates[:,1,-1] - md.md_time_coordinates[:,0,-1])
            energies.append( md.md_time_total_energies)
            energies.append( md.md_time_total_energies2)


#        key['ed_method'] = 'curvy'
#        key['ortho_method'] = 'cholesky'
#        md = MolecularDynamics(key)
#        md.run_dynamics()
#        dists.append( md.md_time_coordinates[:,1,-1] - md.md_time_coordinates[:,0,-1])
#        energies.append( md.md_time_total_energies)

        key['ed_method'] = 'grassmann'
        key['ortho_method'] = 'lowdin'
        md = MolecularDynamics(key)
        md.run_dynamics()
        dists.append( md.md_time_coordinates[:,1,-1] - md.md_time_coordinates[:,0,-1])
        energies.append( md.md_time_total_energies)


        np.savetxt('bond.txt', np.array(dists))
        np.savetxt('energy.txt', np.array(energies))
        plot_time_variables(md.total_time, md.nuclear_nsteps, dists, energies)

    elif mdtype == 5:
        dists = np.loadtxt('bond.txt')
        energies = np.loadtxt('energy.txt')
        total_time, nuclear_dt = key['total_time'], key['nuclear_dt']
        plot_time_variables(total_time, int(total_time/nuclear_dt) + 1, dists, energies)


