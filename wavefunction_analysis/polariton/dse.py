import os, sys
import numpy as np

import pyscf

from wavefunction_analysis.utils.pyscf_parser import *
from wavefunction_analysis.utils import convert_units, print_matrix

def get_scaled_lambda(c_lambda, frequency):
    """
    return frequency-scaled coupling strength (c_lambda)
    """
    if isinstance(frequency, float):
        frequency = [frequency]
    return np.einsum('x,i->ix', c_lambda, np.sqrt(frequency)/np.sqrt(2.))


def get_nuclear_dipoles(mol, c_lambda, origin=None):
    """
    lambda \cdot nuclear_dipole
    """
    if origin is None:
        origin = np.zeros(3)
    if isinstance(c_lambda, list):
        c_lambda = np.array(c_lambda)

    charges = mol.atom_charges()
    # the subtraction is along the common axis, and already in bohr
    coords  = np.subtract(mol.atom_coords(), origin)
    nuc_dip = np.einsum('i,ix->x', charges, coords)
    return np.einsum('x,...x->...', nuc_dip, c_lambda)


def get_multipole_matrix(mol, itype='dipole', c_lambda=None, origin=None):
    """
    c_lambda: (n_mode, 3) = coupling_strength * sqrt(2.*photon_frequency)
    """
    if origin is None:
        origin = np.zeros(3)
    if isinstance(c_lambda, list):
        c_lambda = np.array(c_lambda)

    multipoles = {}
    with mol.with_common_orig(origin):
        if 'dipole' in itype:
            dipole = mol.intor('int1e_r', comp=3, hermi=0)
            if isinstance(c_lambda, np.ndarray):
                dipole = np.einsum('xpq,...x->...pq', dipole, c_lambda)
            multipoles['dipole'] = dipole

        if 'quadrupole' in itype:
            nao = mol.nao_nr()
            quadrupole = mol.intor('int1e_rr', comp=9, hermi=0).reshape(3,3,nao,nao)
            if isinstance(c_lambda, np.ndarray):
                c2 = np.einsum('...x,...y->...xy', c_lambda, c_lambda)
                if c2.ndim == 3: # contract modes
                    c2 = np.sum(c2, axis=0)
                quadrupole = np.einsum('xypq,xy->pq', quadrupole, c2)
            multipoles['quadrupole'] = quadrupole

    return multipoles


def get_dse_elec_nuc(dipole, nuc_dip): # c_lambda is included
    if isinstance(nuc_dip, float):
        return -nuc_dip * dipole
    else: # numpy does not sum over ellipsis
        return -np.einsum('lpq,l->pq', dipole, nuc_dip) # l is the number of photon modes


def get_energy_nuc_dip(nuc_dip):
    energy = .5 * np.dot(nuc_dip, nuc_dip)
    return energy


def get_dse_2e(dipole, den, with_j=False): # c_lambda is included
    if dipole.ndim == 2:
        vk = np.einsum('pq,rs,...qs->...pr', dipole, dipole, den)
        if with_j is False:
            return vk
        else:
            vj = np.einsum('pq,rs,...rs->...pq', dipole, dipole, den)
            return [vj, vk]
    else: # contract modes
        vk = np.einsum('ipq,irs,...qs->...pr', dipole, dipole, den)
        if with_j is False:
            return vk
        else:
            vj = np.einsum('ipq,irs,...rs->...pq', dipole, dipole, den)
            return [vj, vk]


def get_dse_2e_xyz(dipole, den): # xyz without coupling
    return np.einsum('xpq,yrs,...qs->...xypr', dipole, dipole, den)


def cal_dse_gs(mol, den, c_lambda, dipole=None, quadrupole=None,
               coherent_state=True):

    itype = ''
    if isinstance(dipole, np.ndarray):
        dipole = np.einsum('xpq,...x->...pq', dipole, c_lambda)
    else:
        itype = itype + 'dipole'
    if isinstance(quadrupole, np.ndarray):
        c2 = np.einsum('...x,...y->...xy', c_lambda, c_lambda)
        if c2.ndim == 3: # contract modes
            c2 = np.sum(c2, axis=0)
        quadrupole = np.einsum('xypq,xy->pq', quadrupole, c2)
    else:
        itype = itype + 'quadrupole'

    if 'pole' in itype:
        multipoles = get_multipole_matrix(mol, itype, c_lambda)
        dipole = multipoles.get('dipole', dipole)
        quadrupole = multipoles.get('quadrupole', quadrupole)

    if den.ndim != 2 and den.ndim != 3:
        # 2: restricted total density matrix
        # 3: alpha and beta density matrices
        raise ValueError('wrong density matrix')

    if coherent_state:
        vk = get_dse_2e(dipole, den, with_j=False)

        dse = np.zeros(3)
        dse[0] = .5* np.einsum('pq,...pq->', quadrupole, den) # need 1/2 for dse here
        dse[1] = -.5* np.einsum('...pq,...pq->', vk, den) # need 1/2 for dse here
        if den.ndim == 2:
            dse[1] *= .5 # remove exchange for different spins

    else:
        nuc_dip = get_nuclear_dipoles(mol, c_lambda)
        nuc_dip_fock = get_dse_elec_nuc(dipole, nuc_dip)
        vj, vk = get_dse_2e(dipole, den, with_j=True)

        dse = np.zeros(6)
        dse[0] = .5* np.einsum('pq,...pq->', quadrupole, den) # need 1/2 for dse here
        dse[1] = np.einsum('pq,...pq->', nuc_dip_fock, den)
        dse[2] = .5* np.einsum('...pq,...pq->', vj, den) # need 1/2 for dse here
        dse[3] = -.5* np.einsum('...pq,...pq->', vk, den) # need 1/2 for dse here
        dse[4] = get_energy_nuc_dip(nuc_dip)
        if den.ndim == 2:
            dse[3] *= .5 # remove exchange for different spins

    dse[-1] = np.sum(dse)
    return dse



from pyscf import dft
from pyscf.dft.rks import RKS
class polariton(RKS):
    """
    QED-RKS ground state, independent of photon frequency
    """
    def get_multipole_matrix(self, c_lambda):
        mol = self.mol
        multipoles = get_multipole_matrix(mol, 'dipole_quadrupole', c_lambda)
        self.c_lambda = c_lambda
        self.dipole = multipoles['dipole']
        self.quadrupole = multipoles['quadrupole']

        self.get_exchange_factor()


    def get_exchange_factor(self):
        # the dse_2e term is added to exchange integral in fock
        # so we need to correct the hyb factor for dse_2e
        dft_type = 0
        if self.xc != 'wb97xd':
            dft_type = dft.xcfun.parse_xc(self.xc)
        else:
            dft_type = dft.libxc.parse_xc(self.xc)
            omega, alpha, hyb = self._numint.rsh_and_hybrid_coeff(self.mf.xc)
            dft_type[0][:] = [hyb, alpha, omega]
        hyb, alpha, omega = dft_type[0]
        self.hyb = hyb



class polariton_cs(polariton):
    """
    in photon coherent states
    """
    def get_hcore(self, mol=None):
        h = super().get_hcore(mol) # from RKS class
        h += .5* self.quadrupole # need 1/2 for dse
        return h


    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if dm is None: dm = self.make_rdm1()
        vj, vk = super().get_jk(mol, dm, hermi, with_j, with_k, omega)
        vk += get_dse_2e(self.dipole, dm, with_j=False) / self.hyb # need 1/2 for dse
        return vj, vk



class polariton_ns(polariton):
    """
    in photon number states
    """
    def get_hcore(self, mol=None):
        h = super().get_hcore(mol) # from RKS class
        h += .5* self.quadrupole # need 1/2 for dse
        self.nuc_dip = get_nuclear_dipoles(self.mol, self.c_lambda)
        h += get_dse_elec_nuc(self.dipole, self.nuc_dip)
        return h


    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if dm is None: dm = self.make_rdm1()
        vj, vk = super().get_jk(mol, dm, hermi, with_j, with_k, omega)
        dse_j, dse_k = get_dse_2e(self.dipole, dm, with_j=True)
        vj += dse_j
        vk += dse_k / self.hyb
        return vj, vk


    def energy_tot(self, dm=None, h1e=None, vhf=None):
        e_tot = super().energy_tot(dm, h1e, vhf)
        e_tot += get_energy_nuc_dip(self.nuc_dip)
        return e_tot



def print_qed_dse_energy(coupling, e0, e1, e_tot, unit='ev'):
    # e0: dse energy with gas-phase density
    # e1: dse energy with qed-ks density
    # e_tot: gas-phase and qed-ks total energy
    print('coupling:', end=' ')
    if isinstance(coupling, float):
        print('%7.5f' % coupling, end=' ')
    else:
        for i in range(len(coupling)):
            print('%7.5f' % coupling[i], end=' ')
    for i, e in enumerate([e0, e1]):
        print(' dse'+str(i)+':', end='')
        for i in range(len(e)):
            print('%11.5f' % e[i], end=' ')
    print(' polariton: %11.5f %s' % (e_tot[1]-e_tot[0], unit))



if __name__ == '__main__':
    #infile = 'h2o.in'
    #parameters = parser(infile)

    #charge, spin, atom = parameters.get(section_names[0])[1:4]
    #functional, basis = get_rem_info(parameters.get(section_names[1]))[:2]
    #mol = build_single_molecule(charge, spin, atom, basis, verbose=0)

    atom = sys.argv[1]
    h2 = """
            H    0. 0. -0.373
            H    0. 0.  0.373
    """
    hf = """
            H    0. 0. -0.459
            F    0. 0.  0.459
    """
    lif = """
           Li    0. 0. -0.791
            F    0. 0.  0.791
    """
    functional = 'pbe0'
    mol = build_single_molecule(0, 0, locals()[atom], '6-311++g**')
    mf = scf.RKS(mol)

    mf.xc = functional
    mf.grids.prune = True
    e_tot0 = mf.kernel()
    nocc = mol.nelectron // 2

    den = mf.make_rdm1()
    dipole, quadrupole = get_multipole_matrix(mol, 'dipole_quadrupole')

    frequency = 0.42978 # gs doesn't depend on frequency

    #coherent_state = False
    coherent_state = True

    dse = []
    for c in np.linspace(0, 10, 21): # better to use integer here
        for x in range(2, 3):
            coupling = np.zeros(3)
            coupling[x] = c*1e-2
            e0 = cal_dse_gs(mol, den, coupling, dipole, quadrupole, coherent_state)
            e0 = convert_units(e0, 'hartree', 'ev')

            if coherent_state:
                mf1 = polariton_cs(mol) # in coherent state
            else:
                mf1 = polariton_ns(mol) # in number (Fock) state

            #mf1.verbose = 10
            mf1.xc = functional
            mf1.grids.prune = True
            mf1.get_multipole_matrix(coupling)
            e_tot = mf1.kernel()#(dm0=den)

            e1 = cal_dse_gs(mol, mf1.make_rdm1(), coupling, dipole, quadrupole, coherent_state)
            e1 = convert_units(e1, 'hartree', 'ev')

            e_tot = np.array([e_tot0, e_tot])
            e_tot = convert_units(e_tot, 'hartree', 'ev')

            print_qed_dse_energy(coupling[x], e0, e1, e_tot)
