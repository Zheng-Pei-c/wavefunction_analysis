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
    lambda cdot nuclear_dipole
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


def get_multipole_matrix(mol, itype='dipole', dipole=None, quadrupole=None,
                         c_lambda=None, origin=None):
    """
    c_lambda: (n_mode, 3) = coupling_strength * sqrt(2.*photon_frequency)
    """
    if origin is None:
        origin = np.zeros(3)
    if isinstance(c_lambda, list):
        c_lambda = np.array(c_lambda)

    if itype == 'all':
        if not isinstance(dipole, np.ndarray):
            itype += '_dipole'
        if not isinstance(quadrupole, np.ndarray):
            itype += '_quadrupole'

    with mol.with_common_orig(origin):
        if 'dipole' in itype:
            dipole = mol.intor('int1e_r', comp=3, hermi=0)

        if 'quadrupole' in itype:
            nao = mol.nao_nr()
            quadrupole = mol.intor('int1e_rr', comp=9, hermi=0).reshape(3,3,nao,nao)

    if isinstance(c_lambda, np.ndarray):
        if isinstance(dipole, np.ndarray):
            dipole = np.einsum('xpq,...x->...pq', dipole, c_lambda)

        if isinstance(quadrupole, np.ndarray):
            c2 = np.einsum('...x,...y->...xy', c_lambda, c_lambda)
            if c2.ndim == 3: # contract modes
                c2 = np.sum(c2, axis=0)
            quadrupole = np.einsum('xypq,xy->pq', quadrupole, c2)

    multipoles = {'dipole': dipole, 'quadrupole': quadrupole}
    return multipoles


def get_dse_elec_nuc(dipole, nuc_dip): # c_lambda is included
    if isinstance(nuc_dip, float):
        return -nuc_dip * dipole
    else: # numpy does not sum over ellipsis
        return -np.einsum('lpq,l->pq', dipole, nuc_dip) # l is the number of photon modes


def get_energy_nuc_dip(nuc_dip):
    energy = .5 * np.dot(nuc_dip, nuc_dip)
    return energy


def get_dse_2e(dipole, dm, with_j=False, scale_k=.5): # c_lambda is included
    # scale k by 1/2 for restricted orbital case by default
    if dipole.ndim == 2:
        vk = np.einsum('pq,rs,...qs->...pr', dipole, dipole, dm)
        if with_j is False:
            return vk*scale_k
        else:
            vj = np.einsum('pq,rs,...rs->...pq', dipole, dipole, dm)
            return [vj, vk*scale_k]
    else: # contract modes
        vk = np.einsum('ipq,irs,...qs->...pr', dipole, dipole, dm)
        if with_j is False:
            return vk*scale_k
        else:
            vj = np.einsum('ipq,irs,...rs->...pq', dipole, dipole, dm)
            return [vj, vk*scale_k]


def get_dse_2e_xyz(dipole, dm): # xyz without coupling
    return np.einsum('xpq,yrs,...qs->...xypr', dipole, dipole, dm)



from pyscf import lib, dft
from pyscf.lib import logger
from pyscf.dft.rks import RKS
class polariton(RKS):
    """
    QED-RKS ground state, independent of photon frequency
    """
    def get_multipole_matrix(self, c_lambda, dipole=None, quadrupole=None):
        multipoles = get_multipole_matrix(self.mol, 'all', dipole, quadrupole, c_lambda=c_lambda)
        self.c_lambda = c_lambda
        self.dipole = multipoles['dipole']
        self.quadrupole = multipoles['quadrupole']



class polariton_cs(polariton):
    """
    in photon coherent states
    """
    def get_hcore(self, mol=None):
        h = super().get_hcore(mol) # from RKS class
        hquad = .5* self.quadrupole # need 1/2 for dse
        h += hquad
        h = lib.tag_array(h, hquad=hquad)
        return h


    def get_veff(self, mol=None, dm=None, *args, **kwargs):
        if dm is None: dm = self.make_rdm1()
        vxc = super().get_veff(mol, dm, *args, **kwargs)
        vdse_k = get_dse_2e(self.dipole, dm, with_j=False) # need 1/2 for dse
        edse_k = -.5* np.einsum('...pq,...pq->', vdse_k, dm) # dse exchange energy

        # old tags are destroyed after the number operations
        ecoul, exc, vj, vk = vxc.ecoul, vxc.exc, vxc.vj, vxc.vk
        vxc -= vdse_k
        vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk, edse_k=edse_k, vdse_k=vdse_k)
        return vxc


    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None: dm = self.make_rdm1()
        if h1e is None: h1e = self.get_hcore()
        if vhf is None or getattr(vhf, 'edse_k', None) is None:
            vhf = self.get_veff(self.mol, dm)

        e_tot, e2 = super().energy_elec(dm, h1e, vhf)

        equad = np.einsum('pq,...pq->', h1e.hquad, dm)
        self.scf_summary['equad'] = equad.real
        self.scf_summary['edse_k'] = vhf.edse_k.real
        logger.debug(self, 'Quadrupole Energy = %s  DSE-K Energy = %s', equad, vhf.edse_k)
        return e_tot, e2


    def get_coupling_energy(self, dm=None, unit='ev'):
        if isinstance(dm, np.ndarray):
            self.energy_tot(dm=dm)

        e = self.scf_summary
        e = [e['equad'], e['edse_k']]
        e.append(e[0] + e[1])
        return convert_units(np.array(e), 'hartree', unit)



class polariton_ns(polariton):
    """
    in photon number states, not recommended!
    """
    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        h = super().get_hcore(mol) # from RKS class
        hquad = .5* self.quadrupole # need 1/2 for dse
        self.nuc_dip = get_nuclear_dipoles(mol, self.c_lambda)
        hdipe = get_dse_elec_nuc(self.dipole, self.nuc_dip)

        h += (hquad + hdipe)
        h = lib.tag_array(h, hquad=hquad, hdipe=hdipe)
        return h


    def get_veff(self, mol=None, dm=None, *args, **kwargs):
        if dm is None: dm = self.make_rdm1()
        vxc = super().get_veff(mol, dm, *args, **kwargs)
        vdse_j, vdse_k = get_dse_2e(self.dipole, dm, with_j=True)
        edse_j = .5* np.einsum('...pq,...pq->', vdse_j, dm) # need 1/2 for dse here
        edse_k = -.5* np.einsum('...pq,...pq->', vdse_k, dm) # need 1/2 for dse here

        ecoul, exc, vj, vk = vxc.ecoul, vxc.exc, vxc.vj, vxc.vk
        vxc += (vdse_j - vdse_k)
        vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk, edse_j=edse_j, edse_k=edse_k, vdse_j=vdse_j, vdse_k=vdse_k)
        return vxc


    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None: dm = self.make_rdm1()
        if h1e is None: h1e = self.get_hcore()
        if vhf is None or getattr(vhf, 'edse_k', None) is None:
            vhf = self.get_veff(self.mol, dm)

        e_tot, e2 = super().energy_elec(dm, h1e, vhf)

        equad = np.einsum('pq,...pq->', h1e.hquad, dm)
        edipe = np.einsum('pq,...pq->', h1e.hdipe, dm)
        self.scf_summary['equad'] = equad.real
        self.scf_summary['edipe'] = edipe.real
        self.scf_summary['edse_j'] = vhf.edse_j.real
        self.scf_summary['edse_k'] = vhf.edse_k.real
        logger.debug(self, 'Quadrupole Energy = %s  Nuclear-Electronic Dipole Energy = %s  DSE-J Energy = %s  DSE-K Energy = %s', equad, edipe, vhf.edse_j, vhf.edse_k)
        return e_tot, e2


    def energy_nuc(self):
        enuc = super().energy_nuc()
        edipn = get_energy_nuc_dip(self.nuc_dip)

        self.scf_summary['edipn'] = edipn
        logger.debug(self, 'Nuclear Dipole Energy = %s', edipn)
        return (enuc+edipn)


    def get_coupling_energy(self, dm=None, unit='ev'):
        if isinstance(dm, np.ndarray):
            self.get_hcore() # get nuc_dip for the energy
            self.energy_tot(dm=dm)

        e = self.scf_summary
        e = [e['equad'], e['edipe'], e['edse_j'], e['edse_k'], e['edipn']]
        e.append(np.sum(e))
        return convert_units(np.array(e), 'hartree', unit)



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

    dm = mf.make_rdm1()
    multipoles = get_multipole_matrix(mol, itype='dipole_quadrupole')
    dipole, quadrupole = multipoles['dipole'], multipoles['quadrupole']

    frequency = 0.42978 # gs doesn't depend on frequency

    #coherent_state = False
    coherent_state = True

    dse = []
    for c in np.linspace(0, 10, 21): # better to use integer here
        for x in range(2, 3):
            coupling = np.zeros(3)
            coupling[x] = c*1e-2

            if coherent_state:
                mf1 = polariton_cs(mol) # in coherent state
            else:
                mf1 = polariton_ns(mol) # in number (Fock) state

            #mf1.verbose = 10
            mf1.xc = functional
            mf1.grids.prune = True
            mf1.get_multipole_matrix(coupling)

            e0 = mf1.get_coupling_energy(dm=dm)
            e_tot = mf1.kernel()#(dm0=dm)
            e1 = mf1.get_coupling_energy()

            e_tot = np.array([e_tot0, e_tot])
            e_tot = convert_units(e_tot, 'hartree', 'ev')
            print_qed_dse_energy(coupling[x], e0, e1, e_tot)
