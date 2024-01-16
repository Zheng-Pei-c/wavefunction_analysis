import os, sys
import numpy as np

import pyscf

from wavefunction_analysis.utils.pyscf_parser import *
from wavefunction_analysis.utils import convert_units, print_matrix

def get_c_lambda(coupling_strength, frequency):
    if isinstance(frequency, float):
        frequency = [frequency]
    return np.einsum('x,i->ix', coupling_strength, np.sqrt(frequency)/np.sqrt(2.))


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


def get_dse_2e(dipole, den): # c_lambda is included
    if dipole.ndim == 2:
        return np.einsum('pq,rs,...qs->...pr', dipole, dipole, den)
    else: # contract modes
        return np.einsum('ipq,irs,...qs->...pr', dipole, dipole, den)


def get_dse_2e_xyz(dipole, den): # xyz without coupling
    return np.einsum('xpq,yrs,...qs->...xypr', dipole, dipole, den)


def cal_dse_gs(mol, den, c_lambda, dipole=None, quadrupole=None):

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
        if 'dipole' in itype:
            dipole = multipoles.get('dipole', dipole)
        if 'quadrupole' in itype:
            quadrupole = multipoles.get('quadrupole', quadrupole)

    if den.ndim == 2: # assume restricted total density
        quadrupole -= .5* get_dse_2e(dipole, den)
        dse = .5* np.einsum('pq,pq->', quadrupole, den) # need 1/2 for dse here
    else: # alpha and beta density matrices
        dse = .5* np.einsum('pq,npq->', quadrupole, den)
        dse -= .5* np.einsum('npq,npq->', get_dse_2e(dipole, den), den)

    return dse



from pyscf.dft.rks import RKS
class polariton(RKS):
    def get_multipole_matrix(self, c_lambda):
        mol = self.mol
        multipoles = get_multipole_matrix(mol, 'dipole_quadrupole', c_lambda)
        self.dipole = multipoles['dipole']
        self.quadrupole = multipoles['quadrupole']


    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol

        h = mol.intor_symmetric('int1e_kin')
        h += mol.intor_symmetric('int1e_nuc')
        h += .5* self.quadrupole # need 1/2 for dse
        return h


    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if dm is None: dm = self.make_rdm1()
        vj, vk = super().get_jk(mol, dm, hermi, with_j, with_k, omega)
        vk += .5* get_dse_2e(self.dipole, dm) # need 1/2 for dse
        return vj, vk



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
    mf.kernel()
    nocc = mol.nelectron // 2

    den = mf.make_rdm1()
    dipole, quadrupole = get_multipole_matrix(mol, 'dipole_quadrupole')

    frequency = 0.42978 # gs doesn't depend on frequency

    dse = []
    for c in np.linspace(0, 10, 21): # better to use integer here
        for x in range(2, 3):
            coupling = np.zeros(3)
            coupling[x] = c*1e-2
            #c_lambda = get_c_lambda(coupling, frequency)
            e = cal_dse_gs(mol, den, coupling, dipole, quadrupole)
            dse.append(convert_units(e, 'hartree', 'ev'))

            mf1 = polariton(mol)
            mf1.xc = functional
            mf1.grids.prune = True
            mf1.get_multipole_matrix(coupling)
            mf1.kernel()
            e1 = mf1.energy_elec()[0] - mf.energy_elec()[0]
            e1 = convert_units(e1, 'hartree', 'ev')

            e2 = cal_dse_gs(mol, mf1.make_rdm1(), coupling, dipole, quadrupole)
            e2 = convert_units(e2, 'hartree', 'ev')

            print('coupling: %8.5f  dse: %8.5f eV  polariton: %8.5f eV  dse2: %8.5f eV' % (coupling[x], dse[-1], e1, e2))
            print(np.diag(np.einsum('pi,pq,qj->ij', mf.mo_coeff, mf.get_ovlp(), mf1.mo_coeff)))
