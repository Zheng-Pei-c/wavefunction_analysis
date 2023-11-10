import os, sys
import numpy as np

import pyscf

from wavefunction_analysis.utils.pyscf_parser import *
from wavefunction_analysis.utils import convert_units


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
                quadrupole = np.einsum('xypq,...x,...y->...pq', quadrupole, c_lambda, c_lambda)
            multipoles['quadrupole'] = quadrupole

        return multipoles


def get_dse_2e(dipole, den): # c_lambda is included
    return np.einsum('...pq,...rs,qs->...pr', dipole, dipole, den)


def get_dse_2e_xyz(dipole, den): # xyz
    return np.einsum('xpq,yrs,qs->xypr', dipole, dipole, den)


def cal_dse_gs(mol, den, c_lambda, dipole=None, quadrupole=None):
    itype = ''
    if isinstance(dipole, np.ndarray):
        dipole = np.einsum('xpq,...x->...pq', dipole, c_lambda)
    else:
        itype = itype + 'dipole'
    if isinstance(quadrupole, np.ndarray):
        quadrupole = np.einsum('xypq,...x,...y->...pq', quadrupole, c_lambda, c_lambda)
    else:
        itype = itype + 'quadrupole'

    if 'pole' in itype:
        multipoles = get_multipole_matrix(mol, itype, c_lambda)
        dipole = multipoles.get('dipole', dipole)
        quadrupole = multipoles.get('quadrupole', quadrupole)

    quadrupole -= .5* get_dse_2e(dipole, den)
    dse = .5* np.einsum('...pq,pq->', quadrupole, den) # need 1/2 for dse here
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
        h += .5* self.quadrupole
        return h


    #def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    #    if dm is None: dm = self.make_rdm1()
    #    vhf = super().get_veff(dm=dm)
    #    dse_2e = get_dse_2e(self.dipole, dm)
    #    vhf.vk += dse_2e
    #    #vhf.vxc -= .5* dse_2e
    #    return vhf


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
    atom = """
            H    0. 0. -0.373
            H    0. 0.  0.373
    """
    functional = 'pbe0'
    mol = build_single_molecule(0, 0, atom, '6-311++g**')
    mf = scf.RKS(mol)

    mf.xc = functional
    mf.grids.prune = True
    mf.kernel()

    den = mf.make_rdm1()
    dipole, quadrupole = get_multipole_matrix(mol, 'dipole_quadrupole')

    frequency = 0.42978

    dse = []
    for c in np.linspace(0, 10, 21): # better to use integer here
        for x in range(2, 3):
            coupling = np.zeros(3)
            coupling[x] = c*1e-2
            c_lambda = coupling * np.sqrt(2.*frequency)
            e = cal_dse_gs(mol, den, c_lambda, dipole, quadrupole)
            dse.append(convert_units(e, 'hartree', 'ev'))

            mf1 = polariton(mol)
            mf1.xc = functional
            mf1.grids.prune = True
            mf1.get_multipole_matrix(c_lambda)
            mf1.kernel()
            e1 = mf1.energy_elec()[0] - mf.energy_elec()[0]
            e1 = convert_units(e1, 'hartree', 'ev')

            print('coupling: %8.5f  dse: %8.5f eV  polariton: %8.5f eV' % (coupling[x], dse[-1], e1))
