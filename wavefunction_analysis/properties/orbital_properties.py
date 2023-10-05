import os, sys
import numpy as np

import pyscf
from pyscf import scf, tdscf, gto, lib

from utils import convert_units
from utils.pyscf_parser import *
from utils.unit_conversion import EV2J, EMass

def cal_orbital_properties(mol, mf):
    orbital_properties = {}
    units = {}

    nbas, nocc, nvir, nov = get_basis_info(mol)
    coeff = mf.mo_coeff[:,:nocc]
    energy = mf.mo_energy[:nocc]
    #print_matrix('mo energy occ:', energy[:nocc])
    #print_matrix('mo energy:', energy)

    # these two approaches give same values
    #time = 2.*np.pi / (-energy)
    #time = convert_units(time, 'au', 'fs')
    #print('time:', time)
    time = -convert_units(energy, 'hartree', 'fs') # it is period
    print_matrix('time:', time)
    orbital_properties['time'] = time
    units['time'] = 'fs'

    length = mol.intor('int1e_r', comp=3)
    length = np.einsum('mi,xmn,ni->ix', coeff, length, coeff)
    print_matrix('length:', length)
    length = convert_units(length, 'bohr', 'aa')
    length = np.linalg.norm(length, axis=1)
    orbital_properties['length'] = length
    units['length'] = 'aa'

    length2 = mol.intor('int1e_rr')
    length2 = np.einsum('mi,xmn,ni->ix', coeff, length2, coeff)
    length2 = convert_units(length2, 'bohr', 'aa')
    orbital_properties['length2'] = length2
    units['length2'] = 'aa'

    velocity = 2.*convert_units(length, units['length'], 'nm') / convert_units(time, units['time'], 'ns')
    orbital_properties['velocity'] = velocity
    units['velocity'] = 'm/s'
    print_matrix('velocity:', velocity)

    kinetic = mol.intor('int1e_kin')
    kinetic = np.einsum('mi,mn,ni->i', coeff, kinetic, coeff)
    kinetic = convert_units(kinetic, 'hartree', 'ev')
    orbital_properties['kinetic'] = kinetic
    units['kinetic'] = 'ev'

    velocity = np.sqrt(2.*kinetic/EV2J/EMass)
    orbital_properties['velocity2'] = velocity
    units['velocity2'] = 'm/s'

    momentum = -mol.intor('int1e_ipovlp', comp=3)
    momentum = np.einsum('mi,xmn,ni->ix', coeff, momentum, coeff)
    #print('momentum:', momentum)
    orbital_properties['velocity3'] = np.linalg.norm(momentum/convert_units(EMass, 'kg', 'dalton'), axis=1)
    units['velocity3'] = 'm/s'

    energy = convert_units(energy, 'hartree', 'ev')
    orbital_properties['energy'] = energy
    units['energy'] = 'eV'

    orbital_properties['units'] = units
    return orbital_properties


def cal_properties_from_scratch(infile):
    parameters = parser(infile)
    mol, mf = run_pyscf_final(parameters)
    return cal_orbital_properties(mol, mf)


if __name__ == '__main__':
    infile = sys.argv[1]
    orbital_properties = cal_properties_from_scratch(infile)
