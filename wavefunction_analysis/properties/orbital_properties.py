import os, sys
import numpy as np

import pyscf
from pyscf import scf, tdscf, gto, lib

from utils import convert_units
from utils.pyscf_parser import *

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

    length = mol.intor('int1e_r')
    length = np.einsum('mi,xmn,ni->ix', coeff, length, coeff)
    print_matrix('length:', length)
    length = convert_units(length, 'bohr', 'aa')
    length = np.abs(length[:,2])
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
