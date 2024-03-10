import sys
import numpy as np

import pyscf
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft.rks import RKS

from wavefunction_analysis.utils.pyscf_parser import *
from wavefunction_analysis.utils import convert_units, print_matrix
from wavefunction_analysis.polariton.qed_ks import polariton_cs
from wavefunction_analysis.polariton.qed_thermo import harmonic_analysis


def infrared_spectrum(results, dip_dev):
    pass


if __name__ == '__main__':
    #infile = 'h2o.in'
    #parameters = parser(infile)

    #charge, spin, atom = parameters.get(section_names[0])[1:4]
    #functional, basis = get_rem_info(parameters.get(section_names[1]))[:2]
    #mol = build_single_molecule(charge, spin, atom, basis, verbose=0)

    hf = """
            H    0. 0. -0.459
            F    0. 0.  0.459
    """
    functional = 'pbe0'
    mol = build_single_molecule(0, 0, hf, '3-21g')

    frequency = 0.42978 # gs doesn't depend on frequency
    coupling = np.array([0, 0, .05])

    mf = polariton_cs(mol) # in coherent state
    mf.xc = functional
    mf.grids.prune = True
    mf.get_multipole_matrix(coupling)

    e_tot = mf.kernel()
    de2 = mf.Hessian().kernel()
    results = harmonic_analysis(mol, de2)

    dip_dev = mf.get_dip_dev()
    infrared_spectrum(results, dip_dev)
