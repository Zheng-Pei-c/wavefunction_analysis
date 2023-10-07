import os, sys
import numpy as np

import pyscf

from wavefunction_analysis.utils.pyscf_parser import *
from wavefunction_analysis.entanglement.fragment_entangle import get_embedding_system


if __name__ == '__main__':
    infile = '../samples/formic_acid_6_h2o.in'
    if len(sys.argv)>1: infile = sys.argv[1]
    parameters = parser(infile)
    results = run_pyscf_final(parameters)
    mol, mf = results['mol'], results['mf']


    frgm_idx = get_frgm_idx(parameters)
    print('frgm_idx:', frgm_idx)

    e, nocc_eo, eo_energy, coeff_eo_canon = get_embedding_system(mol, mf, frgm_idx)
