import os, sys
import numpy as np

import pyscf
from pyscf import scf, tdscf

from wavefunction_analysis.utils.pyscf_parser import *
from wavefunction_analysis.entanglement.fragment_entangle import get_embedding_system


if __name__ == '__main__':
    infile = '../samples/formic_acid_6_h2o.in'
    if len(sys.argv)>1: infile = sys.argv[1]
    parameters = parser(infile)
    results = run_pyscf_final(parameters)
    mol, mf = results['mol'], results['mf']
    print_matrix('mo_energy:', mf.mo_energy)

    frgm_idx = get_frgm_idx(parameters)
    print('frgm_idx:', frgm_idx)

    e, nocc_eo, eo_energy, coeff_eo_in_ao = get_embedding_system(mol, mf, frgm_idx)
    mo_occ = np.zeros(len(eo_energy))
    mo_occ[:nocc_eo//2] = 2
    print_matrix('eo_energy:', eo_energy)

    mf_emb = scf.RHF(mol)
    mf_emb.mo_energy = eo_energy
    mf_emb.mo_coeff  = coeff_eo_in_ao
    mf_emb.mo_occ    = mo_occ

    nroots = 3
    td_emb = tdscf.TDA(mf_emb)
    td_emb.kernel(nstates=nroots)
    print('td_emb converged:', td_emb.converged)
    print_matrix('td_emb energy:', td_emb.e)
    #td_emb.analyze()

    td = tdscf.TDA(mf)
    td.kernel(nstates=nroots)
    print('td_emb converged:', td.converged)
    print_matrix('td energy:', td.e)
    #td.analyze()
