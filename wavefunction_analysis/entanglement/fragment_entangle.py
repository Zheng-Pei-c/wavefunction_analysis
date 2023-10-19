import os, sys
import numpy as np
np.set_printoptions(precision=4, linewidth=200, suppress=True)

import pyscf
from pyscf import lo, scf, ao2mo
from pyscf.scf.hf import dot_eri_dm

from wavefunction_analysis.utils.pyscf_parser import *
from wavefunction_analysis.entanglement.mol_lo_tools import partition_lo_to_imps
from wavefunction_analysis.riemannian_manifold.optimization_grassmann import get_orthogonal_basis

def get_localized_orbital(mol, coeff, method='pipek_mezey'):
    if method == 'pipek_mezey':
        pm = lo.PM(mol, coeff)
        pm.conv_tol = 1e-8 # need high resolution
        return pm.kernel()
    elif method == 'cholesky':
        return lo.cholesky.cholesky_mos(coeff)


def get_localized_orbital_rdm(coeff_lo_in_ao, coeff_mo_in_ao, ovlp_ao, nocc, scale=2.):
    """
    total density matrix alpha+beta
    """
    coeff_lo_in_mo = np.einsum('...pi,pq,...qj->...ij', coeff_lo_in_ao, ovlp_ao, coeff_mo_in_ao)
    dm_lo_in_ao = np.einsum('...ik,...jk->...ij', coeff_lo_in_mo[:,:nocc], coeff_lo_in_mo[:,:nocc])

    return scale * dm_lo_in_ao


def get_embedding_orbital(dm_lo_in_ao, coeff_lo_in_ao, ovlp_ao,
        imp_lo_idx, env_lo_idx, method=0):
    threshold = 1e-12
    if method == 0: # singular value vectors of off-diagonal block of the dm in lo
        dm_imp_env_lo = dm_lo_in_ao[np.ix_(imp_lo_idx, env_lo_idx)] # get environmental orbitals
        _, s, Vt = np.linalg.svd(dm_imp_env_lo, full_matrices=False)
        print_matrix('singular values:', s)
        V = Vt[s>threshold].T
    elif method == 1: # eigenvectors of environment diagonal block of the dm in lo
        dm_env_env_lo = dm_lo_in_ao[np.ix_(env_lo_idx, env_lo_idx)]
        s, V = np.linalg.eigh(dm_env_env_lo)
        print_matrix('eigen-values:', s)
        V = V[:, (s>threshold)&(s<2.-threshold)]
    print(V.shape)
    #print_matrix('V:', V, 10)

    coeff_imp = np.copy(coeff_lo_in_ao[:, imp_lo_idx]) # idensity transformation
    coeff_env = np.einsum('pi,ij->pj', coeff_lo_in_ao[:, env_lo_idx], V)
    coeff_eo_in_ao = np.concatenate((coeff_imp, coeff_env), axis=1)
    #print_matrix('coeff_eo_in_ao:', coeff_eo_in_ao, 10)

    coeff_eo_in_lo = np.einsum('pi,pq,qj->ij', coeff_lo_in_ao, ovlp_ao, coeff_eo_in_ao)
    dm_eo_in_ao = np.einsum('pi,pq,qj->ij', coeff_eo_in_lo, dm_lo_in_ao, coeff_eo_in_lo)

    return coeff_eo_in_ao, dm_eo_in_ao


def get_embedding_energy(mol, mf, coeff_eo_in_ao, dm_eo_in_ao, neo_imp):
    """
    electronic energy of the impurity in the embedding basis
    """
    neo = dm_eo_in_ao.shape[1]
    nocc_eo = int(np.round(np.trace(dm_eo_in_ao))) # round up to an integer
    print('total %d electrons in %d embedding (%d impurity + %d bath) orbitals'
           % (nocc_eo, neo, neo_imp, neo-neo_imp))

    hcore_ao = mf.get_hcore()
    fock_ao  = mf.get_fock()

    #ovlp_ao = mf.get_ovlp()
    #Z, L, _ = get_orthogonal_basis(ovlp_ao)
    #fock_orth = np.einsum('pq,qr,sr->ps', Z, fock_ao, Z)
    #e, v = np.linalg.eigh(fock_orth)
    #print_matrix('mo energy:', e)

    h1e_eo = np.einsum('pi,pq,qj->ij', coeff_eo_in_ao, hcore_ao, coeff_eo_in_ao)
    f1e_eo = np.einsum('pi,pq,qj->ij', coeff_eo_in_ao, fock_ao, coeff_eo_in_ao)

    energy = np.einsum('pq,pq->', (h1e_eo+f1e_eo)[:neo_imp], dm_eo_in_ao[:neo_imp])

    # do we need to use the ``core'' electrons at all?
    #eri_eo = ao2mo.kernel(mol, coeff_eo_in_ao, 4, 'eri')
    #eri_eo = ao2mo.restore(1, eri_eo, neo)
    #j1e_eo, k1e_eo = dot_eri_dm(eri_eo, dm_eo_in_ao, hermi=1, with_j=True, with_k=True)

    #f1e_eo -= (j1e_eo - k1e_eo * .5)
    # end of ``core'' electron contribution

    return energy*.5, nocc_eo


def get_embedding_orbital_energy(mf, coeff_eo_in_ao):
    """
    embedding orbital energy for excited state calculations
    """
    mo_energy = mf.mo_energy
    ovlp_ao = mf.get_ovlp()
    Z, L, _ = get_orthogonal_basis(ovlp_ao)
    #fock_orth = np.einsum('pq,qr,sr->ps', Z, fock_ao, Z)
    #e, v = np.linalg.eigh(fock_orth)
    #print_matrix('mo energy:', e)

    coeff_eo_canon = np.einsum('pq,qi->pi', L, coeff_eo_in_ao)
    eo_energy = np.einsum('pi,p,pi->i', coeff_eo_canon, mo_energy, coeff_eo_canon)
    idx = np.argsort(eo_energy)
    return eo_energy[idx], coeff_eo_in_ao[:, idx]

    #ao_slice_by_atom = mol.aoslice_by_atom()[:,2:4]
    #print('ao_slice_by_atom:\n', ao_slice_by_atom)


def get_embedding_system(mol, mf, frgm_idx, ifrgm=0):
    nocc = mol.nelectron // 2
    ovlp_ao = mf.get_ovlp()
    coeff_mo_in_ao = mf.mo_coeff

    # local orbital depends on the localization method
    coeff_lo_in_ao = get_localized_orbital(mol, coeff_mo_in_ao)
    dm_lo_in_ao = get_localized_orbital_rdm(coeff_lo_in_ao, coeff_mo_in_ao, ovlp_ao, nocc)

    frgm_lo_idx = partition_lo_to_imps(frgm_idx, mol, coeff_lo_in_ao, min_weight=0.8)

    def embedding(ifrgm=0, embed_method=0):
        imp_lo_idx = frgm_lo_idx.copy()
        imp_lo_idx, env_lo_idx = np.array(imp_lo_idx.pop(ifrgm)), np.sort(np.concatenate(imp_lo_idx))
        neo_imp = len(imp_lo_idx)
        print('imp_lo_idx:', imp_lo_idx)
        print('env_lo_idx:', env_lo_idx)

        coeff_eo_in_ao, dm_eo_in_ao = get_embedding_orbital(dm_lo_in_ao, coeff_lo_in_ao,
                ovlp_ao, imp_lo_idx, env_lo_idx, embed_method)
        e, nocc_eo = get_embedding_energy(mol, mf, coeff_eo_in_ao, dm_eo_in_ao, neo_imp)
        eo_energy, coeff_eo_in_ao = get_embedding_orbital_energy(mf, coeff_eo_in_ao)
        return e, nocc_eo, eo_energy, coeff_eo_in_ao

    if ifrgm >=0:
        return embedding(ifrgm)

    energy = 0
    for f in range(len(frgm_idx)):
        e = embedding(f)[0]
        energy += e

    energy_ref = mf.energy_elec()[0]
    print('energy:', energy_ref, energy, energy_ref-energy)



if __name__ == '__main__':
    #infile = '../samples/formic_acid_6_h2o.in'
    #infile = sys.argv[1]
    #parameters = parser(infile)
    #results = run_pyscf_final(parameters)
    #mol, mf = results['mol'], results['mf']

    #natm = mol.natm
    #frgm_idx = parameters[section_names[1]]['impurity'].split('-')
    #frgm_idx = [list(range(int(frgm_idx[0])-1, int(frgm_idx[1]))), [0]]
    #frgm_idx[1] = list(set(range(natm)) - set(frgm_idx[0]))
    ##print('frgm_idx:', frgm_idx)

    mol = gto.Mole()
    mol.build(
        atom = """
           O         0.4183272099    0.1671038379    0.1010361156
           H         0.8784893276   -0.0368266484    0.9330933285
           H        -0.3195928737    0.7774121014    0.3045311682
           O         3.0208058979    0.6163509592   -0.7203724735
           H         3.3050376617    1.4762564664   -1.0295977027
           H         2.0477791789    0.6319690134   -0.7090745711
           O         2.5143150551   -0.2441947452    1.8660305097
           H         2.8954132119   -1.0661605274    2.1741344071
           H         3.0247679096    0.0221180670    1.0833062723
        """,
        basis = 'sto-3g',
        verbose=0
    )

    frgm_idx = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.max_cycle = 100
    mf.conv_tol  = 1e-8
    mf.conv_tol_grad = 1e-8
    mf.kernel()
    print_matrix('mo_energy:', mf.mo_energy)

    get_embedding_system(mol, mf, frgm_idx, -1)