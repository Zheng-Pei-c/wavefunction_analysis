from wavefunction_analysis import sys, np
from wavefunction_analysis.utils import print_matrix
from wavefunction_analysis.embedding.mol_lo_tools import partition_lo_to_imps

def get_localized_orbital(mol, coeff, method='pipek_mezey'):
    from pyscf import lo
    r"""
    localized orbital basis C_lo = S^{-1/2}
    method: 'pipek_mezey', 'cholesky', 'lowdin', 'meta_lowdin'
    """
    if method == 'pipek_mezey':
        pm = lo.PM(mol, coeff)
        pm.conv_tol = 1e-8 # need high resolution
        return pm.kernel()
    elif method == 'cholesky':
        return lo.cholesky.cholesky_mos(coeff)
    elif 'lowdin' in method: #(meta_)lowdin
        return lo.orth.orth_ao(mol, method)


def get_localized_orbital_rdm(coeff_lo_in_ao, coeff_mo_in_ao, ovlp_ao, nocc, scale=2., extra_orb=0):
    r"""
    total density matrix alpha+beta in local orbital basis
    S^{1/2} P S^{1/2} = C_lo^T S C_mo occ C_mo occ^T S C_lo
    """
    # identity = np.einsum('...pi,pq,...qj->...ij', coeff_lo_in_ao, ovlp_ao, coeff_lo_in_ao)
    coeff_lo_in_mo = np.einsum('...pi,pq,...qj->...ij', coeff_lo_in_ao, ovlp_ao, coeff_mo_in_ao)
    dm_lo_in_ao = np.einsum('...ik,...jk->...ij', coeff_lo_in_mo[:,:nocc+extra_orb], coeff_lo_in_mo[:,:nocc+extra_orb])

    return scale * dm_lo_in_ao


def fragment_localization(mf, frgm_list, extra_orb=0):
    """
    get the localized orbitals and density matrix for fragment analysis
    """
    mol = mf.mol
    coeff_mo_in_ao = mf.mo_coeff
    ovlp_ao = mf.get_ovlp()

    nocc = mol.nelectron // 2
    if nocc > ovlp_ao.shape[0]-nocc: extra_orb = -extra_orb

    # local orbital depends on the localization method
    coeff_lo_in_ao = get_localized_orbital(mol, coeff_mo_in_ao)
    dm_lo_in_ao = get_localized_orbital_rdm(coeff_lo_in_ao, coeff_mo_in_ao,
                                            ovlp_ao, nocc,
                                            extra_orb=extra_orb)
    frgm_lo_idx = partition_lo_to_imps(frgm_list, mol=mol,
                                       coeff_ao_lo=coeff_lo_in_ao,
                                       min_weight=.8)

    return coeff_lo_in_ao, dm_lo_in_ao, frgm_lo_idx


def get_embedding_orbital(dm_lo_in_ao, coeff_lo_in_ao, ovlp_ao,
                          imp_lo_idx, env_lo_idx, method=0, threshold=1e-12):
    """
    embedding orbital construction from the density matrix in local orbital basis
    method=0: singular value vectors of off-diagonal block of the dm in lo
    method=1: eigenvectors of environment diagonal block of the dm in lo
    ----------------------------------------------------------------------
    returns:
    coeff_eo_in_ao: embedding orbital coefficients in ao basis
    dm_eo_in_ao: density matrix in embedding orbital basis
    """
    def embed_spin_orbital(dm_lo_in_ao, iprint=0):
        if method == 0: # singular value vectors of off-diagonal block of the dm in lo
            dm_imp_env_lo = dm_lo_in_ao[np.ix_(imp_lo_idx, env_lo_idx)] # get environmental orbitals
            _, s, Vt = np.linalg.svd(dm_imp_env_lo, full_matrices=False)
            if iprint > 0: print_matrix('singular values:'+str(np.sum(s)), s)
            V = Vt[s>threshold].T
        elif method == 1: # eigenvectors of environment diagonal block of the dm in lo
            dm_env_env_lo = dm_lo_in_ao[np.ix_(env_lo_idx, env_lo_idx)]
            s, V = np.linalg.eigh(dm_env_env_lo)
            if iprint > 0: print_matrix('eigen-values:', s)
            V = V[:, (s>threshold)&(s<2.-threshold)]
        return V

    nspin = dm_lo_in_ao.shape[0] if dm_lo_in_ao.ndim > 2 else 0
    V = []
    if nspin > 0:
        for i in range(nspin):
            V.append(embed_spin_orbital(dm_lo_in_ao[i], iprint=1))
        V = np.array(V)
    else:
        V = embed_spin_orbital(dm_lo_in_ao, iprint=1)
    print('V shape:', V.shape)

    coeff_imp = coeff_lo_in_ao[..., imp_lo_idx] # idensity transformation
    coeff_env = np.einsum('...pi,...ij->...pj', coeff_lo_in_ao[..., env_lo_idx], V)
    coeff_eo_in_ao = np.concatenate((coeff_imp, coeff_env), axis=-1)
    #print_matrix('coeff_eo_in_ao:', coeff_eo_in_ao, 10)
    # identity = np.einsum('...pi,pq,...qj->...ij', coeff_eo_in_ao, ovlp_ao, coeff_eo_in_ao)

    # a round-over approach to get dm_eo
    coeff_eo_in_lo = np.einsum('...pi,pq,...qj->...ij', coeff_lo_in_ao, ovlp_ao, coeff_eo_in_ao)
    # identity = np.einsum('...ij,...ik->...jk', coeff_eo_in_lo, coeff_eo_in_lo)
    dm_eo_in_ao = np.einsum('...pi,...pq,...qj->...ij', coeff_eo_in_lo, dm_lo_in_ao, coeff_eo_in_lo)

    # the straightforward way to get dm_eo but lenghthy
    #IV = np.concatenate((np.eye(len(imp_lo_idx)), V), axis=-1)
    #dm_lo_ii = dm_lo_in_ao[np.ix_(imp_lo_idx, imp_lo_idx)]
    #dm_lo_ie = dm_lo_in_ao[np.ix_(imp_lo_idx, env_lo_idx)]
    #dm_lo_ei = dm_lo_in_ao[np.ix_(env_lo_idx, imp_lo_idx)]
    #dm_lo_ee = dm_lo_in_ao[np.ix_(env_lo_idx, env_lo_idx)]
    #dm_lo = np.asarray(np.block([[dm_lo_ii, dm_lo_ie], [dm_lo_ei, dm_lo_ee]]))
    #dm_eo_in_ao = np.einsum('ji,jk,kl->il', IV, dm_lo, IV)
    #print_matrix('P:', dm_eo_in_ao, 10)
    #print('nelec:', np.trace(dm_eo_in_ao))

    return coeff_eo_in_ao, dm_eo_in_ao


def get_embedding_energy(mf, coeff_eo_in_ao, dm_eo_in_ao, neo_imp, extra_orb=0):
    """
    electronic energy of the impurity in the embedding basis
    """
    neo = dm_eo_in_ao.shape[1]
    # effective number of electrons in embedding space
    nocc_eo = int(round(np.trace(dm_eo_in_ao))) # round up to an integer
    nocc_eo -= extra_orb*2
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
    #from pyscf import ao2mo
    #from pyscf.scf.hf import dot_eri_dm
    #eri_eo = ao2mo.kernel(mf.mol, coeff_eo_in_ao, 4, 'eri')
    #eri_eo = ao2mo.restore(1, eri_eo, neo)
    #j1e_eo, k1e_eo = dot_eri_dm(eri_eo, dm_eo_in_ao, hermi=1, with_j=True, with_k=True)

    #f1e_eo -= (j1e_eo - k1e_eo * .5)
    # end of ``core'' electron contribution

    # embedding orbital and its orbital energy
    e_eo, v_eo = np.linalg.eigh(f1e_eo)
    coeff_eo_in_ao = np.einsum('pi,ij->pj', coeff_eo_in_ao, v_eo)

    return energy*.5, nocc_eo, e_eo, coeff_eo_in_ao


def get_embedding_system(mf, frgm_idx, ifrgm=0, extra_orb=0):
    """
    embedding energy calculation for a given fragment
    mf: pyscf mean-field object
    frgm_idx: list of list of atomic indices for each fragment
    ifrgm: index of the fragment to be treated as impurity; if -1, loop over all fragments
    extra_orb: number of extra orbitals to include in the density matrix (can be negative)
    --------------------------------------------------------------------------
    returns:
    energy: embedded energy of the impurity fragment
    """

    embed = EmbeddingMeanField(mf, frgm_idx, extra_orb=extra_orb)

    if ifrgm >= 0:
        coeff_eo_in_ao, dm_eo_in_ao = embed.emb_basis_dmet(mf, ifrgm)
        neo_imp = len(embed.imp_lo_idx) # number of impurity embedding orbitals
        return get_embedding_energy(mf, coeff_eo_in_ao, dm_eo_in_ao, neo_imp,
                                    extra_orb)

    energy = 0
    for f in range(len(frgm_idx)):
        coeff_eo_in_ao, dm_eo_in_ao = embed.emb_basis_dmet(mf, f)
        e = get_embedding_energy(mf, coeff_eo_in_ao, dm_eo_in_ao, neo_imp,
                                 extra_orb)[0]
        energy += e

    energy_ref = mf.energy_elec()[0]
    print('energy:', energy_ref, energy, energy_ref-energy)



class EmbeddingMeanField():
    def __init__(self, mf, frgm_list, extra_orb=0):
        """
        build localized orbital and density matrix and lo index
        """
        self.coeff_lo_in_ao, self.dm_lo_in_ao, self.frgm_lo_idx = fragment_localization(mf, frgm_list, extra_orb)


    def emb_basis_dmet(self, mf, ifrgm, embed_method=0):
        """
        build embedding basis for a given fragment
        """
        imp_lo_idx = self.frgm_lo_idx.copy()
        self.imp_lo_idx = np.array(imp_lo_idx.pop(ifrgm))
        self.env_lo_idx = np.sort(np.concatenate(imp_lo_idx))
        self.coeff_eo_in_ao, self.dm_eo_in_ao = get_embedding_orbital(
                                        self.dm_lo_in_ao,
                                        self.coeff_lo_in_ao, mf.get_ovlp(),
                                        self.imp_lo_idx, self.env_lo_idx,
                                        method=embed_method)

        return self.coeff_eo_in_ao, self.dm_eo_in_ao


    def get_eomf(self, mf):
        """
        get the embedding mean-field object with reduced mo space
        ready to be used for excited-state calculations
        -----------------------------------------------------------------------
        returns:
        eomf: embedding mf object
        """
        # effective number of electrons in embedding space
        nelec = int(round(np.trace(self.dm_eo_in_ao)))
        print('nelec: %d in neo: %d' % (nelec, self.coeff_eo_in_ao.shape[1]))
        # assume restricted case
        neleca, nelecb = nelec//2, nelec//2 # restricted
        #TODO: unrestricted case


        ovlp_ao = mf.get_ovlp()
        fock_ao = mf.get_fock()

        coeff_eo_in_ao = self.coeff_eo_in_ao
        proj = np.einsum('mi,ni,nl->ml', coeff_eo_in_ao, coeff_eo_in_ao, ovlp_ao)
        fock_ao = np.einsum('nm,nl,ls->ms', proj, fock_ao, proj)

        from scipy.linalg import eigh
        mo_energy, mo_coeff = eigh(fock_ao, ovlp_ao)
        zero_list = np.where(abs(mo_energy) < 10 ** (-7))[0]
        mo_energy = np.delete(mo_energy, zero_list, axis=0)
        mo_coeff = np.delete(mo_coeff, zero_list, axis=1)
        mo_occ = np.zeros_like(mo_energy)
        for i in range(neleca):
            mo_occ[i] = 2
        print_matrix('mo_energy in embedding:', mo_energy)

        mol = mf.mol.copy()
        mol.nelectron = nelec # change effective electrons

        eomf = mf.copy()
        eomf.mo_coeff0 = mf.mo_coeff # full system orbitals for dft
        eomf.mo_occ0 = mf.mo_occ # full system orbitals for dft
        # use effective embedding orbitals
        eomf.mo_coeff = mo_coeff
        eomf.mo_occ = mo_occ
        eomf.mo_energy = mo_energy

        return eomf



if __name__ == '__main__':
    from wavefunction_analysis.utils.pyscf_parser import *
    from pyscf import scf

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
        basis = '6-311++g**',
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

    for i in range(-mol.nelectron//2+1, mf.mo_coeff.shape[0]-mol.nelectron//2):
        print('i:', i, mol.nelectron//2+i)
        get_embedding_system(mf, frgm_idx, extra_orb=i)
