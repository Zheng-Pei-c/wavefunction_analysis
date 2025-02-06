import sys

from wavefunction_analysis import np
from wavefunction_analysis.utils.wf_overlap import assemble_amplitudes, change_phase, cal_wf_overlap, sign_fixing
from wavefunction_analysis.utils import print_matrix, swap_largest_to_diagonal

from pyscf import scf, tdscf, gto

if __name__ == '__main__':
    atom0 = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.12
    """
    atom1 = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.13
    """
    atom2 = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.14
    """
    if 'f' in sys.argv[1]:
        atom0 = """
        N 0.0 0.0 0.0
        N 0.0 0.0 1.336
        """
        atom1 = """
        N 0.0 0.0 0.0
        N 0.0 0.0 1.346
        """
        atom2 = """
        N 0.0 0.0 0.0
        N 0.0 0.0 1.356
        """

    basis = 'sto-3g'
    functional = 'hf'
    nroots = 10
    rpa = 0

    mols, mo, xs, ys = [], [], [], []
    for atom in [atom0, atom1, atom2]:
        mol = gto.M(
                atom = atom,
                basis = basis,
                )

        mf = scf.RKS(mol)
        mf.xc = functional

        e = mf.kernel()
        nocc = int((mf.mo_occ>0).sum())
        nvir = mf.mo_coeff.shape[0] - nocc
        #if nroots > nocc*nvir: nroots = nocc*nvir
        nroots = nocc*nvir

        td_model = tdscf.TDDFT if rpa else tdscf.TDA
        td = td_model(mf)
        td.nroots = nroots
        td.kernel()

        _mo = mf.mo_coeff
        _xs, _ys = assemble_amplitudes(td.xy, nocc, nroots, rpa)

        mols.append(mol)
        mo.append(_mo)
        xs.append(_xs)
        ys.append(_ys)


    state_ovlp = []
    u, s, vt = [], [], []
    for i, j in [(0,1), (1,2), (0,2)]:
        ovlp = gto.intor_cross('int1e_ovlp', mols[i], mols[j])
        _state_ovlp = cal_wf_overlap(xs[i], ys[i], xs[j], ys[j], mo[i], mo[j], ovlp)
        _state_ovlp = swap_largest_to_diagonal(_state_ovlp)

        _u, _s, _vt = np.linalg.svd(_state_ovlp)
        state_ovlp.append(_state_ovlp)
        u.append(_u)
        s.append(_s)
        vt.append(_vt)

    state_ovlp = np.array(state_ovlp)
    s = np.array(s)
    print_matrix('state_ovlp:', state_ovlp)
    print_matrix('singular values:', s)

    Uij = np.zeros((nroots+1, nroots+1))
    for i in range(nroots+1):
        tmp = state_ovlp[2,i,i]
        Uij[i,i] = np.cos(np.arccos(np.abs(tmp))/2.)
        for j in range(i):
            tmp = state_ovlp[2,i,j]
            Uij[i,j] = Uij[j,i] = np.sin(np.arcsin(np.abs(tmp))/2.)

    print_matrix('Uij:', Uij)
