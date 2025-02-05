from wavefunction_analysis import np
from wavefunction_analysis.utils.wf_overlap import assemble_amplitudes, change_phase, cal_wf_overlap
from wavefunction_analysis.utils import print_matrix

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

    basis = 'sto-3g'
    functional = 'hf'
    nroots = 50
    rpa = 0

    u, s, vt = [], [], []
    for atoms in [(atom0, atom1), (atom1, atom2), (atom0, atom2)]:
        mols = []
        mo, xs, ys = [], [], []
        for atom in atoms:
            mol = gto.M(
                    atom = atom,
                    basis = basis,
                    )
            mols.append(mol)

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

            mo.append(mo)
            xs.append(_xs)
            ys.append(_ys)

        ovlp = gto.intor_cross('int1e_ovlp', mols[0], mols[1])
#        mo[1], xs[1], ys[1] = change_phase(xs[0], ys[0], xs[1], ys[1], mo[0], mo[1], ovlp)
        _state_ovlp = cal_wf_overlap(xs[0], ys[0], xs[1], ys[1], mo[0], mo[1], ovlp)
        print_matrix('state_ovlp:', _state_ovlp)

        _u, _s, _vt = np.linalg.svd(_state_ovlp)
        u.append(_u)
        s.append(_s)
        vt.append(_vt)

    print_matrix('singular values:', np.array(s))
