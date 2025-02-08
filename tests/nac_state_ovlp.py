import sys
import scipy

from wavefunction_analysis import np, itertools
from wavefunction_analysis.utils.wf_overlap import assemble_amplitudes, change_phase, cal_wf_overlap, sign_fixing
from wavefunction_analysis.utils import print_matrix, swap_largest_to_diagonal

from pyscf import scf, tdscf, gto

def get_excited_state_mos(C, X, Y, has_y):
    # hold zero matrices
    no, nv = X.shape[-2:]
    zeros_o, zeros_v = np.zeros((no,no)), np.zeros((nv,nv))

    def kernel(i):
        x = X[i]
        if has_y: x += Y[i].conj()

        exp = np.block([[zeros_o, -x], [x.conj().T, zeros_v]])
        exp = scipy.linalg.expm(exp)
        return np.einsum('mp,pq->mq', C, exp)

    return kernel



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
    nroots = 5
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
        _mo = mf.mo_coeff

        nbas = _mo.shape[0]
        nocc = int((mf.mo_occ>0).sum())
        nvir = nbas - nocc
        #if nroots > nocc*nvir: nroots = nocc*nvir
#        nroots = nocc*nvir

        td_model = tdscf.TDDFT if rpa else tdscf.TDA
        td = td_model(mf)
        td.nroots = nroots
        td.kernel()

        _xs, _ys = assemble_amplitudes(td.xy, nocc, nroots, rpa)

        mols.append(mol)
        mo.append(_mo)
        xs.append(_xs)
        ys.append(_ys)

    mo = np.array(mo)
    xs = np.array(xs)
    ys = np.array(ys)

    state_ovlp = []
    u, s, vt = [], [], []
    for i, j in [(0,1), (1,2), (0,2)]:
        ovlp = gto.intor_cross('int1e_ovlp', mols[i], mols[j])
        _state_ovlp = cal_wf_overlap(xs[i], ys[i], xs[j], ys[j], mo[i], mo[j], ovlp)
        #_state_ovlp = swap_largest_to_diagonal(_state_ovlp)

        _u, _s, _vt = np.linalg.svd(_state_ovlp)
        state_ovlp.append(_state_ovlp)
        u.append(_u)
        s.append(_s)
        vt.append(_vt)

        s0 = mf.get_ovlp(mols[i])
        print_matrix('s0:', s0)
        print_matrix('s0 cross:', ovlp)
        print_matrix('s0 cross - s0:', ovlp-s0)
        s0_mo = np.einsum('pi,pq,qj->ij', mo[i][:,:nocc], s0, mo[j][:,:nocc])
        print_matrix('s0_mo:', s0_mo)
        s0_mo_c = np.einsum('pi,pq,qj->ij', mo[i][:,:nocc], ovlp, mo[j][:,:nocc])
        print_matrix('s0_mo cross:', s0_mo_c)
        print_matrix('s0_mo cross - s0_mo:', s0_mo_c - s0_mo)

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



    print('start grassmann')
    #from pymanopt.manifolds import Grassmann
    from wavefunction_analysis.manifold.riemannian import Grassmann
    from wavefunction_analysis.utils import get_ortho_basis

    # orthonormalized occupied orbitals
    coeff = []
    for i in range(3):
        _p = mo[i]
        _L, _Z, _Binv = get_ortho_basis(mf.get_ovlp(mols[i]), method='lowdin')
        _p = np.einsum('pq,qi->pi', _L, _p)
        #print_matrix('s:', np.einsum('pi,pj->ij', _p, _p))
        coeff.append(_p)

    coeff = np.array(coeff)
    ps = coeff[:,:,:nocc] # ground-state
    # excited-state
    ps_2 = np.empty((3,nroots,nbas,nbas))
    for i, k in itertools.product(range(3), range(nroots)):
        x, c = xs[i,k], coeff[i]
        exp = np.block([[np.zeros((nocc,nocc)), -x], [x.T, np.zeros((nvir,nvir))]])
        exp = scipy.linalg.expm(exp)
        ps_2[i,k] = np.einsum('pi,ij->pj', c, exp)
    print_matrix('ps_2:', ps_2)
    #sss = np.einsum('xkpi,xkpj->xkij', ps_2, ps_2)
    #print_matrix('sss0:', sss[0])


    gr = Grassmann(x0=ps[0])
    v2 = gr.log(ps[0], ps[2])
    print_matrix('tangent:', v2)
    dot1 = np.einsum('ji,jk->ik', v2, ps[0])
    dot2 = np.einsum('ji,jk->ik', ps[0], v2)
    print('check tangent:', np.allclose(dot1, dot2))

    p2 = gr.exp(ps[0], v2)
    print_matrix('ps[2]:', ps[2])
    print_matrix('p2:', p2)
    print('are points 2 same?', np.allclose(np.abs(ps[2]), np.abs(p2)), end=' ')
    print(np.linalg.norm(np.abs(ps[2])-np.abs(p2)))

    p1 = gr.exp(ps[0], v2, dt=.5)
    print_matrix('ps[1]:', ps[1])
    print_matrix('p1:', p1)
    print('are points 1 same?', np.allclose(np.abs(ps[1]), np.abs(p1)), end=' ')
    print(np.linalg.norm(np.abs(ps[1])-np.abs(p1)))
