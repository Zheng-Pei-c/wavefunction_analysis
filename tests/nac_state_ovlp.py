import sys
import scipy

from wavefunction_analysis import np, itertools
from wavefunction_analysis.utils.wf_overlap import assemble_amplitudes, change_phase, cal_wf_overlap, sign_fixing
from wavefunction_analysis.utils import print_matrix, swap_largest_to_diagonal, get_ortho_basis

#from pymanopt.manifolds import Grassmann
from wavefunction_analysis.manifold.riemannian import Grassmann

from wavefunction_analysis.plot import plt

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

    atom0 = """
  C   -0.0000000    0.0000000    0.6133835
  O    0.0000000   -0.0000000   -0.6060812
  H   -0.0000000   -0.9391099    1.1556173
  H   -0.0000000    0.9391099    1.1556173
    """
    atom1 = """
  C   -0.0000000    0.0000000    0.6133835
  O    0.0000000   -0.0000000   -0.6160812
  H   -0.0000000   -0.9391099    1.1556173
  H   -0.0000000    0.9391099    1.1556173
    """
    atom2 = """
  C   -0.0000000    0.0000000    0.6133835
  O    0.0000000   -0.0000000   -0.6260812
  H   -0.0000000   -0.9391099    1.1556173
  H   -0.0000000    0.9391099    1.1556173
    """
    basis = 'sto-3g'
    functional = 'hf'
    nroots = 5
    rpa = 0

    aos, mols, mo, xs, ys = [], [], [], [], []
    for i, atom in enumerate([atom0, atom1, atom2]):
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

        aos.append(mf.get_ovlp())
        mols.append(mol)
        mo.append(_mo)
        xs.append(_xs)
        ys.append(_ys)

    aos = np.array(aos)
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


    # orthonormalized occupied orbitals
    coeff, amps = [], []
    for i in range(3):
        _p, _x = mo[i], xs[i]*np.sqrt(2.)
        _L, _Z, _Binv = get_ortho_basis(aos[i], method='lowdin')
        _p = np.einsum('pq,qi->pi', _L, _p)
        #print_matrix('s:', np.einsum('pi,pj->ij', _p, _p))
        coeff.append(_p)
        _x = np.einsum('pi,kia,qa->pqk', _p[:,:nocc], _x, _p[:,nocc:]).reshape((-1, nroots))
        amps.append(_x)

    coeff = np.array(coeff)
    amps = np.array(amps)


    z_oo, z_vv = np.zeros((nocc,nocc)), np.zeros((nvir,nvir))
    ps_2 = np.empty((3,nroots,nbas,nbas)) # excited-state orbitals
    uuu = np.empty((3,nroots,nbas,nocc))
    for i, k in itertools.product(range(3), range(nroots)):
        x, c = xs[i,k], mo[i]
        exp = np.block([[z_oo, -x], [x.T, z_vv]])
        exp = scipy.linalg.expm(exp)
        uuu[i,k] = np.copy(exp[:,:nocc])
        uut = np.einsum('pi,qi->pq', uuu[i,k], uuu[i,k].conj())
        ps_2[i,k] = np.einsum('pi,ij->pj', c, exp)
        print('det0:', np.linalg.det(np.einsum('pi,pq,qj->ij', c[:,:nocc].conj(), aos[i], c[:,:nocc])))
        print('det1:', np.linalg.det(np.einsum('pi,pq,qj->ij', ps_2[i,k,:,:nocc].conj(), aos[i], ps_2[i,k,:,:nocc])))
        print('det2:', np.linalg.det(np.einsum('pi,pq,qj->ij', c[:,:nocc].conj(), aos[i], ps_2[i,k,:,:nocc])))
    ps_2 = ps_2[:,:,:,:nocc]
    #print_matrix('ps_2:', ps_2)
    sss = np.einsum('skpi,spq,slqj->sklij', ps_2.conj(), aos, ps_2)
    sss_det = []
    for s in range(3):
        for k in range(nroots):
            for l in range(nroots):
                _sss = sss[s,k,l]
                sss_det.append(np.linalg.det(_sss))

    print_matrix('identity:', np.array(sss_det).reshape((3, nroots, nroots)))

    ps = mo[:,:,:nocc]
    k = 0
    for k, (i, j) in enumerate([[0,1,], [1,2], [0,2]]):
        o = np.einsum('pm,pn->mn', amps[i].conj(), amps[j])
        print_matrix('state ovlp one-electron approximation', o)

        ovlp = gto.intor_cross('int1e_ovlp', mols[i], mols[j])
        ovlp_00 = np.einsum('pi,pq,qj->ij', ps[i].conj(), ovlp, ps[j])
        ovlp_e0 = np.einsum('kpi,pq,qj->kij', ps_2[i].conj(), ovlp, ps[j])
        ovlp_0e = np.einsum('pi,pq,kqj->kij', ps[i].conj(), ovlp, ps_2[j])
        ovlp_ee = np.einsum('kpi,pq,lqj->klij', ps_2[i].conj(), ovlp, ps_2[j])

        s00 = np.linalg.det(ovlp_00)
        state_ovlp[k,0,0] = s00**2
        se0, s0e, smn = [], [], []
        for m in range(nroots):
            se0.append(np.linalg.det(ovlp_e0[m]))
            s0e.append(np.linalg.det(ovlp_0e[m]))
            for n in range(nroots):
                smn.append(np.linalg.det(ovlp_ee[m,n]))

        state_ovlp[k,1:,0] = np.array(se0) * s00
        state_ovlp[k,0,1:] = np.array(s0e) * s00
        state_ovlp[k,1:,1:] = np.array(smn).reshape((nroots, nroots)) * s00 + np.einsum('k,l->kl', se0, s0e)

    print_matrix('state ovlp orbitals', state_ovlp)


    ps = coeff[:,:,:nocc]
    gr = Grassmann(x0=ps[0], retraction='polar')
    tangent = gr.log(ps[0], ps[2])
    print_matrix('tangent:', tangent, 10)
    gr.check_tangent(ps[0], tangent)

    xs = xs.transpose(0,1,3,2)
    amp_tang = gr.horizontal_lift(coeff[0,:,nocc:], xs[0])


    for i, dt in enumerate([.5, 1.]):
        i = i+1

        p2 = gr.exp(ps[0], tangent)
        if i==2:
            p2[:,[4,5]] = p2[:,[5,4]] # orbital swap
        print_matrix('ps[0]:', ps[0], 10)
        print_matrix('p'+str(i)+':', p2, 10)
        print_matrix('ps['+str(i)+']:', ps[i], 10)
        print('are points 2 same?', np.allclose(np.abs(ps[2]), np.abs(p2)), end=' ')
        print(np.linalg.norm(np.abs(ps[2])-np.abs(p2)))


    for i, dt in enumerate([.5, 1.]):
        i = i+1

        #x2 = gr.projection(ps[i], amp_tang)
        x2 = gr.transport(ps[0], tangent, amp_tang, dt)
        x2 = gr.get_tangent(coeff[i,:,nocc:], x2)

        for k in range(1):
            print('k:', k)
            print_matrix('x[0]:', xs[0,k], 10)
            print_matrix('x'+str(i)+':', x2[k], 10)
            print_matrix('x['+str(i)+']:', xs[i,k], 10)
        print('are tangents at points '+str(i)+' same?', np.allclose(np.abs(xs[2]), np.abs(x2)), end=' ')
        print(np.linalg.norm(np.abs(xs[i])-np.abs(x2)))


    #ps = xs.reshape((3, nroots, -1)).transpose(0,2,1)*np.sqrt(2.)
    ps = np.array(amps)
    gr = Grassmann(x0=ps[0], retraction='polar')
    tangent = gr.log(ps[0], ps[2])

    for i, dt in enumerate([.5, 1.]):
        i = i+1

        p2 = gr.exp(ps[0], tangent)
        print_matrix('ovlp:', np.einsum('pi,pj->ij', ps[i], p2))

    print(ps.shape)
    #ps = xs.reshape((3, nroots, -1)).transpose(0,2,1)*np.sqrt(2.)
    for i in range(nroots):
        print_matrix('ps:', ps[:,:,i].T)
    fig = plt.figure(figsize=[6,3], dpi=300)
    for j in range(ps.shape[2]):
        for i in range(ps.shape[1]):
            plt.plot(range(j*3, j*3+3), np.abs(ps[:,i,j]))
    plt.savefig('amp.png')
