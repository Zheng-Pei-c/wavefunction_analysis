from wavefunction_analysis import sys, np, itertools
from wavefunction_analysis.utils import print_matrix

def mps_canonical(ndims, nsite, rand_seed=True, normalize='both'):
    """
    random bra MPS wavefunction matries at top
    order of legs: left-bottom-right
    ket MPS at bottom is the transpose with legs in right-top-left order
    """
    ntot = np.prod(nsite)

    rng = np.random.default_rng(rand_seed)

    mps = []
    mps.append(rng.random([1]+[n for n in ndims[1:]]))
    for l in range(1, ntot-1):
        mps.append(rng.random(ndims))
    mps.append(rng.random([n for n in ndims[:-1]]+[1]))

    # normalize would reduce the matrix dimensions
    if normalize in {1, 'left', 'both'}:
        mps = mps_canonical_left(mps)
    if normalize in {2, 'right', 'both'}:
        mps = mps_canonical_right(mps)

    print('mps shape:\n', [m.shape for m in mps])
    return mps


def mps_canonical_left(mps, debug=0):
    # normalize MPS matrices from the left
    ntot = len(mps)

    for l in range(ntot):
        ndims = mps[l].shape
        u, s, vt = np.linalg.svd(mps[l].reshape(-1, ndims[-1]), full_matrices=False)
        mps[l] = np.reshape(u, [n for n in ndims[:-1]]+[u.shape[-1]])
        if l < ntot-1:
            mps[l+1] = np.einsum('i,ij,j...->i...', s, vt, mps[l+1])

    if debug: # check normalization
        pass

    return mps


def mps_canonical_right(mps):
    # normalize MPS matrices from the right
    ntot = len(mps)

    for l in range(ntot-1, -1, -1):
        ndims = mps[l].shape
        u, s, vt = np.linalg.svd(mps[l].reshape(ndims[0], -1), full_matrices=False)
        mps[l] = np.reshape(vt, [vt.shape[0]]+[n for n in ndims[1:]])
        if l > 0:
            mps[l-1] = np.einsum('...k,kl,l->...l', mps[l-1], u, s)

    return mps


"""
spin models from
https://github.com/GCatarina/DMRG_MPS_didactic/blob/main/DMRG-MPS_implementation.ipynb
"""

def _mpo_XY(sigma=None):
    r"""
    matrix product operator MPO hamiltonian of XY spin-1/2 model
    H_{XY} = -1/2 \sum_{l} (\sigma_{l}^{+} \sigma_{l+1}^{-} + \sigma_{l}^{-} \sigma_{l+1}^{+})
    """
    if sigma is None:
        sigma = get_spins('0pm')
    I2, sp, sm = sigma

    """
           |
           3
           |
    ---1---O---2---
           |
           4
           |
    """
    # in the order of [left, right, top, bottom]
    # the first two indices (left and right) are used to contract
    # the last two dimensions are the Pauli matrix indices
    Hl = np.zeros((4,4,2,2))
    Hl[0,0] = I2
    Hl[1,0] = sm
    Hl[2,0] = sp
    Hl[3,1] = -.5*sp
    Hl[3,2] = -.5*sm
    Hl[3,3] = I2

    return Hl


def _mpo_heisenberg(sigma=None):
    r"""
    matrix product operator MPO hamiltonian of spin-1 Heisenberg model
    H_{Heis} = J \sum_{l} (\sigma_{l}^{z} \sigma_{l+1}^{z} + 1/2 \sigma_{l}^{+} \sigma_{l+1}^{-} + 1/2 \sigma_{l}^{-} \sigma_{l+1}^{+}) - h \sum_{l} \sigma_{l}^{z}
    """
    if sigma is None:
        sigma = get_spins('0zpm')
    I3, sz, sp, sm = sigma

    # in the order of [left, right, top, bottom]
    # the first two indices are used to contract
    # the last two dimensions are the Pauli matrix indices
    Hl = np.zeros((5,5,3,3))
    Hl[0,0] = I3
    Hl[1,0] = sz
    Hl[2,0] = sm
    Hl[3,0] = sp
    Hl[4,0] = -h*sz
    Hl[4,1] = J*sz
    Hl[4,2] = .5*J*sp
    Hl[4,3] = .5*J*sm
    Hl[4,4] = I3
    return Hl


def mpo_hamil(nsite, Hl=None, model=None, sigma=None):
    # hamiltonian in MPO representation
    if not isinstance(Hl, np.ndarray):
        if model == 'XY':
            Hl = _mpo_XY(sigma)
        elif model == 'heisenberg':
            Hl = _mpo_heisenberg(sigma)
        else:
            raise NotImplementedError('local Hamiltonian is not defined')

    ntot = np.prod(nsite)
    H = [Hl for l in range(ntot)]
    H[0] = H[0][-1:] # keep same n-dimension
    H[ntot-1] = H[ntot-1][:,0:1] # keep same n-dimension

    print('MPO H shape:\n', [m.shape for m in H])
    return H


def get_hamil_from_mpo(h_mpo):
    # check the MPO matrices from the mpo spin function
    # by comparing to the brute-force hamiltonian matrix
    ntot = len(h_mpo)

    """
            |         |
            j         m
            |         |
    ---1---[T]---i---[T]---l---
            |         |
            k         n
            |         |
    """
    Taux = h_mpo[0][0] # the first matrix left leg has dimension of 1
    for l in range(1, ntot):
        Taux = np.einsum('ijk,ilmn->ljmkn', Taux, h_mpo[l])
        ndims = Taux.shape
        Taux = Taux.reshape(ndims[0], ndims[1]*ndims[2], ndims[3]*ndims[4])

    H = Taux[0] # the last matrix right leg has dimension of 1
    return H


def zipper_from_left(Mt, O, Mb, Tl=np.ones((1,1,1))):
    r"""
    Tl legs order from bottom to top (default as np.ones((1,1,1)))
    Contract Tl from the left with 1). bra Mt at top,
                                   2). operator O at middle,
                                   3). ket Mt at bottom.
    /---3---*q*---1---Mt---3-k--
    |                 |
    |                 2
    |                 |
    |                 *l
    |                 |
    |                 3              /---3-k--
    |                 |              |
    Tl--2---*p*---1---O----2-j-- =   Tf--2-j--
    |                 |              |
    |                 4              \---1-i--
    |                 |
    |                 *m
    |                 |
    |                 2
    |                 |
    \---1---*n*---3---Mb---1-i---
    """
    # contract from bottom to top
    # np.einsum('imn,npq,pjlm,qlk->ijk', Mb, Tl, O, Mt)
    Taux = np.einsum('imn,npq->impq', Mb, Tl)
    Taux = np.einsum('impq,pjlm->ijql', Taux, O)
    Taux = np.einsum('ijql,qlk->ijk', Taux, Mt)

    return Taux


def zipper_from_right(Mt, O, Mb, Tr=np.ones((1,1,1))):
    r"""
    Tr legs order from top to bottom (default as np.ones((1,1,1)))
    Contract Tl from the right with 1). bra Mt at top,
                                    2). operator O at middle,
                                    3). ket Mt at bottom.
    --i-1---Mt---3---*q*---1---/
            |                  |
            2                  |
            |                  |
            *l                 |
            |                  |
            3                  |      --1-i--/
            |                  |             |
    --j-1---O----2---*p*---2---Tr =   --2-j--Tf
            |                  |             |
            4                  |      --3-k--\
            |                  |
            *m                 |
            |                  |
            2                  |
            |                  |
    --k-3---Mb---1---*n*---3---\
    """
    # contract from top to bottom
    # np.einsum('ilq,qpn,jplm,nmk->ijk', Mt, Tr, O, Mb)
    Taux = np.einsum('ilq,qpn->ilpn', Mt, Tr)
    Taux = np.einsum('ilpn,jplm->ijnm', Taux, O)
    Taux = np.einsum('ijnm,nmk->ijk', Taux, Mb)

    return Taux


def dmrg_sweep(h_mpo, mps, Tzip, pick_eig='SA'):
    """
    Tzip is given from the right with indecies ordering from up to bottom
    """
    ntot = len(h_mpo)
    e_list = []

    # for Lanczos diagonalize
    from scipy.sparse.linalg import eigsh

    # right sweep from left
    for l in range(ntot):
        r"""
        |             |             |
        3-l           3-m           1-n
        |             |             |
        T---2-*p*-1---O---2-*q*-2---T
        |             |             |
        1-i           4-j           3-k
        |             |             |
        """
        Taux = np.einsum('ipl,pqmj->ijlmq', Tzip[l], h_mpo[l])
        Taux = np.einsum('ijlmq,nqk->ijklmn', Taux, Tzip[l+2])
        ni, nj, nk = Taux.shape[:3]
        H = Taux.reshape(ni*nj*nk, -1)

        # Lanczos diagonalize H to get the lowest energy state
        # v0 is the initial guess
        # which can be SA (smallest algebraic), SM (smallest magnitude)
        e, vec = eigsh(H, k=1, which=pick_eig, v0=mps[l])
        e_list.append(e[0])

        # update MPS wavefunction at site l
        # in the right sweep, the MPS has to be left-normalized
        u, s, vt = np.linalg.svd(vec.reshape(ni*nj, nk), full_matrices=False)
        mps[l] = u.reshape(ni, nj, -1)
        if l < ntot-1:
            mps[l+1] = np.einsum('i,ij,jkl->ikl', s, vt, mps[l+1])

        # update T zipper
        Tzip[l+1] = zipper_from_left(mps[l], h_mpo[l], mps[l].conj().T, Tzip[l])

    # left sweep from right
    for l in range(ntot-1, -1, -1):
        Taux = np.einsum('ipl,pqmj->ijlmq', Tzip[l], h_mpo[l])
        Taux = np.einsum('ijlmq,nqk->ijklmn', Taux, Tzip[l+2])
        ni, nj, nk = Taux.shape[:3]
        H = Taux.reshape(ni*nj*nk, -1)

        # Lanczos diagonalize H to get the lowest energy state
        # v0 is the initial guess
        # which can be SA (smallest algebraic), SM (smallest magnitude)
        e, vec = eigsh(H, k=1, which=pick_eig, v0=mps[l])
        e_list.append(e[0])

        # update MPS wavefunction at site l
        # in the left sweep, the MPS has to be right-normalized
        u, s, vt = np.linalg.svd(vec.reshape(ni, nj*nk), full_matrices=False)
        mps[l] = vt.reshape(-1, nj, nk)
        if l > 0:
            mps[l-1] = np.einsum('ijk,kl,l->ijl', mps[l-1], u, s)

        # update T zipper
        Tzip[l+1] = zipper_from_right(mps[l], h_mpo[l], mps[l].conj().T, Tzip[l+2])

    return np.ravel(e_list), mps, Tzip


def dmrg_opt_gs(h_mpo, mps=None, Tzip=None, nbond=None, pick_eig='SA', nmax=10):
    ntot = len(h_mpo)

    if mps is None:
        ndims = [nbond, h_mpo[0].shape[2], nbond]
        mps = mps_canonical(ndims, ntot, normalize='both')

    if Tzip is None:
        # initialize T zipper from right, indexing from up to bottom
        Tzip = [None]*(ntot+2)
        Tzip[ntot+1] = np.ones((1,1,1))
        Tzip[0] = np.ones((1,1,1))
        for l in range(ntot-1, -1, -1):
            Tzip[l+1] = zipper_from_right(mps[l], h_mpo[l], mps[l].conj().T, Tzip[l+2])

    e_list = []
    for it in range(nmax):
        es, mps, Tzip = dmrg_sweep(h_mpo, mps, Tzip, pick_eig)
        e_list.append(es)

    return e_list, mps


def mpo_spin_correlation(mps, sigma=None, ns=None):
    r"""
    \expval{\sigma_{l}^{+} \sigma_{l+1}^{-}}
    """
    if sigma is None:
        sigma = get_spins('0pm')
    mpo_I2, mpo_sp, mpo_sm = sigma
    mpo_I2 = mpo_I2.reshape((1,1,2,2))
    mpo_sp = mpo_sp.reshape((1,1,2,2))
    mpo_sm = mpo_sm.reshape((1,1,2,2))

    if ns is None:
        ntot = len(mps)
        ns = range(ntot-1)
    elif isinstance(ns, int): # require single site
        ns = [ns]

    correlation = []
    for n in ns: # loop over the sites
        Taux = np.ones((1,1,1))
        for l in range(ntot): # get correlation of n-th site
            if l == n:
                Taux = zipper_from_left(mps[l], mpo_sp, mps[l].conj().T, Taux)
            elif l == n+1:
                Taux = zipper_from_left(mps[l], mpo_sm, mps[l].conj().T, Taux)
            else:
                Taux = zipper_from_left(mps[l], mpo_I2, mps[l].conj().T, Taux)
        correlation.append(Taux[0,0,0])

    return np.array(correlation)


if __name__ == '__main__':
    ndims = [20, 3, 20]
    nsite = 10
    M = mps_canonical(ndims, nsite, normalize='both')

    I2 = np.eye(2)
    sp = np.zeros((2,2))
    sp[0,1] = 1
    sm = np.zeros((2,2))
    sm[1,0] = 1
    sigma = [I2, sp, sm]

    N = 20
    D = 10
    nmax = 10
    h_mpo = mpo_hamil(N, model='XY', sigma=sigma)
    e_list, mps = dmrg_opt_gs(h_mpo, nbond=D, nmax=nmax)
    print_matrix('e_list:', np.array(e_list))
    correlation = mpo_spin_correlation(mps, sigma)
    print_matrix('correlation:', correlation)
