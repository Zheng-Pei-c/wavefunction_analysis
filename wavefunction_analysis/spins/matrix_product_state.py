from wavefunction_analysis import sys, np, itertools
from wavefunction_analysis.utils import print_matrix
from wavefunction_analysis.spins import get_spins

from wavefunction_analysis.utils import monitor_performance

def mps_canonical(ndims, nsite, rand_seed=True, normalize='both'):
    r"""
    random bra MPS wavefunction matries at top
    order of legs: left-bottom-right
    ket MPS at bottom is the transpose with legs in right-top-left order
    ---1---Mt---3---                  |
           |                          2
           2                          |
           |                   ---3---Mb---1---
    the left and right legs are virtual bounds to be contracted to adjacent MPS
    the bottom or top legs are physical bounds (spins) to be contracted to MPO
    ndims: an array of leg dimensions [nd,ns,nd] in the order [left, bottom, right]
           such that the first and third numbers should be same
    """
    ntot = np.prod(nsite)

    rng = np.random.default_rng(rand_seed)

    mps = []
    # the legs of MPS at end sites have virtual bond dimension of 1
    # to be contracted with left/right zippers
    mps.append(rng.random([1]+[n for n in ndims[1:]])) # [1,ns,nd]
    for l in range(1, ntot-1):
        mps.append(rng.random(ndims)) # [nd,ns,nd]
    mps.append(rng.random([n for n in ndims[:-1]]+[1])) # [nd,ns,1]

    # normalization would reduce the matrix dimensions (effective virtual bonds)
    if normalize in {1, 'left', 'both'}:
        mps = mps_canonical_left(mps)
    if normalize in {2, 'right', 'both'}:
        mps = mps_canonical_right(mps)

    print('mps shape:\n', [m.shape for m in mps])
    return mps


def mps_canonical_left(mps, debug=0):
    r"""
    normalize MPS matrices from the left
    treat the first two dimensions as a vector to reduce the right leg numbers
    M1 * M2 = U*s*V^T * M2 = U * (s*V^T*M2)
    such that M1 is replaced by a smaller U
    """
    ntot = len(mps)

    # scan from the left
    for l in range(ntot):
        ndims = mps[l].shape
        u, s, vt = np.linalg.svd(mps[l].reshape(-1, ndims[-1]), full_matrices=False)
        mps[l] = np.reshape(u, [n for n in ndims[:-1]]+[u.shape[-1]])
        if l < ntot-1: # update the next at right
            mps[l+1] = np.einsum('i,ij,j...->i...', s, vt, mps[l+1])

    if debug: # check normalization
        pass

    return mps


def mps_canonical_right(mps):
    r"""
    normalize MPS matrices from the right
    treat the last two dimensions as a vector to reduce the left leg numbers
    M1 * M2 = M1 * U*s*V^T = (M1*U*s) * V^T
    such that M2 is replaced by a smaller V^T
    """
    ntot = len(mps)

    # scan from right
    for l in range(ntot-1, -1, -1):
        ndims = mps[l].shape
        u, s, vt = np.linalg.svd(mps[l].reshape(ndims[0], -1), full_matrices=False)
        mps[l] = np.reshape(vt, [vt.shape[0]]+[n for n in ndims[1:]])
        if l > 0: # update the next at left
            mps[l-1] = np.einsum('...k,kl,l->...l', mps[l-1], u, s)

    return mps


"""
spin models from
https://doi.org/10.1140/epjb/s10051-023-00575-2
https://github.com/GCatarina/DMRG_MPS_didactic/blob/main/DMRG-MPS_implementation.ipynb
"""

"""
the last two indices (top and bottom) of MPOs are physical bounds
in the implementation, whose dimensions depend on spin multiplicity
"""

def _mpo_xx_1d(j, np_matrix=True, spin_j=.5, sigma=None):
    r"""
    one-site matrix product operator MPO hamiltonian of XY spin model
    coefficients are put at the last row
    H_{xy} = 1/2 j sum_{i} (sigma_{i,+} sigma_{i+1,-} + sigma_{i,-} sigma_{i+1,+})
    """
    if sigma is None:
        sigma = get_spins('0+-', j=spin_j, np_matrix=True)
    si, sp, sm = sigma

    # in the order of [left, right, top, bottom]
    # the first two indices are used to contract
    # the last two dimensions are the Pauli matrix indices
    Hl = np.zeros((4,4,int(2*spin_j+1),int(2*spin_j+1)))
    Hl[0,0] = si
    Hl[1,0] = sm
    Hl[2,0] = sp
    Hl[3,1] = (.5*j)*sp
    Hl[3,2] = (.5*j)*sm
    Hl[3,3] = si

    return Hl


def _mpo_heisenberg_1d(j, hz, np_matrix=True, spin_j=.5, sigma=None):
    r"""
    one-site matrix product operator MPO hamiltonian of Heisenberg model
    H_{Heis} = sum_{i} ( j_{x} * sigma_{i,x} sigma_{l+1,x}
                       + j_{y} * sigma_{i,y} sigma_{i+1,y}
                       + j_{z} * sigma_{i,z} sigma_{i+1,z})
                  + sum_{i} hz * sigma_{i,z}
    the indices are in the order of [left, right, top, bottom]
    the first two indices (left and right) are used to contract neighboring MPO
    the last two dimensions are the spin matrix indices
           |
           3
           |
    ---1---O---2---
           |
           4
           |
    """
    if sigma is None:
        sigma = get_spins('0xyz', j=spin_j, np_matrix=True)
    si, sx, sy, sz = sigma

    Hl = np.zeros((5,5,int(2*spin_j+1),int(2*spin_j+1)), dtype=sy.dtype)
    Hl[0,0] = si
    Hl[1,0] = sx
    Hl[2,0] = sy
    Hl[3,0] = sz
    Hl[4,0] = hz*sz
    Hl[4,1] = j[0]*sx
    Hl[4,2] = j[1]*sy
    Hl[4,3] = j[2]*sz
    Hl[4,4] = si
    return Hl

_mpo_xyz_1d = _mpo_heisenberg_1d


def _mpo_xxz_1d(j, hz, np_matrix=True, spin_j=.5, sigma=None):
    r"""
    one-site matrix product operator MPO hamiltonian of XXZ model
    similar to XYZ Heisenberg model
    but use ladder operator to have real matrix
    """
    if sigma is None:
        sigma = get_spins('0+-z', j=spin_j, np_matrix=True)
    si, sp, sm, sz = sigma

    # in the order of [left, right, top, bottom]
    # the first two indices are used to contract
    # the last two dimensions are the Pauli matrix indices
    Hl = np.zeros((5,5,int(2*spin_j+1),int(2*spin_j+1)))
    Hl[0,0] = si
    Hl[1,0] = sp
    Hl[2,0] = sm
    Hl[3,0] = sz
    Hl[4,0] = hz*sz
    Hl[4,1] = (.5*j[0])*sm
    Hl[4,2] = (.5*j[0])*sp
    Hl[4,3] = j[1]*sz
    Hl[4,4] = si
    return Hl


def mpo_hamil_uniform(nsite, j=None, hz=None, Hl=None, model=None,
              spin_j=.5, sigma=None):
    # hamiltonian in MPO representation
    if not isinstance(Hl, np.ndarray):
        if model == 'XX':
            Hl = _mpo_xx_1d(j, spin_j=spin_j, sigma=sigma)
        elif model == 'XXZ':
            Hl = _mpo_xxz_1d(j, hz, spin_j=spin_j, sigma=sigma)
        elif model in {'XYZ', 'heisenberg'}:
            Hl = _mpo_xyz_1d(j, hz, spin_j=spin_j, sigma=sigma)
        else:
            raise NotImplementedError('local Hamiltonian is not defined')

    ntot = np.prod(nsite)
    H = [Hl for l in range(ntot)]
    # apply boundaries
    H[0] = H[0][-1:] # keep same n-dimension
    H[ntot-1] = H[ntot-1][:,0:1] # keep same n-dimension

    print('MPO H shape:\n', [m.shape for m in H])
    return H


def mpo_hamil_disordered(nsite, j=None, hz=None, model=None,
                         spin_j=.5, sigma=None):
    if sigma is None:
        sigma = get_spins('0+-z', j=spin_j, np_matrix=True)

    ntot = np.prod(nsite)
    H = [None]*ntot

    for i in range(ntot):
        _j, _hz = j[i], hz[i]
        # hamiltonian in MPO representation
        if model == 'XX':
            Hl = _mpo_xx_1d(_j, spin_j=spin_j, sigma=sigma)
        elif model == 'XXZ':
            Hl = _mpo_xxz_1d(_j, _hz, spin_j=spin_j, sigma=sigma)
        elif model in {'XYZ', 'heisenberg'}:
            Hl = _mpo_xyz_1d(_j, _hz, spin_j=spin_j, sigma=sigma)
        else:
            raise NotImplementedError('local Hamiltonian is not defined')
        H[i] = Hl

    # apply boundaries
    H[0] = H[0][-1:] # keep same n-dimension
    H[ntot-1] = H[ntot-1][:,0:1] # keep same n-dimension

    print('MPO H shape:\n', [m.shape for m in H])
    return H


def get_hamil_from_mpo(h_mpo):
    # check the MPO matrices from the mpo spin function
    # by comparing to the brute-force hamiltonian matrix in Hilbert space
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
    Taux = h_mpo[0][0] # the left leg of the first matrix has dimension of 1
    for l in range(1, ntot):
        # keep the top/bottom indices together
        Taux = np.einsum('ijk,ilmn->ljmkn', Taux, h_mpo[l])
        ndims = Taux.shape # flexiable dimensions
        Taux = Taux.reshape(ndims[0], ndims[1]*ndims[2], ndims[3]*ndims[4])

    H = Taux[0] # the last matrix right leg has dimension of 1
    return H


@monitor_performance
def zipper_from_left(Mt, O, Mb, Tl=np.ones((1,1,1))):
    r"""
    Tl legs order from bottom to top, default np.ones((1,1,1)) for boundary site
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
    |                 4             \\---1-i--
    |                 |
    |                 *m
    |                 |
    |                 2
    |                 |
   \\---1---*n*---3---Mb---1-i---
    """
    # contract from bottom to top
    #Taux = np.einsum('imn,npq,pjlm,qlk->ijk', Mb, Tl, O, Mt, optimize=True)
    Taux = np.einsum('imn,npq->impq', Mb, Tl)
    Taux = np.einsum('impq,pjlm->ijql', Taux, O)
    Taux = np.einsum('ijql,qlk->ijk', Taux, Mt)

    return Taux


@monitor_performance
def zipper_from_right(Mt, O, Mb, Tr=np.ones((1,1,1))):
    r"""
    Tr legs order from top to bottom, default np.ones((1,1,1)) for boundary site
    Contract Tl from the right with 1). bra Mt at top,
                                    2). operator O at middle,
                                    3). ket Mt at bottom.
    --i-1---Mt---3---*q*---1--\\
            |                  |
            2                  |
            |                  |
            *l                 |
            |                  |
            3                  |      --1-i-\\
            |                  |             |
    --j-1---O----2---*p*---2---Tr =   --2-j--Tf
            |                  |             |
            4                  |      --3-k--/
            |                  |
            *m                 |
            |                  |
            2                  |
            |                  |
    --k-3---Mb---1---*n*---3---/
    """
    # contract from top to bottom
    # np.einsum('ilq,qpn,jplm,nmk->ijk', Mt, Tr, O, Mb, optimize=True)
    Taux = np.einsum('ilq,qpn->ilpn', Mt, Tr)
    Taux = np.einsum('ilpn,jplm->ijnm', Taux, O)
    Taux = np.einsum('ijnm,nmk->ijk', Taux, Mb)

    return Taux


def dmrg_sweep(h_mpo, mps, Tzip, pick_eig='SA'):
    """
    Tzip is given as the right zipper with indecies ordering from top to bottom
    at every site so that we continue in this function
                          1). sweep from left to right first
                          2). and then from right to left
                          3). return the right zipper in the end
        |             |             |
        3-l           3-m           1-n
        |             |             |
        T---2-*p*-1---O---2-*q*-2---T
        |             |             |
        1-i           4-j           3-k
        |             |             |
    """
    ntot = len(h_mpo)
    e_list = []

    # for Lanczos diagonalization
    from scipy.sparse.linalg import eigsh

    # right sweep from left
    for l in range(ntot):
        """
        l+2 is used for the right because two more zippers at the ends
        and the left-most is labeled by 0 rather than -1
        otherwise one need l-1 and l+1 for the left and right zipper
        """
        #Taux = np.einsum('ipl,p1mj,nqk->ijklmn', Tzip[l], h_mpo[l], Tzip[l+2], optimize=True)
        Taux = np.einsum('ipl,pqmj->ijlmq', Tzip[l], h_mpo[l])
        Taux = np.einsum('ijlmq,nqk->ijklmn', Taux, Tzip[l+2]) # bottom indices first
        ni, nj, nk = Taux.shape[:3]
        H = Taux.reshape(ni*nj*nk, -1) # second index is for top MPS

        # Lanczos diagonalize H to get the lowest energy state
        # v0 is the initial guess
        # which can be SA (smallest algebraic), SM (smallest magnitude)
        e, vec = eigsh(H, k=1, which=pick_eig, v0=mps[l])
        e_list.append(e[0])

        # update MPS wavefunction at site l
        # in the left-to-right sweep, the MPS has to be left-normalized
        u, s, vt = np.linalg.svd(vec.reshape(ni*nj, nk), full_matrices=False)
        mps[l] = u.reshape(ni, nj, -1)
        if l < ntot-1:
            mps[l+1] = np.einsum('i,ij,jkl->ikl', s, vt, mps[l+1])

        # update T zipper
        Tzip[l+1] = zipper_from_left(mps[l], h_mpo[l], mps[l].conj().T, Tzip[l])

    # left sweep from right
    for l in range(ntot-1, -1, -1):
        #Taux = np.einsum('ipl,p1mj,nqk->ijklmn', Tzip[l], h_mpo[l], Tzip[l+2], optimize=True)
        Taux = np.einsum('ipl,pqmj->ijlmq', Tzip[l], h_mpo[l])
        Taux = np.einsum('ijlmq,nqk->ijklmn', Taux, Tzip[l+2])
        ni, nj, nk = Taux.shape[:3]
        H = Taux.reshape(ni*nj*nk, -1) # second index is for top MPS

        # Lanczos diagonalize H to get the lowest energy state
        # v0 is the initial guess
        # which can be SA (smallest algebraic), SM (smallest magnitude)
        e, vec = eigsh(H, k=1, which=pick_eig, v0=mps[l])
        e_list.append(e[0])

        # update MPS wavefunction at site l
        # in the right-to-left sweep, the MPS has to be right-normalized
        u, s, vt = np.linalg.svd(vec.reshape(ni, nj*nk), full_matrices=False)
        mps[l] = vt.reshape(-1, nj, nk)
        if l > 0:
            mps[l-1] = np.einsum('ijk,kl,l->ijl', mps[l-1], u, s)

        # update T zipper
        Tzip[l+1] = zipper_from_right(mps[l], h_mpo[l], mps[l].conj().T, Tzip[l+2])

    return np.ravel(e_list), mps, Tzip


def dmrg_opt_gs(h_mpo, mps=None, Tzip=None, nbond=None, pick_eig='SA', nmax=10):
    ntot = len(h_mpo)

    # use mps as bra at top
    if mps is None:
        ndims = [nbond, h_mpo[0].shape[2], nbond]
        mps = mps_canonical(ndims, ntot, normalize='both')

    if Tzip is None:
        # initialize T zipper from right, indexing from top to bottom
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
    expval{sigma_{l}^{+} sigma_{l+1}^{-}}
    """
    if sigma is None:
        sigma = get_spins('0+-')
    mpo_si, mpo_sp, mpo_sm = sigma
    mpo_si = mpo_si.reshape((1,1,2,2))
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
                Taux = zipper_from_left(mps[l], mpo_si, mps[l].conj().T, Taux)
        correlation.append(Taux[0,0,0])

    return np.array(correlation)


if __name__ == '__main__':
    from wavefunction_analysis.utils import set_performance_log
    set_performance_log(debug=True)
    ndims = [160, 3, 160]
    nsite = 10
    M = mps_canonical(ndims, nsite, normalize='both')

    spin_j = .5 # spin-1/2 particle
    sigma = None
    model = 'XX'

    j = 1.

    N = 20
    D = 10
    nmax = 10
    h_mpo = mpo_hamil_uniform(N, j=j, model=model, spin_j=spin_j, sigma=sigma)
    e_list, mps = dmrg_opt_gs(h_mpo, nbond=D, nmax=nmax)
    print_matrix('e_list:', np.array(e_list))
    correlation = mpo_spin_correlation(mps, sigma)
    print_matrix('correlation:', correlation)
