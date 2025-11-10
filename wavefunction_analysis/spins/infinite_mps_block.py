from wavefunction_analysis import np
from wavefunction_analysis.utils import print_matrix

from scipy.sparse.linalg import LinearOperator, eigsh
from numpy import linalg as LA

from wavefunction_analysis.utils import monitor_performance

r"""
mu and nu indicate the left and right envrionment density matrices
"""

@monitor_performance
def contract_from_left(mu_ba, s_ab, s_ba, mps_a, mps_b, pick_eig='LM'):
    r"""
    update the environment density matrix at left by contracting it
           with one more 2-sites (A and B) unit cell
    mu_ba: density matrix from the left connects A site via s_ba bond,
           bottom-up indexed
    mu_ab: also obtained from updated mu_ba and A site
    s_ab:  weight vector connects site A (left) to B (right)
    s_ba:  weight vector connects site B (left) to A (right)
    mps_a: MPS state at site A, bra at top
    mps_b: MPS state at site B, bra at top
           /-----2---s_{BA}---4---At---7---s_{AB}----9---Bt---12---
           |                      |                      |
         mu_{BA}                  5                      10
           |                      |                      |
          \\-----1---s_{BA}---6---Ab---3---s_{AB}---11---Bb----8---
    it is equivalent to
           /-----2---s_{BA}---4---A----7---s_{AB}---9---B----12---
           |                      |                     |
         mu_{BA}                  5                     10
           |                      |                     |
          \\-----1---s_{BA}---3---A*---6---s_{AB}---8---B*---11---
    we use the following summation as weigths are vectors (or diagonal)
           /-----2-j--s_{BA}---j---A---5-m---s_{AB}---m---B----8-p---
           |                       |                      |
         mu_{BA}                   3-k                    6-n
           |                       |                      |
          \\-----1-i--s_{BA}---i---A*--4-l---s_{AB}---l---B*---7-o---
    """

    chi_ba = mps_a.shape[0]
    if mu_ba.shape[0] == chi_ba:
        v0 = mu_ba.flatten() # return an independent copy
    else:
        v0 = (np.eye(chi_ba)/chi_ba).ravel() # change in place

    def left_iter(mu_ba):
        # mu_ba would be iteratively solved from scipy eigs function
        return np.einsum('ij,i,j,ikl,jkm,l,m,lno,mnp->op',
                         mu_ba.reshape(chi_ba,chi_ba), s_ba, s_ba,
                         mps_a.conj(), mps_a,
                         s_ab, s_ab, mps_b.conj(), mps_b,
                         optimize=True).reshape(-1,1)
    d, mu_ba = eigsh(LinearOperator((chi_ba**2,chi_ba**2), matvec=left_iter),
                     k=1, which=pick_eig, v0=v0)

    # normalize the environment density matrix
    mu_ba = mu_ba.reshape(chi_ba, chi_ba)
    mu_ba = mu_ba / np.trace(mu_ba)

    # compute mu_ab density matrix
    mu_ab = np.einsum('ij,i,j,ikl,jkm->lm',
                      mu_ba, s_ba, s_ba, mps_a.conj(), mps_a, optimize=True)
    mu_ab = mu_ab / np.trace(mu_ab)

    return mu_ba, mu_ab


@monitor_performance
def contract_from_right(nu_ab, s_ab, s_ba, mps_a, mps_b, pick_eig='LM'):
    r"""
    update the environment density matrix at right by contracting it
           with one more 2-sites (A and B) unit cell
    nu_ab: density matrix from the right connects A site via s_ab bond,
           top-down indexed
    nu_ba: also obtained from B site and updated nu_ab
    s_ab:  weight vector connects site A (left) to B (right)
    s_ba:  weight vector connects site B (left) to A (right)
    mps_a: MPS state at site A, bra at top
    mps_b: MPS state at site B, bra at top
            ----8---Bt---11---s_{BA}---3---At---6---s_{AB}---1----\\
                    |                      |                       |
                    10                     5                    nu_{AB}
                    |                      |                       |
            ---12---Bb----9---s_{BA}---7---Ab---4---s_{AB}---2-----/
    it is equivalent to
           ---11---BT----8---s_{BA}---6---AT----3---s_{AB}---1----\\
                   |                      |                        |
                   10                     5                     nu_{AB}
                   |                      |                        |
           ---12---B*T---9---s_{BA}---7---A*T---4---s_{AB}---2-----/
    we use the following summation as weigths are vectors (or diagonal)
        ---o-7---At---l---s_{BA}---l-4---At---i---s_{AB}---i-1----\\
                 |                       |                         |
                 6-n                     3-k                     nu_{AB}
                 |                       |                         |
        ---p-8---Ab---m---s_{BA}---m-5---Ab---j---s_{AB}---j-2-----/
    """

    chi_ab = mps_a.shape[2]
    if nu_ab.shape[0] == chi_ab:
        v0 = nu_ab.flatten() # return an independent copy
    else:
        v0 = (np.eye(chi_ab)/chi_ab).ravel() # change in place

    mps_a, mps_b = mps_a.T, mps_b.T

    def right_iter(mu_ba):
        # nu_ab would be iteratively solved from scipy eigs function
        return np.einsum('ij,i,j,ikl,jkm,l,m,lno,mnp->op',
                         nu_ab.reshape(chi_ab,chi_ab), s_ab, s_ab,
                         mps_a, mps_a.conj(),
                         s_ba, s_ba, mps_b, mps_b.conj(),
                         optimize=True).reshape(-1,1)
    d, nu_ab = eigsh(LinearOperator((chi_ab**2,chi_ab**2), matvec=right_iter),
                     k=1, which=pick_eig, v0=v0)

    # normalize the environment density matrix
    nu_ab = nu_ab.reshape(chi_ab, chi_ab)
    nu_ab = nu_ab / np.trace(nu_ab)

    # compute nu_ba density matrix
    nu_ba = np.einsum('ij,i,j,ikl,jkm->lm',
                      nu_ab, s_ab, s_ab, mps_a, mps_a.conj(), optimize=True)
    nu_ba = nu_ba / np.trace(nu_ba)

    return nu_ab, nu_ba


@monitor_performance
def update_ortho_mps(rho_left, rho_right, weight, mps_left, mps_right,
                     tol=1e-12):
    r"""
    set the dimer AB or BA linker to cononical form
    update the weight connecting A and B sites and the MPSs
    the left and right rho and weight should have same order:
                    mu_ab, nu_ab, s_ab; or mu_ba, nu_ba, s_ba
            /-----s_{AB}----\\                    /-----s_{BA}----\\
            |                |                    |                |
          mu_{AB}^T        nu_{AB}      or      mu_{BA}          nu_{BA}^T
            |                |                    |                |
           \\-----s_{AB}-----/                   \\-----s_{BA}-----/

                           /---4-l---s_{AB}---l-4--\\
                           |                        |
    ---1-i---At---3-k--- mu_{AB}                  nu_{AB}---5-m---Bt---7-o---
           |                                                      |
           2-j                                                    6-n
           |                                                      |
    left rho is bottom-up indexed while right rho is top-down indexed
    first step: orthogonalize density matrices and form new weight
    second step: svd of new weight and construct new MPSs
    A * mu * s * \nu * B = A * l_U * (l_e * l_U^T * s * r_U * r_e) * r_U^T * B
                         = (A * l_U /l_e * U_s) * tilde{s} * (Vt_s /r_e * r_U^T * B)
    the third dimension of A and first dimension of B might be reduced
    """
    def diag(rho):
        e, vec = LA.eigh(rho)
        idx = np.where(e > tol)[0][::-1] # ordering from large to small
        e, vec = e[idx], vec[:,idx]
        return np.sqrt(e), vec

    # diagonalize environment density matrices
    # square-rooted eigenvalues
    l_e, l_vec = diag(rho_left)
    r_e, r_vec = diag(rho_right)

    # update weight matrix (s_ab or s_ba)
    # with sqrt(eigenvalues) of the density matrix
    weight = np.einsum('i,ji,j,jk,k->ik', l_e, l_vec.conj(), weight,
                                          r_vec, r_e, optimize=True)

    u, s, vt = LA.svd(weight, full_matrices=False)
    weight = s / LA.norm(s)

    # update MPSs at A and B sites
    mps_left = np.einsum('ijk,kl,l,lm->ijm', mps_left, l_vec, 1./l_e, u, optimize=True)
    mps_right = np.einsum('ij,j,kj,klm->ilm', vt, 1./r_e, r_vec.conj(), mps_right, optimize=True)

    return weight, mps_left, mps_right


@monitor_performance
def normalize_mps(s_left, s_right, mps):
    r"""
    normalize the MPS at A or B site
        /---1---s_{BA}---2---At---5---s_{AB}---7--\\
        |                    |                     |
        1                    4                     7
        |                    |                     |
       \\---1---s_{BA}---6---Ab---3---s_{AB}---7---/
        since weight vectors are diagonal, it reduces to
        /---1---s_{BA}---1---A----3---s_{AB}---3--\\
        |                    |                     |
        1                    2                     3
        |                    |                     |
       \\---1---s_{BA}---1---A*---3---s_{AB}---3---/
    s_left and s_right are switched for site B
    """
    s_left, s_right = s_left**2, s_right**2
    norm = np.einsum('i,ijk,ijk,k->', s_left, mps, mps.conj(), s_right, optimize=True)
    return mps/np.sqrt(norm)


@monitor_performance
def normalize_mps_2(s_ab, s_ba, mps_a, mps_b):
    """
    normalize mps_a and mps_b at same time
    same functionality as the previous normalize_mps()
    """
    s_ab, s_ba = s_ab**2, s_ba**2

    # MPS_A norm
    norm = np.einsum('i,ijk,ijk,k->', s_ba, mps_a, mps_a.conj(), s_ab, optimize=True)
    mps_a /= np.sqrt(norm)

    # MPS_B norm
    norm = np.einsum('i,ijk,ijk,k->', s_ab, mps_b, mps_b.conj(), s_ba, optimize=True)
    mps_b /= np.sqrt(norm)
    return mps_a, mps_b


@monitor_performance
def get_mps_2rdm(s_ab, s_ba, mps_a, mps_b):
    r"""
    calculate the density matrix of the two-sites in a unit cell
    MPS is assumed in canoncial form (orthonormal)
    rho: ketbra{ket}{bra} = ketbra{Mb}{Mt} = ketbra{26}{37}
    rho_AB:
                           2-j                     6-n
                           |                       |
      /---1---s_{BA}---1---A*---4-l---s_{AB}---4---B*---8---s_{BA}---8---\
      |                                                                  |
      1-i                                                                8-p
      |                                                                  |
      /---1---s_{BA}---1---A----5-m---s_{AB}---5---B----8---s_{BA}---8---/
                           |                       |
                           3-k                     7-o
    rho_BA: switch A and B when calling the function
    """
    s_ab2, s_ba2 = s_ab**2, s_ba**2

    rho_ab = np.einsum('i,ijl,ikm,l,m,lnp,mop,p->jnko',
                       s_ba2, mps_a.conj(), mps_a, s_ab, s_ab,
                       mps_b.conj(), mps_b, s_ba2, optimize=True)

    rho_ba = np.einsum('i,ijl,ikm,l,m,lnp,mop,p->jnko',
                       s_ab2, mps_b.conj(), mps_b, s_ba, s_ba,
                       mps_a.conj(), mps_a, s_ab2, optimize=True)

    return rho_ab, rho_ba


@monitor_performance
def evaluate_energy_mps(mpo_ab, mpo_ba, s_ab, s_ba, mps_a, mps_b):
    r"""
    get energy of the two sites A and B
    use MPS and weights directly rather than build four-index density matrices
    E_AB:
      /---1---s_{BA}---1---A----4-l---s_{AB}---4---B----8---s_{BA}---8---\\
      |                    |                       |                     |
      |                   2-j--------\\   /--------6-n                    |
      |                              |   |                               |
      1-i                            H_{AB}                             8-p
      |                              |   |                               |
      |                   3-k--------/  \\--------7-o                    |
      |                    |                       |                     |
     \\---1---s_{BA}---1---A*---5-m---s_{AB}---5---B*---8---s_{BA}---8---/
    E_BA: switch A and B
    """
    s_ab2, s_ba2 = s_ab**2, s_ba**2

    e_ab = np.einsum('i,ijl,ikm,l,m,jnko,lnp,mop,p->',
                     s_ba2, mps_a, mps_a.conj(), s_ab, s_ab, mpo_ab,
                     mps_b, mps_b.conj(), s_ba2, optimize=True)

    e_ba = np.einsum('i,ijl,ikm,l,m,jnko,lnp,mop,p->',
                     s_ab2, mps_b, mps_b.conj(), s_ba, s_ba, mpo_ba,
                     mps_a, mps_a.conj(), s_ab2, optimize=True)

    return .5*(e_ab+e_ba)


def evaluate_energy_rdm(mpo_ab, mpo_ba, rho_ab, rho_ba):
    """
    use the density matrix from get_mps_2rdm function
    """
    e_ab = np.einsum('jnko,jnko->', mpo_ab, rho_ab, optimize=True)
    e_ba = np.einsum('jnko,jnko->', mpo_ba, rho_ba, optimize=True)

    return .5*(e_ab+e_ba)


@monitor_performance
def apply_gate_on_mps(gate_ab, s_ab, s_ba, mps_a, mps_b, chi, tol=1e-7):
    r"""
    apply a gate from left to AB dimer MPSs (bra side) for time evolution
    use conjugate of evolution operator for the bra state
    gate_ab: e^{1j*t*ham_ab} for real-time or e^{-tau*ham_ab} for imaginary-time
        ---1---ss_{BA}---1---A---3---s_{AB}---3---B---5---ss_{BA}---5---
                             |                    |
                             2-------\\  /--------4
                                      |  |
                          -----6-----U_{AB}-----7-----
    above contraction gives a matrix whose indices in order 1675,
                            first and last are weight dimensions,
                            while the middle two are the physical dimensions
    subjected to svd to form new MPSs and weights s_ab
    gate_ba: switch A and B indices when calling this function
             so that update MPSs and s_ba
    """
    # ensure weights are above tolerance threshold
    s_ba_trim = s_ba * (s_ba > tol) + tol * (s_ba < tol)

    nd = mps_a.shape[1] # real physical dimension, depending on spin type
    chi_ba = s_ba_trim.shape[0]
    mps = np.einsum('i,ijk,k,klm,m,jlno->inom', s_ba_trim, mps_a, s_ab, mps_b,
                    s_ba_trim, gate_ab, optimize=True)

    u, s, vt = LA.svd(mps.reshape(nd*chi_ba, -1), full_matrices=False)
    idx = range(min(chi, len(s))) # truncate space!!! # s in descending order
    s, u, vt = s[idx], u[:,idx].reshape(chi_ba, -1), vt[idx].reshape(-1, chi_ba)

    # update weights
    s_ab = s / LA.norm(s)

    # update MPSs removing environment weights
    s_ba_trim = 1./s_ba_trim
    mps_a = np.einsum('i,ij->ij', s_ba_trim, u).reshape(chi_ba, d, -1)
    mps_b = np.einsum('ij,j->ij', vt, s_ba_trim).reshape(-1, d, chi_ba)

    return s_ab, mps_a, mps_b


def build_evolution_gate(ham, tau, itype='imag'):
    # ham has dimension as (d,d,d,d)
    from scipy.linalg import expm
    fac = 1j if itype=='real' else -1.

    d = ham.shape[0]
    return expm(fac*tau*ham.reshape(d**2,-1)).reshape(d,d,d,d)


def run_tebd(ham_ab, ham_ba, mps_a, mps_b, s_ab, s_ba,
             mu_ba=None, nu_ab=None, tau=.1, itype='imag',
             nmax=1000, midstep=10):
    r"""
    run time evolving block decimation based on MPS of infinite two-site unit cells
    refer: https://doi.org/10.1103/PhysRevLett.93.040502
           https://www.tensors.net/mps
    """
    gate_ab = build_evolution_gate(ham_ab, tau, itype=itype)
    gate_ba = build_evolution_gate(ham_ba, tau, itype=itype)

    # initialize envrionment density matrices
    chi1, n, chi2 = mps_a.shape
    mu_ba = np.eye(chi1) / chi1
    nu_ab = np.eye(chi2) / chi2

    e_tot = []
    for k in range(nmax + 1):
        if np.mod(k, midstep) == 0 or (k == nmax):
            # bring MPS to the normal form

            # contract environment density, weights and MPSs
            mu_ba, mu_ab = contract_from_left(mu_ba, s_ab, s_ba, mps_a, mps_b)
            nu_ab, nu_ba = contract_from_right(nu_ab, s_ab, s_ba, mps_a, mps_b)

            # orthogonalize MPSs and weights
            s_ba, mps_b, mps_a = update_ortho_mps(mu_ba, nu_ba, s_ba, mps_b, mps_a)
            s_ab, mps_a, mps_b = update_ortho_mps(mu_ab, nu_ab, s_ab, mps_a, mps_b)

            # normalize MPSs
            mps_a, mps_b = normalize_mps_2(s_ab, s_ba, mps_a, mps_b)

            # compute energy
            rho_ab, rho_ba = get_mps_2rdm(s_ab, s_ba, mps_a, mps_b)
            e = evaluate_energy_rdm(ham_ab, ham_ba, rho_ab, rho_ba)
            #e = evaluate_energy_mps(ham_ab, ham_ba, s_ab, s_ba, mps_a, mps_b)
            e_tot.append(e)
            print('k:', k, 'e:', e)
            print('k:', k, s_ab.shape, s_ba.shape, mps_a.shape, mps_b.shape, mu_ab.shape, mu_ba.shape, nu_ab.shape, nu_ba.shape)

        # MPS time evolutions
        s_ab, mps_a, mps_b = apply_gate_on_mps(gate_ab, s_ab, s_ba, mps_a, mps_b, chi)
        s_ba, mps_b, mps_a = apply_gate_on_mps(gate_ba, s_ba, s_ab, mps_b, mps_a, chi)

    rho_ab, rho_ba = get_mps_2rdm(s_ab, s_ba, mps_a, mps_b)
    return s_ab, s_ba, mps_a, mps_b, rho_ab, rho_ba, e_tot



if __name__ == '__main__':
    chi = 160  # bond dimension
    d = 2

    from wavefunction_analysis.utils import set_performance_log
    set_performance_log(debug=True)

    from wavefunction_analysis.spins import get_spins
    sx, sy = get_spins('xy', j=.5)
    sx, sy = sx, sy
    ham_ab = (np.real(np.kron(sx, sx) + np.kron(sy, sy))).reshape(2, 2, 2, 2)
    ham_ba = (np.real(np.kron(sx, sx) + np.kron(sy, sy))).reshape(2, 2, 2, 2)

    tau = .1

    """
    exact energy is 1./pi
    """

    # weights
    s_ab = np.ones(chi) / np.sqrt(chi)
    s_ba = np.ones(chi) / np.sqrt(chi)
    # MPSs
    mps_a = np.random.rand(chi, d, chi)
    mps_b = np.random.rand(chi, d, chi)

    s_ab, s_ba, mps_a, mps_b, rho_ab, rho_ba, e_tot = run_tebd(
            ham_ab, ham_ba, mps_a, mps_b, s_ab, s_ba)
    print_matrix('e:', e_tot)
