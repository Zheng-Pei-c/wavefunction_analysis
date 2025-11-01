from wavefunction_analysis import np
from wavefunction_analysis.spins import qt # qutip

from wavefunction_analysis.utils import print_matrix
from wavefunction_analysis.plot import plt

_spin_names = ['0', 'x', 'y', 'z', '+', '-']

def get_spin(x, j=.5):
    """
    spin matrices from qutip package
    """
    # x takes 0, x, y, z, +, -
    if x not in _spin_names:
        raise ValueError('spin matrix variable should be one of {0, x, y, z, +, -}!')

    if x=='0':
        return qt.qeye(int(2*j+1)) # alas to identity()
    else:
        # sigmax(/y/z) function is specific for spin-1/2
        # but gives Pauli matrices, i.e. without .5 factor
        #getattr(qt, 'sigma'+x)()
        return qt.jmat(j, x)


def get_spin_mat(x, j=.5):
    """
    reload get_spin() but return numpy matrix by qutip full() function
    """
    # x takes 0, x, y, z, +, -
    if x not in _spin_names:
        raise ValueError('spin matrix variable should be one of {0, x, y, z, +, -}!')

    if x=='0':
        return qt.qeye(int(2*j+1)).full() # alas to identity()
    else:
        return qt.jmat(j, x).full()


def get_spins(xs='all', j=.5, np_matrix=True):
    if type(xs) is str:
        if xs == 'all':
            xs = _spin_names
        else:
            xs = xs.strip()

    sigma = []

    if np_matrix:
        for x in xs:
            sigma.append(get_spin_mat(x, j))
    else:
        for x in xs:
            sigma.append(get_spin(x, j))

    return sigma


def get_prod_spin_list(n, xs='all', j=.5, np_matrix=True):
    r"""
    the actual \sigma_{ix} matrix:
        loop over n spins
        set i-th spin with spin matrices while the rest are identity matrices
        return n*d supermatrices as Hilbert space operators or basis
    the use of these matrices gives exact Hamiltonian in rather huge dimension
    """
    sigma = get_spins(xs, j, np_matrix=False) # use qutip type here

    spin_list = [None] * n
    for i in range(n):
        op_list = [qt.qeye(int(2*j+1))] * n
        _list = []
        for s in sigma:
            op_list[i] = s
            _list.append(qt.tensor(op_list))
        spin_list[i] = _list

    if np_matrix: # get numpy matrices
        # naively loopover
        for i, spins in enumerate(spin_list):
            for s, spin in enumerate(spins):
                spin_list[i][s] = spin.full()
        spin_list = np.array(spin_list)

    return spin_list


def hamil_xxz_1d(n, j, delta, hz, np_matrix=True):
    r"""
    build xxz 1d-chain spin model hamiltonian H in Hilbert space
    n: number of 1/2 spins
    j: spin coupling constant of xx and yy sigma
    delta: j*delta is the coupling constant of zz sigma
    hz: magnetic field strength along z axis for each spin
    """
    si, sx, sy, sz = get_spins(xs='0xyz', np_matrix=False) # use qutip

    # generalize the parameters to arrays
    # built-in array is faster--so am I told
    if isinstance(j, float):
        j = [j] * (n-1)
    if isinstance(delta, float):
        delta = [delta] * (n-1)
    if isinstance(hz, float):
        hz = [hz] * n

    # build the product state on-the-fly by qutip.tensor() function
    H = 0.
    for i in range(n-1):
        H += j[i] * (
                      qt.tensor([si]*i + [sx, sx] + [si]*(n-i-2))
                    + qt.tensor([si]*i + [sy, sy] + [si]*(n-i-2))
                    + qt.tensor([si]*i + [sz, sz] + [si]*(n-i-2)) * delta[i]
                    )

    for i in range(n):
        H += hz[i] * qt.tensor([si]*i + [sz] + [si]*(n-i-1))

    if np_matrix:
        H = H.full()
    return H



if __name__ == '__main__':
    n = 7
    j = 3.
    delta = .5
    hz = 2.

    si, sx, sy, sz = get_spins('0xyz', np_matrix=False)
    spin_list = get_prod_spin_list(n)

    H = hamil_xxz_1d(n, j, delta, hz, np_matrix=False)
    eigenvalues = H.eigenenergies()
    if n < 5:
        print_matrix('eigenvalues:', eigenvalues)


    # use qutip to find steady state
    # requires collapse operators, dephasing on the edges
    dephasing = np.sqrt(.1)

    # define collapse operators (e.g., dissipation at the edges)
    c_ops = []
    # add a local dephasing channel at site 0
    c_ops.append(dephasing * qt.tensor([sz] + [si]*(n-1)))
    # adding a local dephasing channel at site N-1
    c_ops.append(dephasing * qt.tensor([si]*(n-1) + [sz]))

    # find the steady state
    rho_ss = qt.steadystate(H, c_ops)

    # calculate observables in the steady state
    magnetization_z = qt.expect(qt.tensor([sz] + [si]*(n-1)), rho_ss)
    print('Steady-state magnetization at site 0:', magnetization_z)
