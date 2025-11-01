import qutip as qt
from wavefunction_analysis import np

from wavefunction_analysis.plot import plt

def ssh_hamiltonian(N, t1, t2):
    """Creates the SSH Hamiltonian for N sites."""
    # Create the hopping operators
    hop_intra = [qt.basis(N, i) * qt.basis(N, i + 1).dag() for i in range(0, N - 1, 2)]
    hop_inter = [qt.basis(N, i) * qt.basis(N, i + 1).dag() for i in range(1, N - 1, 2)]

    # Construct the Hamiltonian
    H = -t1 * sum(hop_intra) - t2 * sum(hop_inter)
    return H


def create_ssh_hamiltonian(L, ts):
    H = 0
    for n in range(L - 1):
        i = n % 2
        H += -ts[i] * (qt.destroy(L, n + 1) * qt.create(L, n) + qt.create(L, n + 1) * qt.destroy(L, n))
    return H


if __name__ == '__main__':
    N = 10  # Number of sites
    t1 = 1.0  # Intra-cell hopping
    t2 = 0.5  # Inter-cell hopping

    # Initial state (e.g., an electron at the first site)
    psi0 = qt.basis(N, 0)

    # Collapse operators (if you want to include dissipation)
    c_ops = []  # No collapse operators for now

    times = np.linspace(0, 10, 100)  # Time points for simulation

    # Run the Monte Carlo solver
    result = qt.mcsolve(ssh_hamiltonian(N, t1, t2), psi0, times, c_ops, [qt.num(N)])

    # Access the expectation values
    expect_num = result.expect[0]

    # Plot the results
    plt.plot(times, expect_num)
    plt.xlabel("Time")
    plt.ylabel("Expectation value of number operator")
    plt.show()

