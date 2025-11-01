from wavefunction_analysis import np
from wavefunction_analysis.quantum_computing import cirq, sympy
from wavefunction_analysis.plot import plt

class ising_model():
    """
    E = - Z_i Z_j - h_i Z_i
    h is the magnetic field on each spin
    """
    def __init__(self, nsite, h=None):
        self.nsite = nsite
        self.nrow, self.ncol = nsite

        if not isinstance(h, np.ndarray):
            h = .5 * np.ones(self.nsite)
        self.h = h

        # initialize qubits in 2d array
        self.qubits = [[cirq.GridQubit(r, c) for c in range(self.ncol)]
                            for r in range(self.nrow)]
        #print('qubits:\n', qubits)

        # define a quantum circuit
        # apply Hadamard gate
        # equal superposition as real initial state
        self.circuit = cirq.Circuit(cirq.H.on_each(self.qubits))

        # symbolic parameters
        self.gamma, self.beta = sympy.Symbol('𝛾'), sympy.Symbol('ß')

        # U(gamma, C) operator
        self.circuit.append(self.gamma_layer())

        # U(beta, B) operator
        self.circuit.append(self.beta_layer(), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        print('circuit:\n', self.circuit)


    def gamma_layer(self):
        for r in range(self.nrow):
            for c in range(self.ncol):
                if r < self.nrow - 1:
                    yield cirq.ZZ(self.qubits[r][c], self.qubits[r+1][c]) ** self.gamma
                if c < self.ncol - 1:
                    yield cirq.ZZ(self.qubits[r][c], self.qubits[r][c+1]) ** self.gamma
                yield cirq.Z(self.qubits[r][c]) ** (self.gamma * self.h[r,c])


    def beta_layer(self):
        for r in range(self.nrow):
            for c in range(self.ncol):
                yield cirq.X(self.qubits[r][c]) ** self.beta


    def energy_expectation(self, wf):
        """
        wf: wavefunction is an array of size 2**(nrow*ncol)
        """
        nrow, ncol = self.nrow, self.ncol
        nsite = nrow * ncol

        # pauli-z operator diag(1, -1) introduces a phase flip
        # onsite Z has the shape (nsite, 2**nrow*ncol)
        Z = np.array([(-1) ** (np.arange(2**nsite) >> i) for i in range(nsite-1, -1, -1)])

        # neighboring coupling
        ZZ = np.zeros_like(wf)
        for r in range(nrow):
            for c in range(ncol):
                if r < nrow - 1:
                    ZZ += Z[r*ncol+c] * Z[(r+1)*ncol+c]
                if c < ncol - 1:
                    ZZ += Z[r*ncol+c] * Z[r*ncol+(c+1)]

        energy = - ZZ - np.einsum('i,ij->j', self.h.ravel(), Z)

        # expectation value of the energy per site
        return np.einsum('i,i,i->', wf.conj(), wf, energy).real / nsite


    def energy_evaluation(self, gamma, beta):
        # start simulate
        simulator = cirq.Simulator()
        params = cirq.ParamResolver({self.gamma: gamma, self.beta: beta})
        result = simulator.simulate(self.circuit, param_resolver=params)
        wf = result.final_state_vector

        return self.energy_expectation(wf)


    def fd_gradient_evaluation(self, gamma, beta, eps=10**-3):
        # gamma gradient
        grad_g  = self.energy_evaluation(gamma+eps, beta)
        grad_g -= self.energy_evaluation(gamma-eps, beta)
        grad_g /= 2*eps

        # beta gradient
        grad_b  = self.energy_evaluation(gamma, beta+eps)
        grad_b -= self.energy_evaluation(gamma, beta-eps)
        grad_b /= 2*eps

        return grad_g, grad_b


    def optimizer(self, gamma, beta, eps=10**-3, max_steps=150, thresh=10**-5):
        e = self.energy_evaluation(gamma, beta)
        for i in range(max_steps):
            grad_g, grad_b = self.fd_gradient_evaluation(gamma, beta, eps)

            # update circuit parameters
            gamma -= eps * grad_g
            beta  -= eps * grad_b

            energy = self.energy_evaluation(gamma, beta)
            print('step: %3d, gamma: %4.2f, beta: %4.2f, grad_g: %5.3f, grad_b: %5.3f, energy: %8.5f'
                  % (i, gamma, beta, grad_g, grad_b, energy))

            if np.abs(energy - e) < thresh:
                break
            e = energy


if __name__ == '__main__':
    N = [3,2]
    ising = ising_model(N)

    gamma, beta = -.21, -.28
    ising.optimizer(gamma, beta)
