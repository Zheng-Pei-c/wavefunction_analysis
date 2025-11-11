from wavefunction_analysis import itertools
from wavefunction_analysis.quantum_computing import cirq, cirq_google

from cirq.contrib.svg import SVGCircuit

def create_qubits_2d(nrow, ncol):
    """Initialize qubits in a 2D grid."""
    return [[cirq.GridQubit(r, c) for c in range(ncol)] for r in range(nrow)]


def create_circuit(qubits, gamma=None, beta=None, **kwargs):
    """
    Create a quantum circuit with Hadamard gates on each qubit for
    equal superposition as real initial state.
    gamma: 𝛾 symbol holder as a parameter for U(gamma, C) operator
    beta: ß symbol holder as a parameter for U(beta, B) operator
    h: 2D array of magnetic field values
    return: cirq.Circuit object
    """
    circuit = cirq.Circuit(cirq.H.on_each(qubits))

    if gamma: # U(gamma, C) operator
        h = kwargs['h_field']
        circuit.append(gamma_layer(qubits, gamma, h))

    if beta: # U(beta, B) operator
        circuit.append(beta_layer(qubits, beta), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    return circuit


def gamma_layer(qubits, gamma, h):
    """Define the gamma layer of the circuit."""
    nrow, ncol = len(qubits), len(qubits[0])

    for (r, c) in itertools.product(range(nrow), range(ncol)):
        if r < nrow - 1:
            yield cirq.ZZ(qubits[r][c], qubits[r+1][c]) ** gamma
        if c < ncol - 1:
            yield cirq.ZZ(qubits[r][c], qubits[r][c+1]) ** gamma
        yield cirq.Z(qubits[r][c]) ** (gamma * h[r,c])


def beta_layer(qubits, beta):
    """Define the beta layer of the circuit."""
    nrow, ncol = len(qubits), len(qubits[0])

    for (r, c) in itertools.product(range(nrow), range(ncol)):
        yield cirq.X(qubits[r][c]) ** beta



if __name__ == '__main__':
    circuit = cirq.Circuit()

    q0 = cirq.GridQubit.square(5)
    print('q0:', q0)

    circuit.append(cirq.H(q) for q in q0)

    print(circuit)
