from lumeq import itertools
from lumeq.spins import cirq, cirq_google

from cirq.contrib.svg import SVGCircuit

"""
This module defines functions to create a quantum circuit for qubits using the Cirq library.
cirq.X, cirq.Y, cirq.Z: Pauli gates of spin-half operators
X gate matrix: [[0, 1], [1, 0]]
Y gate matrix: [[0, -i], [i, 0]]
Z gate matrix: [[1, 0], [0, -1]]

cirq.H: Hadamard gate for creating superposition states
H gate matrix: (1/sqrt(2)) * [[1, 1], [1, -1]]
cirq.S: Phase gate, ie, Clifford S gate
S gate matrix: [[1, 0], [0, i]]
cirq.T: T gate (pi/8 gate), ie, non-Clifford T gate
T gate matrix: [[1, 0], [0, exp(i * pi / 4)]]

cirq.Rx, cirq.Ry, cirq.Rz: Rotation gates around X, Y, Z axes with angle theta (in radians)
same as X**(theta/pi), Y**(theta/pi), Z**(theta/pi)
Rx gate matrix: exp(-i * theta/2 * X) = cos(theta/2) * I - i * sin(theta/2) * X
Ry gate matrix: exp(-i * theta/2 * Y) = cos(theta/2) * I - i * sin(theta/2) * Y
Rz gate matrix: exp(-i * theta/2 * Z) = cos(theta/2) * I - i * sin(theta/2) * Z
cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate: Generalized rotation gates
same as Rx, Ry, Rz but with global phase factor exp(i * pi * theta / 2)

cirq.CX: Controlled-X (CNOT) gate
cirq.CZ: Controlled-Z gate
cirq.SWAP: SWAP gate for swapping two qubits

cirq.XX, cirq.YY, cirq.ZZ: Two-qubit interaction gates (tensor products)

cirq.FSimGate: Fermionic simulation gate for fermionic systems
contains all two qubit interactions
"""

def create_qubits_1d(nsite):
    """Initialize qubits in a 1D array."""
    return [cirq.LineQubit(i) for i in range(nsite)]


def create_qubits_2d(nrow, ncol):
    """Initialize qubits in a 2D grid."""
    return [[cirq.GridQubit(r, c) for c in range(ncol)] for r in range(nrow)]


def create_circuit(qubits, gamma=None, beta=None, nlayers=0, **kwargs):
    """
    Create a quantum circuit with Hadamard gates on each qubit for
    equal superposition as real initial state.

    Parameters
        gamma : 𝛾 symbol holder as a parameter for U(gamma, C) operator
        beta : ß symbol holder as a parameter for U(beta, B) operator
        nlayers : number of layers in the circuit (discreted time evolution steps)
        js : coupling strength values between qubits
        hs : magnetic field values
    Returns
        circuit : cirq.Circuit object
    """
    circuit = cirq.Circuit(cirq.H.on_each(qubits))

    if gamma: # U(gamma, C) operator
        js = kwargs['j_coupling']
        hs = kwargs['h_field']
        circuit.append(gamma_layer(qubits, gamma, js, hs))

    if beta: # U(beta, B) operator
        circuit.append(beta_layer(qubits, beta), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    return circuit


def gamma_layer(qubits, gamma, js, hs):
    r"""
    Define the gamma layer of the circuit with cost Hamiltonian.

    Parameters
        qubits : defined cirq Qubit object
        gamma : rotation angle of the targeted Hamiltonian
        js : coupling constants between qubits
        hs : magnetic field constants

    Returns
        circuit with the rotation gates
    """
    nrow, ncol = len(qubits), len(qubits[0])

    for (r, c) in itertools.product(range(nrow), range(ncol)):
        if r < nrow - 1:
            yield cirq.ZZ(qubits[r][c], qubits[r+1][c]) ** (gamma * js[r,c])
        if c < ncol - 1:
            yield cirq.ZZ(qubits[r][c], qubits[r][c+1]) ** (gamma * js[r,c])
        yield cirq.Z(qubits[r][c]) ** (gamma * hs[r,c])


def beta_layer(qubits, beta):
    r"""
    Define the beta layer of the circuit for mixer.

    Parameters
        qubits : defined cirq Qubit object
        beta : rotation angle of the initial hamiltonian \sum_i X_i

    Returns
        circuit with the rotation gates
    """
    nrow, ncol = len(qubits), len(qubits[0])

    for (r, c) in itertools.product(range(nrow), range(ncol)):
        yield cirq.X(qubits[r][c]) ** beta



if __name__ == '__main__':
    from lumeq import np
    from lumeq.spins import sympy
    nsite = [3, 2]
    qubits = create_qubits_2d(*nsite)
    print('qubits:\n', qubits)


    gamma = sympy.Symbol('𝛾')
    beta = sympy.Symbol('ß')
    h_field = 0.5*np.ones(nsite)
    circuit = create_circuit(qubits, gamma=gamma, beta=beta, h_field=h_field)
    print('circuit:\n', circuit)
