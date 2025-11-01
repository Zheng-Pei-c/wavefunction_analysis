import cirq
import cirq_google

from cirq.contrib.svg import SVGCircuit

if __name__ == '__main__':
    circuit = cirq.Circuit()

    q0 = cirq.GridQubit.square(5)
    print('q0:', q0)

    circuit.append(cirq.H(q) for q in q0)

    print(circuit)
