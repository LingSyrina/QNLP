import numpy as np
import pennylane as qml
from lambeq import BobcatParser, IQPAnsatz


def lambeq_to_pennylane_circuit(lambeq_circuit):
    """Convert a Lambeq circuit to a PennyLane quantum function."""

    def circuit():
        # Initialize qubits
        num_qubits = lambeq_circuit.n_qubits
        for i in range(num_qubits):
            qml.Hadamard(wires=i)  # Optional: Add default initialization

        # Apply gates from Lambeq circuit
        for gate in lambeq_circuit:
            if gate.name == 'H':
                qml.Hadamard(wires=gate.qubits)
            elif gate.name == 'Rx':
                qml.RX(gate.parameters[0], wires=gate.qubits)
            elif gate.name == 'Ry':
                qml.RY(gate.parameters[0], wires=gate.qubits)
            elif gate.name == 'Rz':
                qml.RZ(gate.parameters[0], wires=gate.qubits)
            elif gate.name == 'CX':
                qml.CNOT(wires=gate.qubits)
            # Add more gate conversions as needed
        return qml.state()

    return circuit


def pennylane_similarity(sentence1, sentence2, ansatz):
    # Parse sentences
    parser = BobcatParser(verbose='text')
    diagram1 = parser.sentence2diagram(sentence1)
    diagram2 = parser.sentence2diagram(sentence2)

    # Create circuits
    l_circuit1 = ansatz(diagram1)
    l_circuit2 = ansatz(diagram2)

    # Convert to PennyLane circuits
    dev = qml.device("default.qubit", wires=max(l_circuit1.n_qubits, l_circuit2.n_qubits))
    qnode1 = qml.QNode(lambeq_to_pennylane_circuit(l_circuit1), dev)
    qnode2 = qml.QNode(lambeq_to_pennylane_circuit(l_circuit2), dev)

    # Get state vectors
    state1 = qnode1()
    state2 = qnode2()

    # Compute fidelity
    fidelity = np.abs(np.vdot(state1, state2)) ** 2
    return fidelity


if __name__ == '__main__':
    # Example sentences
    sentence1 = "She rode her bicycle through the park."
    sentence2 = "A bicycle was ridden in the park."

    # Create ansatz (same as before)
    ansatz = IQPAnsatz({}, n_layers=1, n_single_qubit_params=1)

    # Calculate similarity
    similarity = pennylane_similarity(sentence1, sentence2, ansatz)
    print(f"PennyLane Similarity (Fidelity): {similarity:.4f}")