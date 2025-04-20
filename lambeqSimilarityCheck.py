import numpy as np
from lambeq import BobcatParser, IQPAnsatz, WebParser
from qiskit.quantum_info import Statevector
from pytket.extensions.qiskit import AerBackend

def lambeq_similarity(sentence1, sentence2, ansatz):
    # Parse sentences into DisCoCat diagrams
    parser = BobcatParser(verbose='text')
    diagram1 = parser.sentence2diagram(sentence1)
    diagram2 = parser.sentence2diagram(sentence2)

    # Convert diagrams to quantum circuits
    circuit1 = ansatz(diagram1)
    circuit2 = ansatz(diagram2)

    # Simulate circuits to get state vectors
    backend = AerBackend()
    state1 = backend.run_circuit(circuit1.to_tk()).get_state()
    state2 = backend.run_circuit(circuit2.to_tk()).get_state()

    # Compute fidelity (similarity) between states
    fidelity = np.abs(np.dot(state1.conj(), state2))**2
    return fidelity

if __name__ == '__main__':
    # Example sentences
    sentence1 = "She rode her bicycle through the park."
    sentence2 = "A bicycle was ridden in the park."

    # Define an ansatz (map words to qubits)
    ansatz = IQPAnsatz({}, n_layers=1, n_single_qubit_params=1)

    # Compute similarity
    similarity = lambeq_similarity(sentence1, sentence2, ansatz)
    print(f"Similarity (Fidelity): {similarity:.4f}")