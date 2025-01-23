import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer  # Updated import for Aer
from qiskit.circuit.library import MCXGate, Initialize  # Corrected imports
import math

def next_power_of_two(n: int) -> int:
    """
    Returns the smallest power of two >= n.
    For example, next_power_of_two(5) -> 8.
    """
    return 1 << (n - 1).bit_length()


class QuantumRetriever:
    """
    QuantumRetriever encodes a set of document embeddings into a quantum state
    and applies a Grover-like amplification for the most relevant document
    (the single doc with the highest similarity).
    """

    def __init__(self, num_docs, embedding_dim, max_grover_iterations=None):
        """
        :param num_docs: Total number of documents in corpus (for demonstration).
        :param embedding_dim: Dimensionality of the embedding vectors.
        :param max_grover_iterations: If provided, limit the number of Grover iterations.
                                      Otherwise, use ~ floor(pi/4 * sqrt(num_docs)).
        """
        self.num_docs = num_docs
        self.embedding_dim = embedding_dim

        # We'll set self.num_qubits AFTER padding in encode_documents
        self.num_qubits = None

        # Default number of Grover iterations ~ floor(pi/4 * sqrt(N))
        if max_grover_iterations is None:
            self.num_iterations = int(np.floor((np.pi / 4) * np.sqrt(num_docs)))
        else:
            self.num_iterations = max_grover_iterations

    def encode_documents(self, doc_embeddings, query_embedding):
        """
        Build and run a quantum circuit that encodes document embeddings and
        applies a multi-iteration Grover search focusing on the document that has
        the highest similarity to 'query_embedding'.

        :param doc_embeddings: np.array of shape (num_docs, embedding_dim)
        :param query_embedding: np.array of shape (embedding_dim,)
        :return: A dictionary with document indices as keys and
                measured probabilities as values.
        """
        if len(doc_embeddings) != self.num_docs:
            raise ValueError(f"Expected {self.num_docs} documents, got {len(doc_embeddings)}.")

        # Step 1: Compute similarity scores (dot product)
        similarity_scores = np.array([
            np.dot(doc_embeddings[i], query_embedding) for i in range(self.num_docs)
        ])
        marked_doc_index = int(np.argmax(similarity_scores))  # Doc with highest similarity

        # Normalize similarity scores
        norm_factor = np.linalg.norm(similarity_scores)
        if norm_factor == 0:
            amplitudes = np.ones(self.num_docs) / np.sqrt(self.num_docs)
        else:
            amplitudes = similarity_scores / norm_factor

        # Ensure sum of squares ~ 1 before padding
        amplitudes = amplitudes / np.sqrt(np.sum(amplitudes**2))

        # Round amplitudes slightly to reduce floating-point noise
        amplitudes = np.round(amplitudes, decimals=10)

        # ----------------
        # PADDING LOGIC
        # ----------------
        original_length = len(amplitudes)
        desired_length = max(2, next_power_of_two(original_length))  # Ensure at least 2

        if desired_length != original_length:
            padded_amps = np.zeros(desired_length, dtype=amplitudes.dtype)
            padded_amps[:original_length] = amplitudes
            amplitudes = padded_amps
            print(f"Padded amplitudes from {original_length} to {desired_length} to satisfy 2^n requirement.")

        # Ensure re-normalization after padding
        amplitudes = amplitudes / np.sqrt(np.sum(amplitudes**2))

        # Set the qubit count to log2 of the padded length
        self.num_qubits = int(np.log2(len(amplitudes)))

        # Debug: final norm check
        final_norm = np.sum(np.abs(amplitudes)**2)
        print(f"Final amplitude array length: {len(amplitudes)} (2^{self.num_qubits}). Norm={final_norm}")

        # Step 2: Build the quantum circuit
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # Initialize amplitude-encoded states
        init_gate = Initialize(amplitudes)
        qc.compose(init_gate, qubits=range(self.num_qubits), inplace=True)

        # Step 3: Grover Iterations
        for _ in range(self.num_iterations):
            qc = self._apply_grover_oracle(qc, marked_doc_index)
            qc = self._apply_grover_diffuser(qc)

        # Step 4: Measure
        qc.measure(range(self.num_qubits), range(self.num_qubits))

        # Step 5: Simulate the circuit
        simulator = Aer.get_backend('aer_simulator')
        transpiled_qc = transpile(qc, simulator)
        job = simulator.run(transpiled_qc, shots=1024)
        result = job.result()
        counts = result.get_counts()

        # Convert measurements to document probabilities
        doc_probability_map = {}
        for measured_state, freq in counts.items():
            doc_index = int(measured_state, 2)
            probability = freq / 1024.0

            if doc_index < original_length:  # Ignore padded indices
                doc_probability_map[doc_index] = doc_probability_map.get(doc_index, 0) + probability

        return doc_probability_map

    def _apply_grover_oracle(self, qc, marked_doc_index):
        """
        Oracle that flips the phase of the 'marked_doc_index' basis state.
        Handles cases with fewer qubits gracefully.
        """
        if self.num_qubits == 1:
            # Single qubit: Apply Z directly
            qc.z(0)
            return qc

        # Multi-qubit case
        binary_str = format(marked_doc_index, f'0{self.num_qubits}b')

        # Apply X gates for qubits where the marked_doc_index bit is 0
        for i, bit in enumerate(reversed(binary_str)):
            if bit == '0':
                qc.x(i)

        # Multi-controlled Z gate
        control_qubits = list(range(self.num_qubits - 1))
        target_qubit = self.num_qubits - 1
        mcx_gate = MCXGate(len(control_qubits))
        qc.append(mcx_gate, control_qubits + [target_qubit])

        # Uncompute the X gates
        for i, bit in enumerate(reversed(binary_str)):
            if bit == '0':
                qc.x(i)

        return qc

    def _apply_grover_diffuser(self, qc):
        """
        Applies the Grover diffuser, which inverts amplitudes about the mean.
        Handles cases with fewer qubits gracefully.
        """
        if self.num_qubits == 1:
            # Single qubit: Apply Z directly
            qc.z(0)
            return qc

        # Multi-qubit case
        qc.h(range(self.num_qubits))
        qc.x(range(self.num_qubits))

        control_qubits = list(range(self.num_qubits - 1))
        target_qubit = self.num_qubits - 1
        mcx_gate = MCXGate(len(control_qubits))
        qc.append(mcx_gate, control_qubits + [target_qubit])

        qc.x(range(self.num_qubits))
        qc.h(range(self.num_qubits))

        return qc
