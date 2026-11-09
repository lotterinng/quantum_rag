"""Grover-style amplitude amplification over document similarity scores.

Similarity scores are computed classically and loaded into a state over
ceil(log2(N)) index qubits. Amplitude amplification then boosts the top-k
documents. The circuit runs on a simulator. Because the oracle is built from
classically computed scores, this demonstrates the technique and offers no
speedup over sorting the scores.
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator


class QuantumRetriever:
    def __init__(
        self,
        num_docs: int,
        embedding_dim: int,
        max_grover_iterations: Optional[int] = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        :param max_grover_iterations: Upper limit; the optimal count is used if lower.
        :param shots: Sample measurements if set, otherwise use exact probabilities.
        """
        if num_docs < 1:
            raise ValueError("QuantumRetriever needs at least one document.")
        self.num_docs = num_docs
        self.embedding_dim = embedding_dim
        self.max_grover_iterations = max_grover_iterations
        self.shots = shots
        self.num_qubits = math.ceil(math.log2(num_docs)) if num_docs > 1 else 0
        self.simulator = AerSimulator(method="statevector", seed_simulator=seed)
        self.last_run: Dict[str, float] = {}

    @staticmethod
    def _similarities(doc_embeddings, query_embedding) -> np.ndarray:
        docs = np.asarray(doc_embeddings, dtype=float)
        query = np.asarray(query_embedding, dtype=float)
        norms = np.linalg.norm(docs, axis=1) * np.linalg.norm(query)
        dots = docs @ query
        scores = np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)
        return np.clip(scores, 0.0, None)

    def _phase_flip(self, circuit: QuantumCircuit, index: int) -> None:
        """Flip the phase of one computational basis state."""
        n = self.num_qubits
        zeros = [q for q in range(n) if not (index >> q) & 1]
        if zeros:
            circuit.x(zeros)
        if n == 1:
            circuit.z(0)
        else:
            circuit.h(n - 1)
            circuit.mcx(list(range(n - 1)), n - 1)
            circuit.h(n - 1)
        if zeros:
            circuit.x(zeros)

    def _build_circuit(self, amplitudes: np.ndarray, marked: Sequence[int], iterations: int):
        n = self.num_qubits
        preparation = QuantumCircuit(n)
        preparation.append(StatePreparation(amplitudes), range(n))
        preparation = transpile(preparation, basis_gates=["u", "cx"], optimization_level=0)
        unpreparation = preparation.inverse()

        oracle = QuantumCircuit(n)
        for index in marked:
            self._phase_flip(oracle, int(index))
        zero_reflection = QuantumCircuit(n)
        self._phase_flip(zero_reflection, 0)

        circuit = preparation.copy()
        for _ in range(iterations):
            circuit.compose(oracle, inplace=True)
            circuit.compose(unpreparation, inplace=True)
            circuit.compose(zero_reflection, inplace=True)
            circuit.compose(preparation, inplace=True)
        return circuit

    def encode_documents(self, doc_embeddings, query_embedding, top_k: int = 1) -> Dict[int, float]:
        """Return {document index: measured probability}."""
        if len(doc_embeddings) != self.num_docs:
            raise ValueError(f"Expected {self.num_docs} documents, got {len(doc_embeddings)}.")

        scores = self._similarities(doc_embeddings, query_embedding)
        if self.num_docs == 1:
            self.last_run = {"qubits": 0, "iterations": 0, "marked_prior": 1.0, "marked_final": 1.0}
            return {0: 1.0}

        # An all-zero score vector means the query shares nothing with the corpus.
        weights = scores if scores.sum() > 0 else np.ones(self.num_docs)
        amplitudes = np.zeros(2**self.num_qubits)
        amplitudes[: self.num_docs] = weights
        amplitudes /= np.linalg.norm(amplitudes)

        order = np.argsort(-scores, kind="stable")
        marked = [int(i) for i in order[: max(1, top_k)] if scores[i] > 0]
        prior = float(np.sum(amplitudes[marked] ** 2))

        iterations = 0
        if marked and prior < 1.0:
            theta = math.asin(math.sqrt(prior))
            iterations = max(0, math.floor(math.pi / (4 * theta)))
            if self.max_grover_iterations is not None:
                iterations = min(iterations, self.max_grover_iterations)

        circuit = self._build_circuit(amplitudes, marked, iterations)
        probabilities = self._execute(circuit)
        probabilities = probabilities[: self.num_docs]

        self.last_run = {
            "qubits": self.num_qubits,
            "iterations": iterations,
            "marked_prior": prior,
            "marked_final": float(np.sum(probabilities[marked])) if marked else 0.0,
        }
        return {i: float(p) for i, p in enumerate(probabilities)}

    def _execute(self, circuit: QuantumCircuit) -> np.ndarray:
        run_circuit = circuit.copy()
        if self.shots:
            run_circuit.measure_all()
            compiled = transpile(run_circuit, self.simulator, optimization_level=0)
            counts = self.simulator.run(compiled, shots=self.shots).result().get_counts()
            probabilities = np.zeros(2**self.num_qubits)
            for bits, count in counts.items():
                probabilities[int(bits.replace(" ", ""), 2)] = count / self.shots
            return probabilities

        run_circuit.save_statevector()
        compiled = transpile(run_circuit, self.simulator, optimization_level=0)
        state = np.asarray(self.simulator.run(compiled).result().get_statevector())
        return np.abs(state) ** 2
