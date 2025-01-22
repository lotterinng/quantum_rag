# quantum_rag_pipeline.py

import numpy as np
from quantum_retriever import QuantumRetriever
from classical_llm import ClassicalLLM
from embeddings import EmbeddingManager


class QuantumRAGPipeline:
    """
    Production-like pipeline integrating:
      1. Document Embeddings
      2. Quantum Retrieval (Grover-like)
      3. LLM Generation
    """

    def __init__(self, 
                 documents,
                 embedding_model_name="BAAI/bge-large-en-v1.5",
                 llm_model_name="meta-llama/Llama-3.1-8B"):
        """
        :param documents: List of strings (the corpus).
        :param embedding_model_name: Name of the embedding model.
        :param llm_model_name: Name of the LLM for generation.
        """
        self.documents = documents
        self.embedding_manager = EmbeddingManager(model_name=embedding_model_name)
        self.llm = ClassicalLLM(model_name=llm_model_name)

        self.doc_embeddings = None
        self.q_retriever = None
        self.num_docs = len(documents)

    def build_embeddings(self):
        """
        Build embeddings for the entire corpus using the configured embedding model.
        """
        self.doc_embeddings = self.embedding_manager.embed_documents(self.documents)

    def init_quantum_retriever(self, max_grover_iterations=None):
        """
        Instantiate the QuantumRetriever with multiple Grover iterations if desired.

        :param max_grover_iterations: If provided, limit the number of Grover iterations;
                                      otherwise use floor(pi/4 * sqrt(num_docs)).
        """
        if self.doc_embeddings is None:
            raise ValueError("Must build document embeddings first!")

        embedding_dim = self.doc_embeddings.shape[1]
        self.q_retriever = QuantumRetriever(
            num_docs=self.num_docs,
            embedding_dim=embedding_dim,
            max_grover_iterations=max_grover_iterations
        )

    def run(self, query, top_k=1):
        """
        Executes the quantum RAG flow:
          1. Compute query embedding
          2. Quantum retrieval (Grover-like)
          3. Select top-k docs from measurement distribution
          4. Prompt LLM with context

        :param query: User query string
        :param top_k: Number of documents to retrieve as context
        :return: The LLM-generated answer string
        """
        if self.q_retriever is None:
            raise ValueError("QuantumRetriever not initialized. Call init_quantum_retriever() first.")
        if self.doc_embeddings is None:
            raise ValueError("Document embeddings not built. Call build_embeddings() first.")

        # 1. Compute query embedding
        query_embedding = self.embedding_manager.embed_query(query)

        # 2. Quantum retrieval -> doc_index -> probability
        doc_probability_map = self.q_retriever.encode_documents(
            self.doc_embeddings,
            query_embedding
        )

        # 3. Sort documents by measured probability (descending) and pick top_k
        sorted_docs = sorted(
            doc_probability_map.items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_indices = [idx for (idx, prob) in sorted_docs[:top_k]]
        selected_docs = [self.documents[i] for i in top_indices]

        # Concatenate top-k docs as "context"
        context_text = "\n\n---\n\n".join(selected_docs)

        # 4. Use the classical LLM to generate an answer
        answer = self.llm.generate(context_text, query)
        return answer
