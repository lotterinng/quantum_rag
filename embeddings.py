from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import numpy as np

class EmbeddingManager:
    """
    Manages embedding logic for documents and queries using a Llama embedding model.
    """

    def __init__(self, model_name):
        """
        Initializes the Llama embedding model.
        
        :param model_name: Name of the Llama embedding model.
        """
        self.model_name = model_name
        self.model = HuggingFaceEmbedding(model_name=model_name)

    def embed_documents(self, documents):
        """
        Embeds a list of documents using the Llama embedding model.

        :param documents: List of text docs
        :return: np.array of shape (num_docs, embedding_dim)
        """
        embeddings = [self.model.get_text_embedding(doc) for doc in documents]
        return np.array(embeddings)

    def embed_query(self, query):
        """
        Embeds a single query using the Llama embedding model.

        :param query: Single query string
        :return: np.array of shape (embedding_dim,)
        """
        embedding = self.model.get_text_embedding(query)
        return np.array(embedding)
