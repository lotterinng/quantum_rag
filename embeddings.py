"""Document and query embeddings: sentence-transformers, or TF-IDF offline."""

from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingManager:
    def __init__(self, model_name: Optional[str] = None, backend: str = "auto"):
        """:param backend: "auto", "sentence-transformers" or "tfidf"."""
        if backend not in {"auto", "sentence-transformers", "tfidf"}:
            raise ValueError(f"Unknown embedding backend: {backend}")
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self.backend = "tfidf"

        if backend != "tfidf" and model_name:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(model_name)
                self.backend = "sentence-transformers"
            except Exception as error:
                if backend == "sentence-transformers":
                    raise
                print(f"WARNING: could not load embedding model {model_name} ({error}). Using TF-IDF.")

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        if self.model is not None:
            return self.model.encode(documents, normalize_embeddings=True, show_progress_bar=False)
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        return self.vectorizer.fit_transform(documents).toarray()

    def embed_query(self, query: str) -> np.ndarray:
        if self.model is not None:
            text = BGE_QUERY_PREFIX + query if "bge" in (self.model_name or "").lower() else query
            return self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        if self.vectorizer is None:
            raise ValueError("Call embed_documents() before embed_query().")
        return self.vectorizer.transform([query]).toarray()[0]
