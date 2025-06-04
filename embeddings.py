"""Simple bag-of-words embedding utilities.
This minimal implementation avoids any third-party dependencies.
"""

from collections import Counter
from typing import List


class EmbeddingManager:
    """Create rudimentary embeddings using term frequency."""

    def __init__(self, model_name: str = None):
        # model_name kept for API compatibility; it's unused in this stub
        self.vocab = {}

    def _tokenize(self, text: str) -> List[str]:
        # Very naive tokenization on whitespace and lower-casing
        return text.lower().split()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        vocab_set = set(word for tokens in tokenized_docs for word in tokens)
        self.vocab = {word: idx for idx, word in enumerate(sorted(vocab_set))}

        embeddings = []
        for tokens in tokenized_docs:
            vec = [0.0] * len(self.vocab)
            counts = Counter(tokens)
            for word, count in counts.items():
                idx = self.vocab[word]
                vec[idx] = float(count)
            embeddings.append(vec)
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        tokens = self._tokenize(query)
        vec = [0.0] * len(self.vocab)
        counts = Counter(tokens)
        for word, count in counts.items():
            if word in self.vocab:
                idx = self.vocab[word]
                vec[idx] = float(count)
        return vec
