"""Minimal LLM stub used for testing without external dependencies."""

from typing import Optional


class ClassicalLLM:
    """Return a canned answer based on the provided context and query."""

    def __init__(self, model_name: Optional[str] = None, max_new_tokens: int = 200):
        self.max_new_tokens = max_new_tokens

    def generate(self, context: str, query: str, do_sample: bool = True) -> str:
        # For demonstration we simply echo a small part of the context.
        snippet = context.strip().split()
        snippet = " ".join(snippet[:20])
        return f"[Stubbed answer based on context snippet: {snippet}]"
