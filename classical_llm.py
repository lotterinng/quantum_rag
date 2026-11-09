"""Answer generation: a Hugging Face model, or sentence extraction offline."""

import re
from typing import Optional

STOPWORDS = {
    "a", "an", "and", "are", "did", "do", "does", "for", "how", "in", "is", "many",
    "much", "of", "on", "the", "to", "was", "were", "what", "when", "where", "which", "who",
}


def _terms(text: str) -> set:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS}


def _is_table_row(text: str) -> bool:
    words = text.split()
    return sum(any(ch.isdigit() for ch in word) for word in words) > 0.4 * len(words)


class ClassicalLLM:
    def __init__(self, model_name: Optional[str] = None, max_new_tokens: int = 200, backend: str = "auto"):
        """:param backend: "auto", "transformers" or "extractive"."""
        if backend not in {"auto", "transformers", "extractive"}:
            raise ValueError(f"Unknown LLM backend: {backend}")
        self.max_new_tokens = max_new_tokens
        self.generator = None
        self.backend = "extractive"

        if backend != "extractive" and model_name:
            try:
                from transformers import pipeline

                self.generator = pipeline("text-generation", model=model_name, device_map="auto", torch_dtype="auto")
                self.backend = "transformers"
            except Exception as error:
                if backend == "transformers":
                    raise
                print(f"WARNING: could not load LLM {model_name} ({error}). Using extractive answers.")

    def generate(self, context: str, query: str, do_sample: bool = False) -> str:
        if self.generator is None:
            return self._extract(context, query)
        messages = [
            {"role": "system", "content": "Answer using only the provided context. If the context does not contain the answer, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        output = self.generator(
            messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            return_full_text=False,
        )
        return output[0]["generated_text"].strip()

    @staticmethod
    def _extract(context: str, query: str) -> str:
        """Return the context line or sentence that best overlaps the query."""
        query_terms = _terms(query)
        # PDF text wraps mid-sentence, so join single line breaks before splitting.
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", context)
        candidates = [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+", text) if part.strip()]

        def relevance(part: str):
            overlap = len(query_terms & _terms(part)) if not _is_table_row(part) else 0
            return overlap, overlap / max(1, len(part.split()))

        best = max(candidates, key=relevance, default="")
        if not best or len(query_terms & _terms(best)) * 2 <= len(query_terms) or _is_table_row(best):
            return "[extractive, no LLM] No matching passage found."
        return f"[extractive, no LLM] {best}"
