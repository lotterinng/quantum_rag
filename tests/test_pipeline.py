import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical_llm import ClassicalLLM
from main import load_documents_from_folder, split_documents_into_chunks
from quantum_rag_pipeline import QuantumRAGPipeline
from quantum_retriever import QuantumRetriever


def corpus(size, best):
    rng = np.random.default_rng(7)
    docs = rng.random((size, 16))
    query = rng.random(16)
    docs[best] = query
    return docs, query


@pytest.mark.parametrize("size,best", [(2, 1), (5, 3), (8, 0), (13, 12), (64, 40)])
def test_grover_amplifies_best_document(size, best):
    docs, query = corpus(size, best)
    retriever = QuantumRetriever(size, 16)
    probabilities = retriever.encode_documents(docs, query)
    assert max(probabilities, key=probabilities.get) == best
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert retriever.last_run["marked_final"] >= retriever.last_run["marked_prior"] - 1e-9


def test_probability_matches_grover_theory():
    docs, query = corpus(64, 40)
    retriever = QuantumRetriever(64, 16)
    probabilities = retriever.encode_documents(docs, query)
    run = retriever.last_run
    theta = np.arcsin(np.sqrt(run["marked_prior"]))
    assert probabilities[40] == pytest.approx(np.sin((2 * run["iterations"] + 1) * theta) ** 2, abs=1e-6)


def test_shots_find_best_document():
    docs, query = corpus(16, 9)
    probabilities = QuantumRetriever(16, 16, shots=4096, seed=11).encode_documents(docs, query)
    assert max(probabilities, key=probabilities.get) == 9


def test_top_k_marks_every_requested_document():
    docs, query = corpus(32, 5)
    docs[21] = query * 2
    probabilities = QuantumRetriever(32, 16).encode_documents(docs, query, top_k=2)
    assert set(sorted(probabilities, key=probabilities.get)[-2:]) == {5, 21}


def test_iteration_limit_is_respected():
    docs, query = corpus(64, 40)
    retriever = QuantumRetriever(64, 16, max_grover_iterations=1)
    retriever.encode_documents(docs, query)
    assert retriever.last_run["iterations"] == 1


def test_query_with_no_overlap_is_safe():
    probabilities = QuantumRetriever(4, 2).encode_documents(np.eye(4, 2), np.zeros(2))
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_pipeline_retrieves_right_passage():
    documents = [
        "Penguins live in the southern hemisphere and eat fish.",
        "The mine recovered 24.1 million carats in 2022.",
        "Grover search offers a quadratic speedup.",
        "Rainfall in the Kalahari varies each year.",
        "Railways carried freight across the region.",
    ]
    pipeline = QuantumRAGPipeline(documents, embedding_backend="tfidf", llm_backend="extractive")
    pipeline.build_embeddings()
    pipeline.init_quantum_retriever()
    assert "24.1 million" in pipeline.run("How many carats were recovered in 2022?")


def test_extractive_answer_joins_wrapped_pdf_lines():
    context = "Intro text.\nThe company recovered 24.1 million\ncarats in 2022. Next sentence."
    assert "24.1 million carats" in ClassicalLLM(backend="extractive").generate(context, "carats recovered in 2022")


def test_extractive_answer_skips_table_rows():
    context = "2020 2021 2022 carats recovered 12.5 7.5 12.9 13.4.\n\nThe mine recovered more carats in 2022 than before."
    assert "The mine" in ClassicalLLM(backend="extractive").generate(context, "carats recovered in 2022")


def test_model_backends_are_wired(monkeypatch):
    class FakeEncoder:
        def __init__(self, name):
            self.texts = []

        def encode(self, texts, **kwargs):
            self.texts.extend(texts)
            return np.ones((len(texts), 4))

    def fake_pipeline(*args, **kwargs):
        return lambda messages, **options: [{"generated_text": " 24.1 million "}]

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeEncoder))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=fake_pipeline))
    pipeline = QuantumRAGPipeline(["one", "two"])
    pipeline.build_embeddings()
    pipeline.init_quantum_retriever()
    assert pipeline.run("carats?") == "24.1 million"
    assert pipeline.embedding_manager.model.texts[-1].startswith("Represent this sentence")


def test_extractive_answer_reports_no_match():
    assert "No matching" in ClassicalLLM(backend="extractive").generate("Only birds here.", "carats?")


def test_chunking_overlaps_and_rejects_bad_settings():
    assert split_documents_into_chunks(["abcdef"], chunk_size=4, chunk_overlap=2) == ["abcd", "cdef"]
    with pytest.raises(ValueError):
        split_documents_into_chunks(["abc"], chunk_size=2, chunk_overlap=2)


def test_lfs_pointer_is_skipped(tmp_path, capsys):
    (tmp_path / "report.pdf").write_text("version https://git-lfs.github.com/spec/v1\n")
    (tmp_path / "note.txt").write_text("hello")
    assert load_documents_from_folder(str(tmp_path)) == ["hello"]
    assert "git lfs pull" in capsys.readouterr().out
