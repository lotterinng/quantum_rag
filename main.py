import argparse
import logging
import os

from pypdf import PdfReader

from quantum_rag_pipeline import QuantumRAGPipeline

logging.getLogger("pypdf").setLevel(logging.ERROR)

LFS_POINTER_PREFIX = b"version https://git-lfs"

def load_documents_from_folder(folder_path=None):
    """
    Loads documents from .txt and .pdf files in the `data` folder relative to the script's location.
    Returns a list of strings, each representing the file contents.
    """
    # Get the current directory of the script
    if folder_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(script_dir, "data")

    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder 'data' does not exist at {folder_path}")

    docs = []
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)

        if filename.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_contents = f.read()
                docs.append(file_contents)

        elif filename.lower().endswith(".pdf"):
            with open(file_path, "rb") as f:
                if f.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
                    print(f"Skipping {filename}: it is a Git LFS pointer. Run `git lfs pull` to download it.")
                    continue
            reader = PdfReader(file_path)
            pdf_text = [page.extract_text() or "" for page in reader.pages]
            docs.append("\n".join(pdf_text))

        else:
            print(f"Skipping file: {filename}")
    
    return docs

def split_documents_into_chunks(documents, chunk_size=512, chunk_overlap=256):
    """
    Splits the given list of document texts into smaller chunks for easier processing.

    :param documents: List of strings (document texts).
    :param chunk_size: Maximum size of each chunk.
    :param chunk_overlap: Overlap size between consecutive chunks.
    :return: List of text chunks.
    """
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be at least 0 and smaller than chunk_size.")
    chunks = []
    for doc in documents:
        start = 0
        while start < len(doc):
            end = start + chunk_size
            chunks.append(doc[start:end])
            start = end - chunk_overlap if end < len(doc) else end
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Answer a question from local documents with QuantumRAG.")
    parser.add_argument("--query", default="How many carats were recovered in 2022?")
    parser.add_argument("--data-dir", default=None, help="Folder of .txt and .pdf files (default: ./data)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=None, help="Limit Grover iterations (default: optimal)")
    parser.add_argument("--embedding-backend", choices=["auto", "sentence-transformers", "tfidf"], default="auto")
    parser.add_argument("--llm-backend", choices=["auto", "transformers", "extractive"], default="auto")
    args = parser.parse_args()

    docs = load_documents_from_folder(args.data_dir)
    doc_chunks = split_documents_into_chunks(docs)

    pipeline = QuantumRAGPipeline(
        documents=doc_chunks,
        embedding_backend=args.embedding_backend,
        llm_backend=args.llm_backend,
    )
    pipeline.build_embeddings()
    pipeline.init_quantum_retriever(max_grover_iterations=args.max_iterations)
    answer = pipeline.run(args.query, top_k=args.top_k)

    print(f"Backends: embeddings={pipeline.embedding_manager.backend}, llm={pipeline.llm.backend}")
    print(f"Quantum retrieval: {pipeline.q_retriever.last_run}")
    print("=== FINAL RAG ANSWER ===")
    print(answer)


if __name__ == "__main__":
    main()
