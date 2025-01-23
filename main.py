import os
from quantum_rag_pipeline import QuantumRAGPipeline
from PyPDF2 import PdfReader

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
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_contents = f.read()
                docs.append(file_contents)

        elif filename.lower().endswith(".pdf"):
            reader = PdfReader(file_path)
            pdf_text = []
            for page in reader.pages:
                pdf_text.append(page.extract_text() or "")
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
    chunks = []
    for doc in documents:
        start = 0
        while start < len(doc):
            end = start + chunk_size
            chunks.append(doc[start:end])
            start = end - chunk_overlap if end < len(doc) else end
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    # 1. Load documents from the local 'data' folder
    folder_path = "data"
    docs = load_documents_from_folder()

    # 2. Split documents into smaller chunks
    print("Splitting documents into smaller chunks...")
    doc_chunks = split_documents_into_chunks(docs)

    # 3. Create and configure the RAG pipeline
    print("Creating QuantumRAGPipeline...")
    pipeline = QuantumRAGPipeline(
        documents=doc_chunks,
        embedding_model_name="BAAI/bge-large-en-v1.5",  # Embedding model
        llm_model_name="meta-llama/Llama-3.1-8B",  # LLM for generation
    )

    # 4. Build document embeddings
    print("Building embeddings...")
    pipeline.build_embeddings()

    # 5. Initialize quantum retriever
    print("Initializing quantum retriever...")
    pipeline.init_quantum_retriever(max_grover_iterations=3)

    # 6. Run a sample query
    print("Running query...")
    query = "How many carats were recovered in 2022."
    answer = pipeline.run(query, top_k=1)
    print("Query complete. Answer:", answer)

    print("=== FINAL RAG ANSWER ===")
    print(answer)
