import os
from quantum_rag_pipeline import QuantumRAGPipeline
from PyPDF2 import PdfReader

import os
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
if __name__ == "__main__":
    # 1. Load documents from the local 'data' folder
    folder_path = "data"
    docs = load_documents_from_folder()

    # 2. Create and configure the RAG pipeline
    #    Replace 'BAAI/bge-large-en-v1.5' or 'meta-llama/Llama-3.1-8B'
    #    with whichever models you want to use
    pipeline = QuantumRAGPipeline(
        documents=docs,
        embedding_model_name="BAAI/bge-large-en-v1.5",  # Llama embedding if desired
        llm_model_name="meta-llama/Llama-3.1-8B",    # Llama 3 for generation
    )

    # 3. Build document embeddings
    pipeline.build_embeddings()

    # 4. Initialize quantum retriever
    pipeline.init_quantum_retriever(max_grover_iterations=3)

    # 5. Run a sample query
    query = "How many carats were recovered in 2022."
    answer = pipeline.run(query, top_k=1)

    print("=== FINAL RAG ANSWER ===")
    print(answer)
