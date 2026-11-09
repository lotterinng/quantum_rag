# QuantumRAG Project

## Overview
QuantumRAG (Quantum Retrieval-Augmented Generation) is a hybrid pipeline that combines quantum computing techniques, document retrieval, and large language models (LLMs) to answer user queries based on a corpus of documents. The pipeline uses Grover-style amplitude amplification, run on the Qiskit Aer simulator, to select relevant documents and generates answers using a pre-trained language model.

> **What the quantum step does:** similarity scores are computed classically, loaded into a quantum state, and the top documents are amplified. Because the oracle is built from those classical scores, this demonstrates the technique and gives no speedup over sorting the scores.

This project is designed to demonstrate the integration of quantum retrieval methods with state-of-the-art machine learning models for document-based question answering.

---

## Features
1. **Quantum Retrieval**:
   - Encodes document embeddings into quantum states.
   - Applies Grover-like iterations for efficient document retrieval.
2. **Retrieval-Augmented Generation**:
   - Retrieves relevant documents based on user queries.
   - Generates human-like responses using a language model.
3. **Support for PDFs and Text Files**:
   - Processes and extracts text from `.txt` and `.pdf` files.
4. **Customizable Models**:
   - Uses transformer-based embedding models (e.g., `BAAI/bge-large-en-v1.5`).
   - Integrates with Llama models (e.g., `meta-llama/Llama-3.1-8B-Instruct`).

---

## Prerequisites

1. **System Requirements**:
   - Python 3.10+
   - Git LFS, to download the PDF in `data/` (`git lfs pull`).
   - GPU for faster inference (Optional).

2. **Python Dependencies**:
   ```bash
   pip install -r requirements.txt          # Quantum retrieval with offline TF-IDF and extractive answers
   pip install -r requirements-models.txt   # Adds the BGE embedding model and Llama
   ```

3. **Additional Tools**:
   - Install NVIDIA drivers if using GPU.
   - Access to Hugging Face for restricted models.

---

## Project Structure

```
quantum_rag/
├── main.py                # Main script to run the pipeline
├── quantum_rag_pipeline.py # Core pipeline implementation
├── quantum_retriever.py    # Quantum-based document retrieval logic
├── embeddings.py           # Embedding manager for documents and queries
├── classical_llm.py        # Large language model wrapper
├── data/                   # Folder containing input documents
└── README.md               # Project readme file
```

---

## Usage

### 1. Prepare Your Documents
Place your `.txt` and `.pdf` files in the `data/` directory.

### 2. Run the Project
```bash
python main.py
python main.py --query "What was the 2022 production?" --top-k 3
```

By default the app uses the BGE and Llama models when they can load, and otherwise prints a warning and falls back to TF-IDF and sentence extraction. To require a backend, pass `--embedding-backend sentence-transformers` or `--llm-backend transformers`. To force offline mode, pass `--embedding-backend tfidf --llm-backend extractive`.

### 3. Run the Tests
```bash
pip install pytest
python -m pytest
```

---

## Key Files

### 1. `main.py`
- Loads documents from the `data/` folder.
- Initializes the QuantumRAG pipeline.
- Processes user queries and prints generated responses.

### 2. `quantum_rag_pipeline.py`
- Orchestrates the pipeline:
  - Embedding generation.
  - Quantum-based document retrieval.
  - Language model integration.

### 3. `quantum_retriever.py`
- Implements Grover-style retrieval in Qiskit:
  - Encodes document embeddings as quantum states.
  - Applies Grover-like search for document selection.

### 4. `embeddings.py`
- Manages embedding generation for documents and queries using `sentence-transformers`, with TF-IDF as the offline fallback.

### 5. `classical_llm.py`
- Wraps the LLM for text generation.
- Customizable for different LLM models, with sentence extraction as the offline fallback.

---

## Example Output
After running `main.py`, you will see output similar to this:
```
Split 1 documents into 3193 chunks.
Backends: embeddings=tfidf, llm=extractive
Quantum retrieval: {'qubits': 12, 'iterations': 1, 'marked_prior': 0.1498, 'marked_final': 0.8633}
=== FINAL RAG ANSWER ===
[extractive, no LLM] In the year to 31 December 2022, the company recovered 24.1 million carats (2021: 22.3 million carats), a 8% increase.
```

---

## Troubleshooting

### Issue: "GPU not detected"
- Ensure that NVIDIA drivers are installed.
- Verify GPU availability:
  ```bash
  nvidia-smi
  ```

### Issue: "Unauthorized for Hugging Face Model"
- Log in to Hugging Face:
  ```python
  from huggingface_hub import notebook_login
  notebook_login()
  ```

### Issue: "Quantum simulator not available"
- Install Qiskit's Aer package:
  ```bash
  pip install qiskit-aer
  ```

---

## Future Enhancements
- Incorporate more efficient quantum algorithms for retrieval.
- Support additional document formats (e.g., `.docx`).
- Enable deployment on cloud platforms like Azure and Google Colab.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgments
- Hugging Face for pre-trained models.
- Qiskit for quantum computing frameworks.
- Sentence Transformers for embedding generation.

