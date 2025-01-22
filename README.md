# QuantumRAG Project

## Overview
QuantumRAG (Quantum Retrieval-Augmented Generation) is a hybrid pipeline that combines quantum computing techniques, document retrieval, and large language models (LLMs) to answer user queries based on a corpus of documents. The pipeline uses quantum-inspired retrieval (Grover-like search) to select relevant documents and generates answers using a pre-trained language model.

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
   - Integrates with Llama models (e.g., `meta-llama/Meta-Llama-3-8B`).

---

## Prerequisites

1. **System Requirements**:
   - Python 3.8+
   - GPU for faster inference (Optional).

2. **Python Dependencies**:
   Install the required Python libraries:
   ```bash
   pip install transformers sentence-transformers qiskit qiskit-aer PyPDF2 numpy
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
Execute the `main.py` script to process the documents and answer a sample query:
```bash
python main.py
```

### 3. Modify Queries
In `main.py`, change the `query` variable to test different questions.

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
- Implements quantum-inspired retrieval:
  - Encodes document embeddings as quantum states.
  - Applies Grover-like search for document selection.

### 4. `embeddings.py`
- Manages embedding generation for documents and queries using models like `sentence-transformers` and `llama`.

### 5. `classical_llm.py`
- Wraps the LLM for text generation.
- Customizable for different LLM models.

---

## Example Output
After running `main.py`, you will see output similar to this:
```
Loading checkpoint shards: 100%
Padded amplitudes from 13 to 16 to satisfy 2^n requirement.
Final amplitude array length: 16 (2^4). Norm=1.0
=== FINAL RAG ANSWER ===
Grover's algorithm provides a quadratic speedup for search problems by...
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

