# JoJo-GO - Local RAG System

A private, secure, and fully local Retrieval-Augmented Generation (RAG) system designed to answer questions about internal company documents using AI.

This project runs **entirely offline** on your local machine using **llama-cpp-python** (for embedded LLMs), **ChromaDB** (for vector storage), and **RapidOCR** (for document vision), ensuring no data leaves the company network.

---

## Table of Contents

* [Project Overview](https://www.google.com/search?q=%23-project-overview)
* [Architecture](https://www.google.com/search?q=%23-architecture)
* [Prerequisites](https://www.google.com/search?q=%23-prerequisites)
* [Installation](https://www.google.com/search?q=%23-installation)
* [Usage Guide](https://www.google.com/search?q=%23-usage-guide)
* [1. Data Ingestion](https://www.google.com/search?q=%231-data-ingestion-setup)
* [2. Running the Chat App](https://www.google.com/search?q=%232-running-the-chat-app)


* [Configuration](https://www.google.com/search?q=%23-configuration)
* [Project Structure](https://www.google.com/search?q=%23-project-structure)

---

## Project Overview

**JoJo-GO** allows users to ask questions in natural language about internal documents (PDFs, DOCX, TXT, Images) and receive accurate, sourced answers via a modern web chat interface.

**Key Features:**

* **Web UI:** A beautiful, interactive chat interface built with Streamlit, complete with real-time streaming and source document citations.
* **Truly Local:** No external API dependencies. The LLM (**LiquidAI LFM2 1.2B RAG**) runs embedded inside the Python process.
* **Semantic Chunking:** Uses AI to chunk documents intelligently based on shifts in meaning and context, rather than arbitrary character counts.
* **Smart Ingestion:** Recursively scans folders and uses **RapidOCR** to read scanned PDFs and images.
* **Deduplication:** Automatically detects and removes duplicate content using MD5 hashing.

---

## Architecture

The system is split into two distinct pipelines to ensure performance. **Both pipelines run 100% locally on your machine.**

1. **Ingestion Pipeline (Background Setup):**
* **Recursive Scan:** Reads files from `data/` and its subfolders.
* **Universal Loading:** Processes `.txt`, `.pdf` (text & scanned), `.docx`, `.jpg`, `.png`.
* **Semantic Chunking:** LangChain's `SemanticChunker` groups sentences by embedding distance.
* **Deduplication:** Hashes content to prevent storing duplicate chunks.
* **Storage:** Saves vector embeddings and metadata to a persistent ChromaDB on disk.


2. **Inference Pipeline (Real-Time Chat):**
* **Query Embedding:** Converts the user's question into a mathematical vector.
* **Semantic Search:** Queries ChromaDB to find the most relevant document chunks based on mathematical similarity.
* **Context Augmentation:** Feeds relevant chunks to the embedded LLM.
* **Local Generation:** Uses `llama-cpp-python` to generate answers locally with automatic KV-cache flushing to prevent context overflow.



---

## Prerequisites

1. **Python 3.10+**
2. **C++ Build Tools** (Required for hardware acceleration):
* **Windows:** Visual Studio Community (Desktop development with C++).
* **Mac:** Xcode Command Line Tools (`xcode-select --install`).
* **Linux:** `build-essential`.



*(Note: You do **NOT** need to install the Ollama desktop app. The model runs directly in the script.)*

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-org/jojo-go.git
cd jojo-go

```


2. **Create a Virtual Environment:**
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

```


3. **Install Dependencies:**
* **Standard Install (CPU Only):**
```bash
pip install -r requirements.txt

```


* **GPU Acceleration (Recommended for Speed):**
* **NVIDIA (Windows/Linux):**
```bash
# PowerShell
$env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install llama-cpp-python --force-reinstall --no-cache-dir
pip install -r requirements.txt

```


* **Mac (M1/M2/M3):**
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
pip install -r requirements.txt

```

---

## Usage Guide

### 1. Data Ingestion (Setup)

*Run this when you add new documents or change chunking settings.*

1. Place your files in the `data/` folder. Use subfolders (e.g., `data/HR`, `data/IT`) to automatically categorize documents.
2. Run the ingestion script:
```bash
python main_ingest.py

```


* *Features: Recursive scan, Semantic AI chunking, OCR for images/scans, Deduplication.*



### 2. Running the Chat App

*Run this to start the interactive web UI.*

1. Start the Streamlit application:
```bash
streamlit run ui_app.py

```


2. The app will open automatically in your web browser (usually at `http://localhost:8501`).
3. Ask questions like:
> "What happened in the last meeting?"
> "Who is the board of directors?"



*(Note: You can still run the terminal-only version using `python main_app.py` if preferred).*

---

## Configuration

All system settings are centralized in `config/settings.py`.

| Setting | Description | Default |
| --- | --- | --- |
| `LLM_REPO_ID` | HuggingFace Repo for GGUF model | `LiquidAI/LFM2-1.2B-RAG-GGUF` |
| `EMBEDDING_MODEL_NAME` | Vector Model | `all-MiniLM-L6-v2` |
| `DISTANCE_METRIC` | Similarity math (`cosine`, `l2`, `ip`) | `cosine` |
| `CHUNKING_TYPE` | Semantic break method | `percentile` |
| `CHUNKING_THRESHOLD` | Threshold for semantic splits | `90` |
| `CONTEXT_WINDOW` | LLM Memory (Tokens) | `8192` |

**Important:** If you change `CHUNKING_TYPE`, `CHUNKING_THRESHOLD`, `DISTANCE_METRIC`, or `EMBEDDING_MODEL_NAME`, you must delete the `chroma_db_data` folder and re-run the ingestion script.

---

## Project Structure

```text
rag_system/
│
├── config/
│   └── settings.py       # Central configuration
│
├── core/
│   ├── ingestion.py      # Semantic chunking, OCR, Deduplication
│   ├── database.py       # Persistent Vector DB (Chroma)
│   ├── retrieval.py      # Embedding generation
│   └── generation.py     # Llama-CPP Engine
│
├── data/                 # Documents (PDF, DOCX, IMG, TXT)
├── models/               # Local Model Files
├── chroma_db_data/       # Vector Database Storage
│
├── download_models.py    # Setup script to fetch AI models
├── main_ingest.py        # ETL Pipeline Script
├── main_app.py           # Terminal Interface & Core RAG Pipeline
├── ui_app.py             # Streamlit Web Interface
└── requirements.txt      # Dependencies

```
