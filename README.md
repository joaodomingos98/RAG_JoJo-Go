# ✨ AI MAGIC - Local RAG System for PolicyCopiloting

A private, secure, and fully local Retrieval-Augmented Generation (RAG) system designed to answer questions about internal company documents using AI.

This project runs **entirely offline** on your local machine using **llama-cpp-python** (for embedded LLMs), **ChromaDB** (for vector storage), and **RapidOCR** (for document vision), ensuring no data leaves the company network.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
  - [1. Data Ingestion](#1-data-ingestion-setup)
  - [2. Running the Chat App](#2-running-the-chat-app)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)

---

## 🔎 Project Overview

**PolicyCopilot** allows employees to ask questions in natural language about company policies (PDFs, DOCX, TXT, Images) and receive accurate, sourced answers.

**Key Features:**
- **Truly Local:** No external API dependencies. The LLM (Llama 3.2 3B) runs embedded inside the Python process.
- **Smart Ingestion:** Recursively scans folders, assigns categories, and uses **RapidOCR** to read scanned PDFs and images.
- **Deduplication:** Automatically detects and removes duplicate content using MD5 hashing.
- **Hardware Optimized:** Auto-detects GPU (CUDA/Metal) and CPU threads for maximum performance.
- **Performance Metrics:** Measures and displays generation time in milliseconds.

---

## 🏗 Architecture

The system is split into two distinct pipelines to ensure performance:

1.  **Ingestion Pipeline (Offline):**
    * **Recursive Scan:** Reads files from `data/` and its subfolders (folder name = category).
    * **Universal Loading:** Processes `.txt`, `.pdf` (text & scanned), `.docx`, `.jpg`, `.png`.
    * **Deduplication:** Hashes content to prevent storing duplicate chunks.
    * **Storage:** Saves vector embeddings to a persistent ChromaDB on disk.

2.  **Inference Pipeline (Online):**
    * **Semantic Search:** Queries ChromaDB using Cosine Similarity.
    * **Context Augmentation:** Feeds relevant chunks to the embedded LLM.
    * **Local Generation:** Uses `llama-cpp-python` (GGUF format) to generate answers locally.

---

## ⚙️ Prerequisites

1.  **Python 3.10+**
2.  **C++ Build Tools** (Required for hardware acceleration):
    * **Windows:** Visual Studio Community (Desktop development with C++).
    * **Mac:** Xcode Command Line Tools (`xcode-select --install`).
    * **Linux:** `build-essential`.

*(Note: You do **NOT** need to install the Ollama desktop app anymore. The model runs directly in the script.)*

---

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-org/techcorp-policy-copilot.git](https://github.com/your-org/techcorp-policy-copilot.git)
    cd techcorp-policy-copilot
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    # Windows: venv\Scripts\activate
    # Mac/Linux: source venv/bin/activate
    ```

3.  **Install Dependencies:**
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

4.  **Download Models:**
    Run this script once to fetch the Embedding model (SentenceTransformers) and the LLM (Llama 3.2 GGUF).
    ```bash
    python download_models.py
    ```

---

## 🚀 Usage Guide

### 1. Data Ingestion (Setup)
*Run this when you add new documents.*

1.  Place your files in the `data/` folder. You can create subfolders (e.g., `data/HR`, `data/IT`) to automatically categorize documents.
2.  Run the ingestion script:
    ```bash
    python ingest_data.py
    ```
    * *Features: Recursive scan, OCR for images/scans, Progress bars, Deduplication.*
    * *Creates a `chroma_db_data/` folder.*

### 2. Running the Chat App
*Run this to start chatting.*

1.  Start the application:
    ```bash
    python main_app.py
    ```
2.  The app will load the Llama 3.2 model into RAM/VRAM.
3.  Ask questions like:
    > "What is the remote work policy?"
    > "How do I claim travel expenses?"

---

## 🔧 Configuration

All settings are in `rag_system/config/settings.py`.

| Setting | Description | Default |
| :--- | :--- | :--- |
| `LLM_REPO_ID` | HuggingFace Repo for GGUF model | `bartowski/Llama-3.2-3B...` |
| `EMBEDDING_MODEL` | Vector Model | `all-MiniLM-L6-v2` |
| `DISTANCE_METRIC` | Similarity math (`cosine`, `l2`) | `cosine` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `CONTEXT_WINDOW` | LLM Memory (Tokens) | `4096` |
| `DEVICE` | OCR Processing Device | `cuda` (if available) |

**Important:** If you change `CHUNK_SIZE`, `DISTANCE_METRIC`, or `EMBEDDING_MODEL`, you must delete the `chroma_db_data` folder and re-run ingestion.

---

## 📂 Project Structure

```text
rag_system/
│
├── config/
│   └── settings.py       # Central configuration
│
├── core/
│   ├── ingestion.py      # Recursive loader, OCR, Deduplication, Chunking
│   ├── database.py       # Persistent Vector DB (Chroma)
│   ├── retrieval.py      # Embedding generation
│   └── generation.py     # Llama-CPP (GGUF) Engine
│
├── data/                 # Documents (PDF, DOCX, IMG, TXT)
├── models/               # Local Model Files (Downloaded via script)
├── chroma_db_data/       # Vector Database Storage
│
├── download_models.py    # Setup script to fetch AI models
├── ingest_data.py        # ETL Pipeline Script
├── main_app.py           # Chat Interface Script
└── requirements.txt      # Dependencies
