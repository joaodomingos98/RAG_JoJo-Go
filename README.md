# 🚀 TechCorp PolicyCopilot (Local RAG System)

A private, secure, and local Retrieval-Augmented Generation (RAG) system designed to answer questions about internal company documents using AI. 

This project runs entirely on your local machine using **Ollama** (for LLMs) and **ChromaDB** (for vector storage), ensuring no data leaves the company network.

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

**TechCorp PolicyCopilot** allows employees to ask questions in natural language about company policies (PDFs/TXTs) and receive accurate, sourced answers.

**Key Features:**
- **Local & Private:** Uses local LLMs (Llama 3, Mistral) via Ollama.
- **Persistent Memory:** Vector embeddings are stored on disk (ChromaDB) so you don't have to reload documents every run.
- **Source Citations:** Every answer cites the specific document chunk used.
- **Modular Design:** Easy to swap out the LLM, Embedding Model, or Vector DB.

---

## 🏗 Architecture

The system is split into two distinct pipelines to ensure performance:

1.  **Ingestion Pipeline (Offline):**
    * Loads documents (`.txt`, `.pdf`) from the `data/` folder.
    * Splits text into manageable chunks.
    * Generates vector embeddings.
    * Stores them in a persistent ChromaDB database on disk.

2.  **Inference Pipeline (Online):**
    * Takes user question.
    * Searches the Vector DB for relevant chunks (Semantic Search).
    * Augments the prompt with retrieved context.
    * Generates a response using the Local LLM.

---

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.10+**
2.  **[Ollama](https://ollama.com/)** (Running in the background)
    * Install Ollama from the official website.
    * Pull the model we are using:
        ```bash
        ollama pull llama3
        ```

---

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-org/techcorp-policy-copilot.git](https://github.com/your-org/techcorp-policy-copilot.git)
    cd techcorp-policy-copilot
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    
    # Windows
    venv\Scripts\activate
    
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage Guide

### 1. Data Ingestion (Setup)
*Run this only when you add new documents.*

1.  Place your PDF or TXT files in the `data/` folder.
2.  Run the ingestion script:
    ```bash
    python ingest_data.py
    ```
    *This will create a `chroma_db_data/` folder. Do not delete this folder unless you want to reset the database.*

### 2. Running the Chat App
*Run this whenever you want to chat.*

1.  Start the application:
    ```bash
    python main_app.py
    ```
2.  Ask questions like:
    > "What is the remote work policy?"
    > "How much can I spend on travel meals?"

---

## 🔧 Configuration

All settings are centralized in `rag_system/config/settings.py`. You can modify:

| Setting | Description | Default |
| :--- | :--- | :--- |
| `LLM_MODEL_NAME` | The Ollama model to use | `llama3` |
| `EMBEDDING_MODEL` | The HuggingFace model for vectors | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Size of text chunks (in chars) | `1000` |
| `DISTANCE_METRIC` | Similarity math (`cosine`, `l2`, `ip`) | `cosine` |

**Note:** If you change `CHUNK_SIZE`, `EMBEDDING_MODEL`, or `DISTANCE_METRIC`, you must delete the `chroma_db_data` folder and re-run `python ingest_data.py`.

---

## 📂 Project Structure

```text
rag_system/
│
├── config/
│   └── settings.py       # Central configuration (Models, Paths, Constants)
│
├── core/
│   ├── ingestion.py      # Logic for loading & chunking docs
│   ├── database.py       # ChromaDB connection & storage logic
│   ├── retrieval.py      # Embedding generation logic
│   └── generation.py     # Ollama LLM interaction logic
│
├── data/                 # DROP YOUR DOCUMENTS HERE
├── chroma_db_data/       # (Generated) Persistent Vector Database
│
├── ingest_data.py        # Script: Run to process documents
├── main_app.py           # Script: Run to start chatting
└── requirements.txt      # Python dependencies
