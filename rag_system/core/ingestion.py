# core/ingestion.py
import os
import glob
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings


# ========================================
# DATA LOADING
# ========================================

def load_text_files(directory_path: str = settings.DATA_DIR) -> List[Dict[str, str]]:
    """
    Scans the configured DATA_DIR for .txt files.
    """
    print(f"\n📂 Scanning directory: {directory_path}")

    # Use str() because pathlib object needs conversion for glob sometimes
    txt_files = glob.glob(os.path.join(str(directory_path), "*.txt"))

    if not txt_files:
        print(f"⚠️  No .txt files found in {directory_path}")
        return []

    print(f"📄 Found {len(txt_files)} text files.")

    raw_documents = []
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()

            doc_id = os.path.basename(txt_file).replace(".txt", "")
            raw_documents.append({
                "id": doc_id,
                "title": doc_id.replace("_", " ").title(),
                "content": content,
                "category": "uploaded_text",
                "source_doc": os.path.basename(txt_file)
            })
        except Exception as e:
            print(f"❌ Error reading {txt_file}: {e}")

    return raw_documents


# ========================================
# CHUNKING LOGIC
# ========================================

def split_documents(documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    print(f"\n✂️  Splitting {len(documents)} documents...")

    # Use settings for chunk size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,  # <--- FROM SETTINGS
        chunk_overlap=settings.CHUNK_OVERLAP,  # <--- FROM SETTINGS
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "id": f"{doc['id']}_chunk_{i}",
                "title": doc.get("title", "Untitled"),
                "content": chunk_text,
                "category": doc.get("category", "General"),
                "source_doc": doc.get("source_doc", "unknown")
            })

    print(f"✅ Generated {len(all_chunks)} chunks.")
    return all_chunks


def load_and_chunk_documents():
    """Main orchestrator."""
    raw_docs = load_text_files(settings.DATA_DIR)
    if not raw_docs:
        print("⚠️ No documents found. Please add .txt files to the 'data' folder.")
        return []

    return split_documents(raw_docs)