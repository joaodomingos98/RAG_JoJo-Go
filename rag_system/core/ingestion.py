import os
import glob
import fitz  # PyMuPDF
import docx
import hashlib
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm
from config import settings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize OCR Engine once (It's very fast to load)
# detach_model=True saves memory by unloading model after use if needed,
# but for batch processing, keep it False for speed.
_OCR_ENGINE = RapidOCR()

def extract_text_from_image(image_obj):
    """
    Uses RapidOCR to extract text from a PIL Image object.
    """
    try:
        # RapidOCR expects a file path or numpy array.
        # We convert PIL image to bytes/numpy to avoid saving temp files.
        import numpy as np
        img_array = np.array(image_obj)

        # Run OCR
        result, elapse = _OCR_ENGINE(img_array)

        if not result:
            return ""

        # Result format is a list of [box, text, confidence]
        # We just want the text combined
        extracted_text = "\n".join([line[1] for line in result])
        return extracted_text

    except Exception as e:
        print(f"⚠️ OCR Error: {e}")
        return ""


# ========================================
# FILE PROCESSORS
# ========================================

def process_pdf(file_path: str) -> str:
    """
    Smart PDF extraction:
    1. Tries standard text extraction (fast).
    2. If page is empty (scanned), renders image and runs OCR.
    """
    text_content = []
    doc = fitz.open(file_path)

    for page_num, page in enumerate(doc):
        # 1. Try fast standard text extraction
        text = page.get_text()

        # 2. Check if the page is likely a scan (very little text)
        if len(text.strip()) < 50:
            # print(f"   Note: Page {page_num+1} seems scanned. Running OCR...")

            # Render page as image (zoom=2 for better quality)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            ocr_text = extract_text_from_image(img)
            text += "\n" + ocr_text

        text_content.append(text)

    return "\n".join(text_content)


def process_docx(file_path: str) -> str:
    """Extracts text from Word Documents."""
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"❌ Error reading Word doc: {e}")
        return ""

def process_image(file_path: str) -> str:
    """Extracts text from direct image files (JPG, PNG)."""
    try:
        img = Image.open(file_path).convert("RGB")
        return extract_text_from_image(img)
    except Exception as e:
        print(f"❌ Error reading Image: {e}")
        return ""


def process_txt(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return ""


# ========================================
# HELPER: DEDUPLICATION
# ========================================

def generate_chunk_hash(content: str) -> str:
    """
    Generates a unique hash for a text string.
    Used to detect duplicate chunks across different files.
    """
    # Normalize text (remove extra whitespace) to catch near-duplicates
    cleaned_text = " ".join(content.split())
    return hashlib.md5(cleaned_text.encode('utf-8')).hexdigest()


# ========================================
# MAIN LOADER
# ========================================

def load_documents(directory_path: str = settings.DATA_DIR) -> List[Dict[str, str]]:
    raw_documents = []
    base_path = Path(directory_path)

    # 1. First, find all valid files to verify count for progress bar
    # (We scan first so tqdm knows the total)
    print(f"\n📂 Scanning {base_path} for files...")

    valid_files = []
    handlers = {
        ".pdf": process_pdf,
        ".docx": process_docx,
        ".txt": process_txt,
        ".jpg": process_image,
        ".png": process_image,
        ".jpeg": process_image
    }

    # Recursive search
    for file_path in base_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in handlers:
            valid_files.append(file_path)

    if not valid_files:
        print("⚠️  No supported files found.")
        return []

    print(f"✅ Found {len(valid_files)} files to process.")

    # 2. Process files with Progress Bar
    # tqdm(valid_files) creates the visual loading bar
    for file_path in tqdm(valid_files, desc="📄 Processing Files", unit="file"):
        try:
            # Determine Category
            parent_folder = file_path.parent
            if parent_folder.resolve() == base_path.resolve():
                category = "general"
            else:
                category = parent_folder.name

                # Process Content
            handler = handlers[file_path.suffix.lower()]
            content = handler(str(file_path))

            if content and len(content.strip()) > 0:
                raw_documents.append({
                    "id": file_path.name,
                    "title": file_path.stem.replace("_", " ").title(),
                    "content": content,
                    "category": category,
                    "source_doc": file_path.name
                })
        except Exception as e:
            # Use tqdm.write so it doesn't break the progress bar visual
            tqdm.write(f"❌ Error processing {file_path.name}: {e}")

    return raw_documents


# ========================================
# CHUNKING & DEDUPLICATION
# ========================================

def split_documents(documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    if not documents:
        return []

    print("\n🧠 Initializing Semantic Chunker (Loading Embedding Model)...")

    # 1. Point LangChain to your local embedding model
    # (Falls back to downloading if the local path doesn't exist yet)
    model_path = str(
        settings.LOCAL_EMBEDDING_PATH) if settings.LOCAL_EMBEDDING_PATH.exists() else settings.EMBEDDING_MODEL_NAME
    hf_embeddings = HuggingFaceEmbeddings(model_name=model_path)

    # 2. Configure the Semantic Chunker
    # "percentile" at 90 means: Split the chunk when the difference in meaning
    # between two sentences is in the top 10% of all differences in the document.
    text_splitter = SemanticChunker(
        hf_embeddings,
        breakpoint_threshold_type= settings.CHUNKING_TYPE,
        breakpoint_threshold_amount=settings.CHUNKING_THRESHOLD
    )

    all_chunks = []
    seen_hashes = set()
    duplicates_removed = 0

    print(f"\n✂️  Semantic Chunking & Deduplicating {len(documents)} documents...")
    print("⏳ Note: Semantic chunking is slower than basic chunking because it runs AI inference on every sentence.")

    for doc in tqdm(documents, desc="🧩 Splitting Chunks", unit="doc"):
        # The text_splitter creates documents based on semantic meaning
        try:
            # SemanticChunker expects create_documents rather than split_text
            langchain_docs = text_splitter.create_documents([doc["content"]])
            chunks = [c.page_content for c in langchain_docs]
        except Exception as e:
            tqdm.write(f"⚠️ Error semantically chunking {doc['id']}: {e}. Skipping.")
            continue

        for i, chunk_text in enumerate(chunks):
            # Generate Hash
            chunk_hash = generate_chunk_hash(chunk_text)

            # Check for duplicate
            if chunk_hash in seen_hashes:
                duplicates_removed += 1
                continue

                # If new, add to set and list
            seen_hashes.add(chunk_hash)

            all_chunks.append({
                "id": f"{doc['id']}_chunk_{i}",
                "title": doc.get("title", "Untitled"),
                "content": chunk_text,
                "category": doc.get("category", "General"),
                "source_doc": doc.get("source_doc", "unknown")
            })

    print(f"✅ Generated {len(all_chunks)} unique semantic chunks.")
    if duplicates_removed > 0:
        print(f"🗑️  Removed {duplicates_removed} duplicate chunks.")

    return all_chunks


def load_and_chunk_documents():
    raw_docs = load_documents(settings.DATA_DIR)
    return split_documents(raw_docs)