import os
import glob
import fitz  # PyMuPDF
import docx
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rapidocr_onnxruntime import RapidOCR
from config import settings

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
# MAIN LOADER (Recursive & Categorized)
# ========================================

def load_documents(directory_path: str = settings.DATA_DIR) -> List[Dict[str, str]]:
    """
    Recursively loads documents from the data directory.
    Assigns 'category' based on the folder name holding the file.
    """
    raw_documents = []
    base_path = Path(directory_path)

    # Map extensions to their processor functions
    # Make sure these functions (process_pdf, etc.) are defined above in your file
    handlers = {
        ".pdf": process_pdf,
        ".docx": process_docx,
        ".txt": process_txt,
        ".jpg": process_image,
        ".png": process_image,
        ".jpeg": process_image
    }

    print(f"\n📂 Recursively scanning {base_path} for files...")

    # rglob('*') finds ALL files recursively
    for file_path in base_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in handlers:

            try:
                # 1. Determine Category based on folder structure
                # If file is in "data/HR", category is "HR"
                # If file is directly in "data/", category is "general"
                parent_folder = file_path.parent

                if parent_folder.resolve() == base_path.resolve():
                    category = "general"
                else:
                    category = parent_folder.name  # e.g., "HR", "Finance"

                # 2. Process the file
                handler = handlers[file_path.suffix.lower()]
                content = handler(str(file_path))

                if content and len(content.strip()) > 0:
                    filename = file_path.name
                    print(f"   Processing: {category}/{filename}")

                    raw_documents.append({
                        "id": filename,
                        "title": file_path.stem.replace("_", " ").title(),  # Clean title
                        "content": content,
                        "category": category,  # <--- Dynamic Category
                        "source_doc": filename
                    })
                else:
                    print(f"   ⚠️  Skipping empty file: {file_path.name}")

            except Exception as e:
                print(f"   ❌ Error processing {file_path.name}: {e}")

    print(f"✅ Loaded {len(raw_documents)} documents from {len(set(d['category'] for d in raw_documents))} categories.")
    return raw_documents

# ========================================
# CHUNKING (Standard)
# ========================================

def split_documents(documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    # Standard chunking logic (Same as before)
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
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

    print(f"✅ Generated {len(all_chunks)} chunks from {len(documents)} files.")
    return all_chunks


def load_and_chunk_documents():
    raw_docs = load_documents(settings.DATA_DIR)
    return split_documents(raw_docs)