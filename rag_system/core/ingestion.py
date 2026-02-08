import os
import glob
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ========================================
# DATA LOADING
# ========================================

def load_text_files_from_directory(directory_path: str) -> List[Dict[str, str]]:
    """
    Scans a directory for .txt files and loads their content.
    """
    print(f"\n📂 Scanning directory for TXT files: {directory_path}")

    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"❌ Error: Directory '{directory_path}' does not exist.")
        return []

    # Find all .txt files
    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))

    if not txt_files:
        print("⚠️  No .txt files found in this directory.")
        return []

    print(f"📄 Found {len(txt_files)} text files.")

    raw_documents = []

    for txt_file in txt_files:
        try:
            print(f"   Processing: {os.path.basename(txt_file)}")

            # Read the text file
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Create document object
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


def get_demo_documents() -> List[Dict[str, str]]:
    """
    Returns hardcoded policy documents for the demo.
    """
    return [
        {
            "id": "policy_001",
            "title": "Home Office Equipment Reimbursement",
            "content": "Employees working from home may claim up to $500 per year for office equipment including desks, chairs, monitors, and computer accessories. Receipts must be submitted within 30 days of purchase. This policy applies to full-time remote workers only. The equipment must be used primarily for work purposes and should be ergonomic and suitable for a professional home office environment.",
            "category": "reimbursement"
        },
        {
            "id": "policy_002",
            "title": "Travel Expense Guidelines",
            "content": "Business travel expenses are reimbursable when pre-approved by your manager. Meals are covered up to $50 per day, hotel stays up to $200 per night. All receipts must be submitted within 14 days of return. International travel requires additional approval from the department head. Travel insurance is mandatory for all business trips exceeding 7 days.",
            "category": "travel"
        },
        {
            "id": "policy_003",
            "title": "Remote Work Furniture Policy",
            "content": "Remote employees may purchase ergonomic furniture for their home office setup. This includes standing desks, ergonomic chairs, and monitor arms. Maximum reimbursement is $300 per item with manager approval required. All furniture must meet ergonomic standards and be purchased from approved vendors. Receipts must be submitted within 45 days of purchase.",
            "category": "reimbursement"
        },
        {
            "id": "policy_004",
            "title": "Equipment and Supplies Reimbursement",
            "content": "Work-related equipment and supplies purchased for home office use are eligible for reimbursement. This covers laptops, monitors, keyboards, mice, and other computer peripherals. Submit expense reports with receipts for approval. Equipment must be used for work purposes and should be compatible with company systems. Annual limit is $1000 per employee.",
            "category": "reimbursement"
        },
        {
            "id": "policy_005",
            "title": "Vacation and PTO Policy",
            "content": "Full-time employees accrue 15 days of paid time off per year. Vacation requests must be submitted at least 2 weeks in advance. Unused PTO does not roll over to the next year. Emergency leave can be taken with manager approval. Sick leave is separate from vacation time and does not count against PTO balance.",
            "category": "benefits"
        }
    ]


# ========================================
# CHUNKING LOGIC
# ========================================

def split_documents(documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Splits raw text documents into smaller chunks for the Vector DB.
    """
    if not documents:
        return []

    print(f"\n✂️  Splitting {len(documents)} documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Good size for general text
        chunk_overlap=200,  # Overlap helps keep context at boundaries
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []

    for doc in documents:
        # Split the content
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


# ========================================
# MAIN ORCHESTRATOR
# ========================================

def load_and_chunk_documents(data_dir="./data"):
    """
    Main function to load docs from disk and chunk them.
    If no files are found in data_dir, it falls back to demo data.
    """
    print("\n📚 SECTION: DOCUMENT LOADING & CHUNKING")
    print("=" * 50)

    # 1. Try loading real TXT files
    raw_docs = load_text_files_from_directory(data_dir)

    # 2. Fallback to Demo Data if directory is empty or missing
    if not raw_docs:
        print("\n⚠️  No text files found. Loading DEMO data instead...")
        raw_docs = get_demo_documents()

    # 3. Chunk whatever we found
    return split_documents(raw_docs)


# Allow running this file directly to test it
if __name__ == "__main__":
    # Create a dummy file to test
    if not os.path.exists("./data"):
        os.makedirs("./data")
        with open("./data/test_policy.txt", "w") as f:
            f.write("This is a test policy for the python system.")

    chunks = load_and_chunk_documents("./data")
    if chunks:
        print(f"\nExample Chunk:\n{chunks[0]}")