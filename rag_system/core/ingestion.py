
# ========================================
# DOCUMENT LOADING & CHUNKING
# ========================================

from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_documents():
    """
    Load sample policy documents and chunk them for better retrieval.

    This section demonstrates:
    - Document loading from sample data
    - Text chunking using LangChain
    - Chunk size and overlap configuration
    """
    print("📚 SECTION 1: DOCUMENT LOADING & CHUNKING")
    print("=" * 50)

    # Sample policy documents (same as previous labs)
    policy_documents = [
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

    print(f"📄 Loaded {len(policy_documents)} policy documents")

    # Configure text splitter (same as chunking lab)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # What is the chunk size?
        chunk_overlap=50,  # What is the overlap?
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # Chunk all documents
    all_chunks = []
    for doc in policy_documents:
        chunks = text_splitter.split_text(doc["content"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{doc['id']}_chunk_{i}",
                "title": doc["title"],
                "content": chunk,
                "category": doc["category"],
                "source_doc": doc["id"]
            })

    print(f"✂️ Created {len(all_chunks)} chunks from {len(policy_documents)} documents")
    print(f"📏 Average chunk size: {sum(len(chunk['content']) for chunk in all_chunks) // len(all_chunks)} characters")

    return all_chunks