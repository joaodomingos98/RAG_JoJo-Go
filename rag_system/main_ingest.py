from core.ingestion import load_and_chunk_documents
from core.database import store_chunks_in_db  # We use the NEW storage function


def main():
    print("🚀 STARTING DATA INGESTION")
    print("=" * 60)

    # Step 1: Load and chunk documents
    # (This reads your PDFs/Text files)
    print("Step 1: Loading and chunking documents...")
    chunks = load_and_chunk_documents()
    print(f"✅ Generated {len(chunks)} chunks.")

    # Step 2: Store in Persistent Vector Database
    # (This saves them to the 'chroma_db_data' folder on disk)
    print("Step 2: Storing in Vector Database...")
    store_chunks_in_db(chunks)

    print("=" * 60)
    print("🎉 INGESTION COMPLETE. You can now run main_app.py")


if __name__ == "__main__":
    main()