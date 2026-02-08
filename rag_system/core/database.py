import chromadb
import os
from chromadb.config import Settings
from typing import List, Dict, Any

# Define where the DB will be saved on your disk
PERSIST_DIRECTORY = "./chroma_db_data"


def get_db_client():
    """
    Returns a persistent ChromaDB client.
    """
    # Check if the directory exists, creating it if necessary (optional but good practice)
    if not os.path.exists(PERSIST_DIRECTORY):
        os.makedirs(PERSIST_DIRECTORY)

    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    return client


def get_or_create_collection(client, name="techcorp_policies"):
    """
    Gets the collection if it exists, or creates it if it doesn't.
    """
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )


# ========================================
# VECTOR DATABASE INGESTION (Write)
# ========================================

def store_chunks_in_db(chunks: List[Dict]):
    """
    Store document chunks in the persistent Vector DB.
    """
    print("\n🗄️ SECTION: VECTOR DATABASE STORAGE")
    print("=" * 50)

    client = get_db_client()
    collection = get_or_create_collection(client)

    print(f"🗄️  Using Collection: {collection.name}")
    print(f"📂  Persist Directory: {PERSIST_DIRECTORY}")

    # Prepare data
    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [
        {
            "title": chunk["title"],
            "category": chunk["category"],
            "source": chunk.get("source_doc", "unknown")
        }
        for chunk in chunks
    ]

    # Check for duplicates or just add (Chroma handles IDs efficiently)
    # Ideally, you check if IDs exist to avoid duplication if you re-run ingestion
    existing_count = collection.count()

    # Upsert (Update or Insert) is safer than Add if you might run this multiple times
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    new_count = collection.count()
    print(f"✅  Added/Updated {len(ids)} chunks.")
    print(f"📈  Total Collection count: {new_count}")

    return collection


# ========================================
# VECTOR SEARCH (Read)
# ========================================

def search_vector_database(query_embedding: List[float], top_k: int = 3):
    """
    Search vector database for relevant document chunks.
    Note: We don't pass 'collection' in; we load it here.
    """
    print("\n🔍 SECTION: VECTOR SEARCH")
    print("=" * 50)

    # Re-connect to the SAME persistent DB
    client = get_db_client()
    collection = get_or_create_collection(client)

    # Perform vector search
    # Ensure query_embedding is a list, not a numpy array, as Chroma expects lists sometimes
    if hasattr(query_embedding, 'tolist'):
        query_embedding = query_embedding.tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    print(f"🎯 Searching for top {top_k} results")

    # Handle case where no results are found
    if not results['ids'] or len(results['ids'][0]) == 0:
        print("⚠️  No relevant chunks found.")
        return []

    print(f"📊 Found {len(results['ids'][0])} relevant chunks")

    # Process and display results
    search_results = []
    for i, (doc_id, distance, content, metadata) in enumerate(zip(
            results['ids'][0],
            results['distances'][0],
            results['documents'][0],
            results['metadatas'][0]
    )):
        # Cosine distance: 0 is identical, 1 is opposite.
        # Similarity = 1 - distance is a common approximation.
        similarity = 1 - distance

        search_results.append({
            'id': doc_id,
            'content': content,
            'metadata': metadata,
            'similarity': similarity
        })

        print(f"\n{i + 1}. {metadata.get('title', 'Untitled')} (Category: {metadata.get('category', 'General')})")
        print(f"   Similarity: {similarity:.3f}")
        # print(f"   Content: {content[:100]}...") # Uncomment to see preview

    return search_results