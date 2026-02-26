import chromadb
import os
from typing import List, Dict, Any
from config import settings


def get_db_client():
    """Returns a persistent ChromaDB client using path from settings."""
    # Convert Path object to string for Chroma
    db_path = str(settings.CHROMA_DB_DIR)

    client = chromadb.PersistentClient(path=db_path)
    return client


def get_collection(client):
    """Gets collection using name from settings."""
    return client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        metadata={"hnsw:space": settings.DISTANCE_METRIC}
    )


# ========================================
# STORAGE (Write)
# ========================================
def store_chunks_in_db(chunks: List[Dict]):
    client = get_db_client()
    collection = get_collection(client)

    print(f"🗄️  Persisting to: {settings.CHROMA_DB_DIR}")

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [{"title": c["title"], "source": c["source_doc"], "category":c["category"]} for c in chunks]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"✅  Stored {len(ids)} chunks in '{settings.COLLECTION_NAME}'")

# ========================================
# VECTOR SEARCH (Read)
# ========================================

def search_vector_database(query_embedding: List[float]):
    """
    Search vector database for relevant document chunks.
    Note: We don't pass 'collection' in; we load it here.
    """

    print("\n🔍 SECTION: VECTOR SEARCH")
    print("=" * 50)

    top_k = settings.VECTOR_SEARCH_TOP_K

    # Re-connect to the SAME persistent DB
    client = get_db_client()
    collection = get_collection(client)

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