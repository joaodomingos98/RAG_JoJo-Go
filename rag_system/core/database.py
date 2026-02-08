# ========================================
# VECTOR DATABASE SETUP
# ========================================

import chromadb
from typing import List, Dict, Any

def setup_vector_database(chunks: List[Dict]):
    """
    Set up ChromaDB vector database and store document chunks.

    This section demonstrates:
    - ChromaDB client initialization
    - Collection creation
    - Document embedding and storage
    - Vector database configuration
    """
    print("\n🗄️ SECTION 2: VECTOR DATABASE SETUP")
    print("=" * 50)

    # Initialize ChromaDB client
    client = chromadb.Client()

    # Create collection (what is the collection name?)
    try:
        collection = client.create_collection(
            name="techcorp_policies",  # What is the collection name?
            metadata={"hnsw:space": "cosine"}  # What similarity metric is used?
        )
    except Exception:
        # Collection already exists, get it
        collection = client.get_collection("techcorp_policies")

    print(f"🗄️ Created collection: {collection.name}")
    print(f"📊 Similarity metric: {collection.metadata['hnsw:space']}")

    # Prepare data for storage
    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [{"title": chunk["title"], "category": chunk["category"], "source": chunk["source_doc"]} for chunk in
                 chunks]

    # Add documents to collection (embeddings will be generated automatically)
    if collection.count() == 0:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"✅ Stored {len(chunks)} chunks in vector database")
    else:
        print(f"✅ Collection already contains {collection.count()} chunks")

    print(f"📈 Collection count: {collection.count()}")

    return collection

# ========================================================================================================================
# VECTOR SEARCH
# ========================================================================================================================

def search_vector_database(collection, query_embedding, top_k: int = 3):
    """
    Search vector database for relevant document chunks.

    This section demonstrates:
    - Vector similarity search
    - Result ranking and filtering
    - Similarity scoring
    - Top-k result selection
    """
    print("\n🔍 SECTION 4: VECTOR SEARCH")
    print("=" * 50)

    # Perform vector search
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k  # How many results are returned?
    )

    print(f"🎯 Searching for top {top_k} results")
    print(f"📊 Found {len(results['ids'][0])} relevant chunks")

    # Process and display results
    search_results = []
    for i, (doc_id, distance, content, metadata) in enumerate(zip(
            results['ids'][0],
            results['distances'][0],
            results['documents'][0],
            results['metadatas'][0]
    )):
        similarity = 1 - distance  # Convert distance to similarity
        search_results.append({
            'id': doc_id,
            'content': content,
            'metadata': metadata,
            'similarity': similarity
        })

        print(f"\n{i + 1}. {metadata['title']} (Category: {metadata['category']})")
        print(f"   Similarity: {similarity:.3f}")
        print(f"   Content: {content[:100]}...")

    return search_results