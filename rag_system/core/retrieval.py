from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Any

# Global variable to store the model in memory
_EMBEDDING_MODEL = None

def get_embedding_model():
    """
    Singleton pattern to load the model only once.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        print("⚙️  Loading embedding model (this happens only once)...")
        # You can swap this for 'all-mpnet-base-v2' for better quality (but slower)
        _EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBEDDING_MODEL

# ========================================
# QUERY PROCESSING
# ========================================

def process_user_query(query: str) -> List[float]:
    """
    Process user query and convert to embedding for vector search.

    Returns:
        List[float]: The vector embedding of the query.
    """
    print("\n🔍 SECTION: QUERY PROCESSING")
    print("=" * 50)

    # 1. Get the cached model
    model = get_embedding_model()

    # 2. Preprocess query (Basic cleaning)
    cleaned_query = query.lower().strip()

    # 3. Generate Embedding
    # encode() returns a numpy array. We convert to list for JSON/DB compatibility.
    query_embedding = model.encode(cleaned_query)

    # Convert numpy array to python list
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    print(f"📝 Query: '{cleaned_query}'")
    print(f"🔢 Vector Dimensions: {len(query_embedding)}")

    return query_embedding