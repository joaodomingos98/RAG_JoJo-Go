from sentence_transformers import SentenceTransformer
import numpy as np
import os
from typing import List
from config import settings

_EMBEDDING_MODEL = None


def get_embedding_model():
    global _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is None:
        # Check if local model exists
        if os.path.exists(settings.LOCAL_EMBEDDING_PATH):
            print(f"⚙️  Loading local embedding model from: {settings.LOCAL_EMBEDDING_PATH}")
            # Load from the local directory
            _EMBEDDING_MODEL = SentenceTransformer(str(settings.LOCAL_EMBEDDING_PATH))
        else:
            # Fallback: Download from internet if local files are missing
            print(f"⚠️  Local model not found at {settings.LOCAL_EMBEDDING_PATH}")
            print(f"⚙️  Downloading and loading from HuggingFace: {settings.EMBEDDING_MODEL_NAME}...")
            _EMBEDDING_MODEL = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

            # Auto-save it locally for next time
            _EMBEDDING_MODEL.save(str(settings.LOCAL_EMBEDDING_PATH))

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