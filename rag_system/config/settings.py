# TODO: UPDATE ALL RELEvANT FILES

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# Resolve the root directory (rag_system/)
# logical path: settings.py -> config/ -> rag_system/
BASE_DIR = Path(__file__).resolve().parent.parent

# Define data directories
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db_data"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Embedding Model (HuggingFace / SentenceTransformers)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# EMBEDDING_MODEL_NAME = "all-mpnet-base-v2" # Better quality, slower

# LLM Model (Ollama)
LLM_MODEL_NAME = "llama3.2"
LLM_TEMPERATURE = 0.1  # Low temperature = more factual/deterministic


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
COLLECTION_NAME = "techcorp_policies"
VECTOR_SEARCH_TOP_K = 3

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# =============================================================================
#❗NOT IMPLEMENTED (requires altering search function)❗
# Distance Metric Configuration
# =============================================================================
# D - Options: "cosine" (Default for text), "l2" (Euclidean), "ip" (Inner Product)
DISTANCE_METRIC = "cosine"