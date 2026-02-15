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

# AI Models directory
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# 1. EMBEDDING MODEL (SentenceTransformers)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LOCAL_EMBEDDING_PATH = MODELS_DIR / EMBEDDING_MODEL_NAME

# 2. LOCAL LLM (Llama 3.2 GGUF)
# We use the 3B parameter model, Q4_K_M quantization (Balanced quality/speed)
LLM_REPO_ID = "bartowski/Llama-3.2-3B-Instruct-GGUF"
LLM_FILENAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
LOCAL_LLM_PATH = MODELS_DIR / LLM_FILENAME

# LlamaCPP Settings
CONTEXT_WINDOW = 8192  # Llama 3.2 supports up to 128k, but 8k is safe for local RAM
LLM_TEMPERATURE = 0.1

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
COLLECTION_NAME = "ecoativo_docs"
VECTOR_SEARCH_TOP_K = 5

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