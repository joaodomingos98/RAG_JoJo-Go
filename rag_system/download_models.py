import os
import shutil
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
from config import settings


def download_embedding_model():
    """Downloads the SentenceTransformer model for vector embeddings."""
    print(f"\n⬇️  Step 1/2: Downloading Embedding Model ({settings.EMBEDDING_MODEL_NAME})...")

    if os.path.exists(settings.LOCAL_EMBEDDING_PATH):
        print(f"✅  Embedding model already exists at: {settings.LOCAL_EMBEDDING_PATH}")
        return

    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    model.save(str(settings.LOCAL_EMBEDDING_PATH))
    print(f"🎉  Saved embedding model to: {settings.LOCAL_EMBEDDING_PATH}")


def download_llm_model():
    """Downloads the Llama 3.2 GGUF model for text generation."""
    print(f"\n⬇️  Step 2/2: Downloading Llama 3.2 Model ({settings.LLM_FILENAME})...")
    print(f"    Source: HuggingFace ({settings.LLM_REPO_ID})")

    if os.path.exists(settings.LOCAL_LLM_PATH):
        print(f"✅  Llama model already exists at: {settings.LOCAL_LLM_PATH}")
        return

    try:
        # This function handles the download and progress bar automatically
        hf_hub_download(
            repo_id=settings.LLM_REPO_ID,
            filename=settings.LLM_FILENAME,
            local_dir=settings.MODELS_DIR,
            local_dir_use_symlinks=False  # Ensure we get the actual file, not a symlink
        )
        print(f"🎉  Saved Llama 3.2 model to: {settings.LOCAL_LLM_PATH}")
    except Exception as e:
        print(f"❌  Error downloading LLM: {e}")
        print("    Tip: Check your internet connection or install: pip install huggingface_hub")


if __name__ == "__main__":
    print("🚀  Starting Model Downloads...")
    print("=" * 50)

    download_embedding_model()
    download_llm_model()

    print("=" * 50)
    print("✅  All models are ready! You can now run the system offline.")