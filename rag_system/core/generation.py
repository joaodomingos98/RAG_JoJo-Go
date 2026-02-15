# ========================================
# CONTEXT AUGMENTATION
# ========================================
import os
import multiprocessing
from llama_cpp import Llama
from config import settings

# Global variable to hold the model in memory (Singleton)
_LLM_ENGINE = None

def get_optimal_threads():
    """
    Dynamically determine the best thread count.
    Rule of thumb: Physical cores usually perform better than Logical cores (Hyperthreading).
    We leave 1 or 2 cores free for the OS/Background tasks.
    """
    try:
        # Get total cores
        count = multiprocessing.cpu_count()
        # Cap at 8 or (Cores - 2), whichever is safer, to avoid CPU contention
        # On many consumer CPUs, going beyond 8 threads yields diminishing returns.
        return max(1, min(count - 2, 8))
    except:
        return 4 # Safe fallback


def get_llm_engine():
    global _LLM_ENGINE
    if _LLM_ENGINE is None:
        print(f"⚙️  Loading local Llama model from: {settings.LOCAL_LLM_PATH}")

        if not os.path.exists(settings.LOCAL_LLM_PATH):
            raise FileNotFoundError(f"❌ Model file not found. Please run download_models.py")

        # Dynamic Hardware Configuration
        n_threads = get_optimal_threads()
        print(f"🖥️  Detected CPU Cores. Using {n_threads} threads for inference.")

        _LLM_ENGINE = Llama(
            model_path=str(settings.LOCAL_LLM_PATH),

            # --- PERFORMANCE SETTINGS ---
            n_ctx=settings.CONTEXT_WINDOW,  # Keep this reasonable (e.g. 2048-8192)
            n_threads=n_threads,  # CPU threads for generation
            n_batch=512,  # How many tokens to process at once (higher = faster prompt processing)

            # --- GPU SETTINGS ---
            # -1 moves ALL layers to GPU. If you don't have a GPU configured,
            # llama-cpp-python will just ignore this or use CPU.
            n_gpu_layers=-1,

            verbose=False  # Set True if you want to see the "tokens per second" stats
        )
    return _LLM_ENGINE

def augment_prompt_with_context(query: str, search_results: list[dict]) -> str:
    """
    Build augmented prompt with retrieved context for LLM.

    This section demonstrates:
    - Context assembly from search results
    - Prompt construction
    - Information formatting
    - Context length management
    """
    print("\n📝 SECTION 5: CONTEXT AUGMENTATION")
    print("=" * 50)

    # Assemble context from search results
    context_parts = []
    for i, result in enumerate(search_results, 1):
        # Fallback if 'title' or 'content' keys might be missing
        title = result.get('metadata', {}).get('title', f"Source {i}")
        content = result.get('content', '')
        context_parts.append(f"Source {i}: {title}\n{content}")

    context = "\n\n".join(context_parts)

    print(f"📄 Assembled context from {len(search_results)} sources")
    print(f"📏 Context length: {len(context)} characters")

    # Build augmented prompt
    augmented_prompt = f"""
    Use the following context to answer the user's question.
        
    CONTEXT:
    {context}
    
    USER QUESTION: 
    {query}
    
    INSTRUCTIONS:
    - Answer strictly based on the provided context.
    - If the answer is not in the context, state "I cannot answer this based on the provided documents."
    - Be concise and professional.
    - If relevant: go outside your prompt and use all the information you know outside of the context given.
    """

    print(f"📝 Augmented prompt length: {len(augmented_prompt)} characters")
    print(f"🔗 Context sources: {[result['metadata']['title'] for result in search_results]}")

    return augmented_prompt

# ========================================================================================================================
# SECTION 6: RESPONSE GENERATION
# ========================================================================================================================

def generate_response(augmented_prompt: str) -> str:
    """
    Generate response using the embedded Llama-cpp engine.

    Args:
        augmented_prompt: The full prompt with context.
    """
    print("\n🤖 SECTION 6: RESPONSE GENERATION")
    print("=" * 50)
    print("✏️  Generating response (Embedded Llama)...")

    llm = get_llm_engine()

    # Create chat completion (OpenAI-compatible format)
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant for a private company."},
            {"role": "user", "content": augmented_prompt}
        ],
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=512  # Limit response length
    )

    # Extract text
    return output['choices'][0]['message']['content']