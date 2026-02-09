# ========================================
# CONTEXT AUGMENTATION
# ========================================
from typing import List, Dict, Any
import ollama
from config import settings

def augment_prompt_with_context(query: str, search_results: List[Dict]) -> str:
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
    You are a helpful assistant for a private company. 
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
    Generate response using model defined in settings.

    Args:
        augmented_prompt: The full prompt with context.
    """
    print("\n🤖 SECTION 6: RESPONSE GENERATION")
    print("=" * 50)
    print(f"⚙️  Connecting to Ollama (Model: {settings.LLM_MODEL_NAME})...")

    try:
        # Call the Ollama API
        response_object = ollama.chat(
            model = settings.LLM_MODEL_NAME,
            messages=[
                {
                    'role': 'user',
                    'content': augmented_prompt
                },
            ],
            options={
                'temperature': settings.LLM_TEMPERATURE  # Temperature defines how factual RAG responses are
            }
        )

        # Extract the actual text content
        generated_text = response_object['message']['content']

        print(f"✅ Generated response length: {len(generated_text)} characters")
        return generated_text

    except  Exception as e:
        error_msg = f"❌ Error connecting to Ollama: {str(e)}"
        print(error_msg)
        print("💡 Tip: Is 'ollama serve' running? Did you run 'ollama pull llama3'?")
        return "System Error: Could not generate response."