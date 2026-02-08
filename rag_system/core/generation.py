# ========================================
# CONTEXT AUGMENTATION
# ========================================
from typing import List, Dict, Any
import ollama

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
    """

    print(f"📝 Augmented prompt length: {len(augmented_prompt)} characters")
    print(f"🔗 Context sources: {[result['metadata']['title'] for result in search_results]}")

    return augmented_prompt

# ========================================================================================================================
# SECTION 6: RESPONSE GENERATION
# ========================================================================================================================

def generate_response(augmented_prompt: str, model_name: str = "llama3.2") -> str:
    """
    Generate response using a local Ollama model.

    Args:
        augmented_prompt: The full prompt with context.
        model_name: The name of the model to use (default: "llama3")
    """
    print("\n🤖 SECTION 6: RESPONSE GENERATION")
    print("=" * 50)
    print(f"⚙️  Connecting to Ollama (Model: {model_name})...")

    try:
        # Call the Ollama API
        response_object = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': augmented_prompt
                },
            ],
            options={
                'temperature': 0.1  # Low temperature for factual RAG responses
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