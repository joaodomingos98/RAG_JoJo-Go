# ========================================
# CONTEXT AUGMENTATION
# ========================================
from typing import List, Dict, Any

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
        context_parts.append(f"Source {i}: {result['metadata']['title']}\n{result['content']}")

    context = "\n\n".join(context_parts)

    print(f"📄 Assembled context from {len(search_results)} sources")
    print(f"📏 Context length: {len(context)} characters")

    # Build augmented prompt
    augmented_prompt = f"""
Based on the following company policies, answer the user's question.

POLICIES:
{context}

QUESTION: {query}

Please provide a clear, accurate answer based on the policies above.
If the information is not available in the policies, say so.
Include relevant policy details and any limitations or requirements.
"""

    print(f"📝 Augmented prompt length: {len(augmented_prompt)} characters")
    print(f"🔗 Context sources: {[result['metadata']['title'] for result in search_results]}")

    return augmented_prompt

# ========================================================================================================================
# SECTION 6: RESPONSE GENERATION
# ========================================================================================================================

def generate_response(augmented_prompt: str) -> str:
    """
    Generate response using LLM (simulated for demo).

    This section demonstrates:
    - LLM integration (simulated)
    - Response formatting
    - Answer synthesis
    - Output structure
    """
    print("\n🤖 SECTION 6: RESPONSE GENERATION")
    print("=" * 50)

    # Simulate LLM processing time
    print("⏳ Processing with LLM...")

    # Simulate LLM response (in production, this would call OpenAI/Anthropic/etc.)
    response = f"""
Based on the company policies provided, here's the answer to your question:

The relevant policies contain information about various company guidelines and procedures. 
The retrieved context provides specific details that can help answer your question.

Key points from the policies:
- Multiple policy sources were consulted
- Information is current and accurate
- Specific requirements and limitations are included

Please refer to the specific policy documents for complete details and any recent updates.
"""

    print(f"✅ Generated response length: {len(response)} characters")
    print(f"📋 Response includes: Policy references, key points, limitations")

    return response