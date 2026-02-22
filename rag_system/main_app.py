import sys
import time
from core.database import search_vector_database
from core.retrieval import process_user_query
from core.generation import augment_prompt_with_context, generate_response


# ========================================
# RAG INFERENCE PIPELINE
# ========================================

def run_rag_pipeline(query: str):
    """
    Run the RAG pipeline using the existing database.
    """
    print(f"\n❓ User Question: {query}")
    print("-" * 60)

    # Start the timer
    start_time = time.perf_counter()

    # Step 1: Process User Query (Generate Embedding)
    # We assume process_user_query returns (model, embedding_list)
    query_embedding = process_user_query(query)

    # Step 2: Search Vector Database
    # Note: We do NOT pass a 'collection' object anymore.
    # The function connects to the persistent DB automatically.
    search_results = search_vector_database(query_embedding)

    if not search_results:
        print("❌ No relevant information found in the database.")
        return "I couldn't find any information about that in the policy documents.", []

    # Step 3: Augment Prompt
    augmented_prompt = augment_prompt_with_context(query, search_results)

    # Step 4: Generate Response (using Ollama)
    response = generate_response(augmented_prompt)

    # Stop the timer
    end_time = time.perf_counter()

    # Calculate duration in seconds
    elapsed_time_s = (end_time - start_time)

    print("\n🤖 AI Answer:")
    print(response)
    print("-" * 60)
    print(f"⏱️  Answer generated in {elapsed_time_s:.4f} s")

    return response, search_results


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    print("🎯 TechCorp PolicyCopilot - RAG Interface")
    print("=" * 60)

    # Simple loop to chat with the system
    while True:
        try:
            user_input = input("\n📝 Enter your question (or 'q' to quit): ")
            if user_input.lower() in ['q', 'quit', 'exit']:
                break

            run_rag_pipeline(user_input)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            sys.exit()
        except Exception as e:
            print(f"❌ Error: {e}")