#!/usr/bin/env python3
"""
Complete RAG Pipeline Demo - TechCorp PolicyCopilot
===================================================

This script demonstrates a complete RAG (Retrieval-Augmented Generation) system
that combines all the concepts from previous labs:
- Document chunking
- Vector database storage
- Query processing
- Vector search
- Context augmentation
- Response generation

Run this script to see the complete RAG pipeline in action!
"""

from core.ingestion import load_and_chunk_documents
from core.database import setup_vector_database, search_vector_database
from core.retrieval import process_user_query
from core.generation import augment_prompt_with_context, generate_response

# ========================================
# SECTION 7: COMPLETE RAG PIPELINE
# ========================================

def run_complete_rag_pipeline(query: str):
    """
    Run the complete RAG pipeline from start to finish.

    This demonstrates the full flow:
    1. Document loading and chunking
    2. Vector database setup
    3. Query processing
    4. Vector search
    5. Context augmentation
    6. Response generation
    """
    print("\n🚀 COMPLETE RAG PIPELINE DEMO")
    print("=" * 60)
    print(f"❓ User Question: {query}")
    print("=" * 60)

    # Step 1: Load and chunk documents
    chunks = load_and_chunk_documents()

    # Step 2: Setup vector database
    collection = setup_vector_database(chunks)

    # Step 3: Process user query
    model, query_embedding = process_user_query(query)

    # Step 4: Search vector database
    search_results = search_vector_database(collection, query_embedding)

    # Step 5: Augment prompt with context
    augmented_prompt = augment_prompt_with_context(query, search_results)

    # Step 6: Generate response
    response = generate_response(augmented_prompt)

    # Display final result
    print("\n🎉 FINAL RESULT")
    print("=" * 60)
    print(response)
    print("=" * 60)

    return response


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    print("🎯 TechCorp PolicyCopilot - Complete RAG Pipeline Demo")
    print("=" * 60)
    print("This demo shows how all RAG components work together!")
    print("=" * 60)

    # Test queries
    test_queries = [
        "What's the reimbursement policy for home office equipment?",
        "Can I get money back for buying a desk?",
        "How much can I claim for my home office?",
        "What's the travel expense policy?",
        "How many vacation days do I get?"
    ]

    # Run demo for each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"DEMO {i}: {query}")
        print(f"{'=' * 60}")

        try:
            run_complete_rag_pipeline(query)
        except Exception as e:
            print(f"❌ Error in demo {i}: {e}")

        if i < len(test_queries):
            input("\nPress Enter to continue to next demo...")

    print("\n🎉 All demos completed!")
    print("You've seen how the complete RAG pipeline works!")
