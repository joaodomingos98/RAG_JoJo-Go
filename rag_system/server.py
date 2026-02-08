import time
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Import your existing RAG modules
from core.retrieval import process_user_query
from core.database import search_vector_database
from core.generation import augment_prompt_with_context, generate_response

app = FastAPI(title="Local RAG API")


# Define the data format Open WebUI sends
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: List[Message]


# ========================================
# RAG LOGIC WRAPPER
# ========================================
def run_rag_logic(query: str):
    """
    Your custom RAG pipeline logic (simplified for the API).
    """
    print(f"\n📩 API Request: {query}")

    # 1. Retrieval
    query_embedding = process_user_query(query)
    results = search_vector_database(query_embedding, top_k=3)

    if not results:
        return "I checked the policy documents but couldn't find any relevant information."

    # 2. Augmentation
    augmented_prompt = augment_prompt_with_context(query, results)

    # 3. Generation (Using your Ollama connection)
    # Note: You can pass the model name requested by the UI if you want
    response = generate_response(augmented_prompt, model_name="llama3")

    return response


# ========================================
# OPENAI-COMPATIBLE ENDPOINT
# ========================================
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    This endpoint mimics OpenAI. Open WebUI sends messages here.
    """
    try:
        # 1. Extract the last user message
        last_message = request.messages[-1]
        user_query = last_message.content

        # 2. Run your RAG Pipeline
        response_text = run_rag_logic(user_query)

        # 3. Return response in OpenAI format
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """
    Tells Open WebUI that this server offers a model named 'techcorp-rag'.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "techcorp-rag",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user"
            }
        ]
    }


if __name__ == "__main__":
    # Runs the server on port 8000
    print("🚀 Starting RAG API Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)