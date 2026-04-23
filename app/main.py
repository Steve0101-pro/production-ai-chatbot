from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.models import ChatRequest
from app.services.auth import api_validation
from app.services.llm import get_llm_client
from app.services.embeddings import get_embedding_client
from app.services.memory import load_memory, search_memory, multi_chat
from app.services.session import save_current_session
from app.utils.logger import logger

import mlflow
import time
import uuid

# =========================
# MLflow CONFIG
# =========================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("fastapi_llm_chatbot")

app = FastAPI()
memory = load_memory()

# =========================
# TRACE HELPERS (IMPORTANT)
# =========================

def llm_generate(client, prompt):
    """LLM traceable wrapper"""
    start = time.time()

    response = ""
    for chunk in client.stream(prompt):
        if chunk.content:
            response += chunk.content

    duration = time.time() - start
    return response, duration


# =========================
# ROUTES
# =========================

@app.get("/")
def home():
    return {"message": "AI Backend Running 🚀"}


@app.get("/chat")
def chat_get():
    return JSONResponse(
        status_code=405,
        content={
            "error": "Method Not Allowed",
            "message": "Use POST with JSON payload"
        }
    )


# =========================
# MAIN CHAT ENDPOINT
# =========================

@app.post("/chat")
def request_chat(red: ChatRequest):

    start_time = time.time()
    run_name = f"chat-{uuid.uuid4().hex[:8]}"

    try:
        with mlflow.start_run(run_name=run_name):

            # =========================
            # INPUT LOGGING
            # =========================
            api_key = red.api_key
            user_input = red.messages
            session_id = red.session_id
            chat_history = red.chat_history

            mlflow.log_param("session_id", session_id)
            mlflow.log_param("input_length", len(user_input))

            # =========================
            # AUTH
            # =========================
            llm_ok, embed_ok = api_validation(api_key)

            mlflow.log_param("llm_ok", llm_ok)
            mlflow.log_param("embed_ok", embed_ok)

            if not llm_ok or not embed_ok:
                mlflow.log_param("status", "invalid_api_key")
                raise HTTPException(status_code=401, detail="Invalid API Key")

            # =========================
            # CLIENTS
            # =========================
            client = get_llm_client(api_key)
            embed = get_embedding_client(api_key)

            chat_history.append({"role": "user", "content": user_input})
            format_history = multi_chat(chat_history)

            # =========================
            # TRACE: MEMORY SEARCH
            # =========================
            memory_context = ""
            with mlflow.start_span("memory_retrieval"):
                if embed_ok:
                    memory_context = "\n".join(
                        search_memory(user_input, embed)
                    )

            # =========================
            # PROMPT BUILDING
            # =========================
            prompt = f"""
<UserQuestion>
{user_input}
</UserQuestion>

<ChatHistory>
{format_history}
</ChatHistory>

<ChatMemory>
{memory_context}
</ChatMemory>

<Instruction>
You are an AI assistant.
Be accurate, concise, and do not hallucinate.
</Instruction>
"""

            # =========================
            # TRACE: LLM GENERATION
            # =========================
            with mlflow.start_span("llm_generation"):
                response, llm_time = llm_generate(client, prompt)

            # =========================
            # HISTORY UPDATE
            # =========================
            chat_history.append({"role": "assistant", "content": response})

            if embed_ok:
                save_current_session(memory, session_id, chat_history, embed)

            # =========================
            # METRICS
            # =========================
            latency = time.time() - start_time

            mlflow.log_metric("latency_sec", latency)
            mlflow.log_metric("llm_time_sec", llm_time)
            mlflow.log_metric("count_token", len(prompt))
            mlflow.log_metric("response_chars", len(response))
            mlflow.log_metric("history_messages", len(chat_history))

            mlflow.log_param("status", "success")

            # =========================
            # OPTIONAL: MODEL ARTIFACT HOOK (for registry later)
            # =========================
            mlflow.log_param("model_type", "external_llm")

            return {
                "response": response,
                "session_id": session_id
            }

    except Exception as e:
        mlflow.log_param("status", "error")
        mlflow.log_param("error_message", str(e))
        logger.info("Error in /chat endpoint")

        raise HTTPException(status_code=500, detail=str(e))