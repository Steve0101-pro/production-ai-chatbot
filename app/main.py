import os
import time
import uuid
import json
from contextlib import nullcontext

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.models import ChatRequest
from app.services.auth import api_validation
from app.services.llm import get_llm_client
from app.services.embeddings import get_embedding_client
from app.services.memory import load_memory, search_memory, multi_chat
from app.services.session import save_current_session
from app.utils.logger import logger


# ==================================================
# SAFE OPTIONAL MLFLOW IMPORT (UNCHANGED)
# ==================================================

MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "false").lower() == "true"
mlflow = None

if MLFLOW_ENABLED:
    try:
        import mlflow

        MLFLOW_URI = os.getenv("MLFLOW_URI", "http://127.0.0.1:5000")
        MLFLOW_EXPERIMENT = os.getenv(
            "MLFLOW_EXPERIMENT",
            "fastapi_llm_chatbot_tracking"
        )

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    except Exception as e:
        mlflow = None
        MLFLOW_ENABLED = False
        print(f"[MLFLOW] Disabled due to error: {e}")


# ==================================================
# APP INIT
# ==================================================
app = FastAPI(title="AI Backend API", version="1.0.0")


# ==================================================
# MEMORY LOAD
# ==================================================
try:
    memory = load_memory()
except Exception as e:
    logger.info(f"Memory load failed: {str(e)}")
    memory = []


# ==================================================
# SAFE MLFLOW HELPERS (UNCHANGED)
# ==================================================

def start_run_safe(run_name: str):
    if MLFLOW_ENABLED and mlflow:
        try:
            return mlflow.start_run(run_name=run_name)
        except Exception:
            return nullcontext()
    return nullcontext()


def start_span_safe(name: str):
    if MLFLOW_ENABLED and mlflow and hasattr(mlflow, "start_span"):
        try:
            return mlflow.start_span(name)
        except Exception:
            return nullcontext()
    return nullcontext()


def log_param_safe(key, value):
    if MLFLOW_ENABLED and mlflow:
        try:
            mlflow.log_param(key, value)
        except Exception:
            pass


def log_metric_safe(key, value):
    if MLFLOW_ENABLED and mlflow:
        try:
            mlflow.log_metric(key, value)
        except Exception:
            pass


# ==================================================
# LLM GENERATION
# ==================================================
def llm_generate(client, prompt):
    start = time.time()
    response = ""

    for chunk in client.stream(prompt):
        if getattr(chunk, "content", None):
            response += chunk.content

    duration = time.time() - start
    return response, duration


# ==================================================
# ROUTES
# ==================================================

@app.get("/")
def home():
    return {
        "message": "AI Backend Running 🚀",
        "mlflow_enabled": MLFLOW_ENABLED
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/chat")
def chat_get():
    return JSONResponse(
        status_code=405,
        content={
            "error": "Method Not Allowed",
            "message": "Use POST with JSON payload"
        }
    )


# ==================================================
# 🔥 MAIN CHAT ENDPOINT WITH STREAMING
# ==================================================

@app.post("/chat")
def request_chat(red: ChatRequest):

    start_time = time.time()
    run_name = f"chat-{uuid.uuid4().hex[:8]}"

    try:
        with start_run_safe(run_name):

            # ---------------- INPUT ---------------- #
            api_key = red.api_key
            user_input = red.messages
            session_id = red.session_id
            chat_history = red.chat_history or []

            log_param_safe("session_id", session_id)
            log_param_safe("input_length", len(user_input))

            # ---------------- AUTH ---------------- #
            llm_ok, embed_ok = api_validation(api_key)

            if not llm_ok or not embed_ok:
                log_param_safe("status", "invalid_api_key")
                raise HTTPException(
                    status_code=401,
                    detail="Invalid API Key"
                )

            # ---------------- CLIENTS ---------------- #
            client = get_llm_client(api_key)
            embed = get_embedding_client(api_key)

            # ---------------- HISTORY ---------------- #
            chat_history.append({
                "role": "user",
                "content": user_input
            })

            format_history = multi_chat(chat_history)

            # ---------------- MEMORY SEARCH ---------------- #
            memory_context = ""

            with start_span_safe("memory_retrieval"):

                if embed_ok:
                    try:
                        memory_context = "\n".join(
                            search_memory(user_input, embed)
                        )
                    except Exception:
                        memory_context = ""

            # ---------------- PROMPT ---------------- #
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

            # ==================================================
            # 🔥 STREAMING GENERATOR
            # ==================================================
            def generate():

                start_llm = time.time()

                full_response = ""

                with start_span_safe("llm_generation"):

                    for chunk in client.stream(prompt):

                        if getattr(chunk, "content", None):

                            token = chunk.content

                            full_response += token

                            yield (
                                f"data: {json.dumps({'token': token})}\n\n"
                            )

                # ---------------- SAVE CHAT ---------------- #
                chat_history.append({
                    "role": "assistant",
                    "content": full_response
                })

                # ---------------- SAVE MEMORY ---------------- #
                if embed_ok:
                    try:
                        save_current_session(
                            memory,
                            session_id,
                            chat_history,
                            embed
                        )
                    except Exception as e:
                        logger.info(
                            f"Session save failed: {str(e)}"
                        )

                # ---------------- METRICS ---------------- #
                latency = time.time() - start_time
                llm_time = time.time() - start_llm

                log_metric_safe("latency_sec", latency)
                log_metric_safe("llm_time_sec", llm_time)
                log_metric_safe("count_token", len(prompt))
                log_metric_safe(
                    "response_chars",
                    len(full_response)
                )
                log_metric_safe(
                    "history_messages",
                    len(chat_history)
                )

                log_param_safe("status", "success")
                log_param_safe("model_type", "external_llm")

                # ---------------- DONE SIGNAL ---------------- #
                yield (
                    f"data: {json.dumps({'done': True})}\n\n"
                )

            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )

    except HTTPException:
        raise

    except Exception as e:

        logger.info(f"Error in /chat endpoint: {str(e)}")

        log_param_safe("status", "error")
        log_param_safe("error_message", str(e))

        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )


# ==================================================
# 🔥 SEARCH MEMORY ENDPOINT
# ==================================================

@app.post("/search_memory")
def search_memory_api(red: ChatRequest):

    llm_ok, embed_ok = api_validation(red.api_key)

    if not embed_ok:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )

    try:
        embed = get_embedding_client(red.api_key)

        results = search_memory(
            red.messages,
            embed
        )

        return {
            "query": red.messages,
            "results": results
        }

    except Exception as e:

        logger.info(f"Search memory error: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Search failed"
        )