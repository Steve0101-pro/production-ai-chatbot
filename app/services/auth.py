from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import nvidia_model, nvidia_embed
from app.utils.logger import logger


def api_validation(api_key: str):
    llm_ok = False
    embed_ok = False

    # 🔹 Validate LLM
    try:
        client = ChatNVIDIA(
            api_key=api_key,
            model=nvidia_model,
            max_tokens=5,
        )

        messages = [
            SystemMessage(content="Validation"),
            HumanMessage(content="test")
        ]

        list(client.stream(messages))
        llm_ok = True

    except Exception as e:
        logger.error(f"LLM validation failed: {e}")

    # 🔹 Validate Embeddings
    try:
        embed_client = NVIDIAEmbeddings(
            api_key=api_key,
            model=nvidia_embed
        )

        embed_client.embed_query("test")
        embed_ok = True

    except Exception as e:
        logger.error(f"Embedding validation failed: {e}")

    return llm_ok, embed_ok