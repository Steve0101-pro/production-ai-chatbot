import os
import json
import datetime
import uuid
from langchain_community.vectorstores import FAISS
from app.utils.logger import logger
from app.config import memory_file, faiss_file



def load_memory():
    """Load memory from JSON file. Return empty list if missing or invalid."""
    if not os.path.exists(memory_file):
        return []

    try:
        with open(memory_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except json.JSONDecodeError as e:
        logger.info(f"Memory file is empty or corrupted: {e}")
        return []
    except Exception as e:
        logger.info(f"Unexpected error loading memory: {e}")
        return []


def save_memory(memory):
    """Save memory to JSON file."""
    try:
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)
    except Exception as e:
        logger.info(f"Failed to save memory: {e}")


def load_vectorstore(embed):
    """Load FAISS vectorstore if exists, else return None."""
    if os.path.exists(faiss_file):
        try:
            return FAISS.load_local(
                faiss_file,
                embeddings=embed,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.info(f"Failed to load FAISS vectorstore: {e}")
            return None
    return None


def save_vectorstore(vs):
    """Save FAISS vectorstore."""
    try:
        vs.save_local(faiss_file)
    except Exception as e:
        logger.info(f"Failed to save FAISS vectorstore: {e}")


def update_vectorstore(convo, embed):
    """Update or create FAISS vectorstore with new conversation."""
    if not embed:
        logger.warning("Embeddings client not provided. Skipping vectorstore update.")
        return

    texts, metadata = [], []

    for msg in convo.get("messages", []):
        content = msg.get("content")
        if content and isinstance(content, str) and content.strip():
            texts.append(content.strip())
            metadata.append({
                "session_id": str(uuid.uuid4()),
                "role": msg.get("role"),
                "time": str(datetime.datetime.now())
            })

    if not texts:
        return

    vs = load_vectorstore(embed)
    if vs is None:
        vs = FAISS.from_texts(texts, embedding=embed, metadatas=metadata)
    else:
        vs.add_texts(texts, metadatas=metadata)

    save_vectorstore(vs)


def search_memory(query, embed, k=3):
    """Search vectorstore for relevant past messages."""
    if not embed:
        return []

    vs = load_vectorstore(embed)
    if vs is None:
        return []

    try:
        docs = vs.similarity_search(query, k=k)
        return [d.page_content for d in docs]
    except Exception as e:
        logger.info(f"Error searching FAISS vectorstore: {e}")
        return []


def multi_chat(chat_history):
    """Format chat history for prompt input."""
    formatted = ""
    for chat in chat_history:
        role = chat.get("role")
        content = chat.get("content")
        if role and content:
            formatted += f"<{role.capitalize()}>\n{content}\n</{role.capitalize()}>\n"
    return formatted