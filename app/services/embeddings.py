from app.config import  nvidia_embed
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from app.utils.logger import logger

def get_embedding_client(api_key: str):
   try: 
    return NVIDIAEmbeddings(api_key=api_key, model=nvidia_embed,truncate="NONE")
   except Exception as e:
        logger.info(f"Failed to initialize embeddings: {e}")
        raise ValueError("Invalid NVIDIA Embedding API Key") 
def get_embed(api_key: str):
    return get_embedding_client(api_key)