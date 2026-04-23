from langchain_nvidia_ai_endpoints import ChatNVIDIA
from app.config import nvidia_model

def get_llm_client(api_key: str):
   try: 
    client = ChatNVIDIA(api_key=api_key, 
                      model=nvidia_model, 
                      max_completion_tokens=1024,
                      temperature=0.7, top_p=0.9, 
                      frequency_penalty=0, 
                      presence_penalty=0)
    return client
   except Exception as e:
          print(f"Failed to initialize LLM client: {e}")
          raise ValueError("Invalid NVIDIA LLM API Key")

