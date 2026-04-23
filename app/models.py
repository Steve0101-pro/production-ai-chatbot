from typing import List, Dict

from pydantic import BaseModel

class ChatRequest(BaseModel):
    api_key: str
    messages: str
    session_id: str
    chat_history: List[Dict] = []