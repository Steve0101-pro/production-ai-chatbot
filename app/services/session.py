from datetime import datetime
from app.services.memory import save_memory
from app.services.memory import update_vectorstore


def save_current_session(memory, session_id, messages, embed):
    title = messages[0]["content"][:50] if messages else "New Chat"

    found = False

    for convo in memory:
        if convo["session_id"] == session_id:
            convo["messages"] = messages
            convo["title"] = title
            convo["updated_at"] = str(datetime.now())
            found = True
            break

    if not found:
        new_convo = {
            "session_id": session_id,
            "messages": messages,  # ✅ FIXED
            "title": title,
            "created_at": str(datetime.now()),
            "updated_at": str(datetime.now()),
        }
        memory.append(new_convo)

    # ✅ Update vectorstore ONCE
    convo_data = {
        "session_id": session_id,
        "messages": messages
    }
    update_vectorstore(convo_data, embed)

    # ✅ Save to JSON
    save_memory(memory)