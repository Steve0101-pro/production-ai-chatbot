
import streamlit as st
import requests
import uuid
from app.services.embeddings import get_embed
import os




# Import memory functions
from app.services.memory import load_memory, save_memory, search_memory

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")

st.set_page_config(page_title="AI Chat", layout="wide")
st.title("AI Chat with Long-Term Memory 🚀")

# ---------------- SESSION ---------------- #
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load memory DB
db = load_memory()

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("🧵 Conversations")

search_query = st.sidebar.text_input("🔍 Search")

user_api_key = st.sidebar.text_input(
    "Enter NVIDIA API Key",
    type="password"
)

if not user_api_key:
    st.sidebar.warning("Please enter your NVIDIA API key.")
    st.stop()
embed=get_embed(user_api_key)
# 🔍 Search memory
if search_query:
    results = search_memory(search_query,embed)
    st.sidebar.write("### 🔎 Results")
    for r in results:
        st.sidebar.write("-",r[:100])
    

# 📂 Show conversations
for convo in sorted(db, key=lambda x: x.get("updated_at", ""), reverse=True):
    col1, col2 = st.sidebar.columns([4, 1])

    if col1.button(convo.get("title", "Untitled"), key=convo["session_id"]):
        st.session_state.session_id = convo["session_id"]
        st.session_state.messages = convo["messages"]
        st.rerun()

    if col2.button("🗑️", key=f"del_{convo['session_id']}"):
        db.remove(convo)
        save_memory(db)
        st.rerun()

# ➕ New chat
if st.sidebar.button("➕ New Chat"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()

# ---------------- DISPLAY CHAT ---------------- #
st.subheader("Conversation")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- USER INPUT ---------------- #
user_input = st.chat_input("Ask anything...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    payload = {
        "api_key": user_api_key,  # ✅ FIXED
        "messages": user_input,
        "chat_history": st.session_state.chat_history,
        "session_id": st.session_state.session_id
    }

    # Call backend
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            res = requests.post(API_URL, json=payload)
            res.raise_for_status()
            data = res.json()
            full_response = data.get("response", "No response")

            # Update chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

        except requests.exceptions.RequestException as e:
            full_response = f"⚠️ Error: {e}"

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})