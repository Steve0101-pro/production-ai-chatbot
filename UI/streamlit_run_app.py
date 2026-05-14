import streamlit as st
import requests
import uuid
import os
import json

API_URL = os.getenv(
    "API_URL",
    "https://ai-chatbot-lsav.onrender.com"
)

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="AI Chat",
    layout="wide"
)

# ==================================================
# HEADER
# ==================================================
st.markdown(
    """
    <h1>
        🤖 AI Chat + Memory
        <a href="https://www.flaticon.com/free-icons/chatbot"
           target="_blank"
           style="text-decoration:none; font-size:18px;">
           🔗
        </a>
    </h1>
    """,
    unsafe_allow_html=True
)

# ==================================================
# SESSION STATE
# ==================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "db" not in st.session_state:
    st.session_state.db = []

# ==================================================
# SIDEBAR AUTH
# ==================================================
st.sidebar.title("🔐 Authentication")

user_api_key = st.sidebar.text_input(
    "Enter API Key to continue",
    type="password"
)

# ==================================================
# BLOCK APP UNTIL AUTH
# ==================================================
if not user_api_key:
    st.sidebar.warning(
        "🔐 Please enter your API key."
    )
    st.stop()

st.sidebar.success("✅ Authenticated")

# ==================================================
# MEMORY SEARCH
# ==================================================
st.sidebar.title("🧠 Search Past Conversation")

search_query = st.sidebar.text_input(
    "Search past conversations"
)

if search_query:

    try:

        res = requests.post(
            f"{API_URL}/search_memory",
            json={
                "api_key": user_api_key,
                "messages": search_query,
                "chat_history": [],
                "session_id": st.session_state.session_id
            }
        )

        data = res.json()

        st.sidebar.write("### Results")

        for r in data.get("results", []):
            st.sidebar.write("•", r)

    except Exception as e:

        st.sidebar.error(f"Search error: {e}")

# ==================================================
# CONVERSATIONS
# ==================================================
st.sidebar.title("🧵 Conversations")

for convo in sorted(
    st.session_state.db,
    key=lambda x: x.get("updated_at", ""),
    reverse=True
):

    col1, col2 = st.sidebar.columns([4, 1])

    # LOAD CHAT
    if col1.button(
        convo.get("title", "Untitled"),
        key=convo["session_id"]
    ):

        st.session_state.session_id = convo["session_id"]

        st.session_state.messages = convo["messages"]

        st.session_state.chat_history = convo["messages"]

        st.rerun()

    # DELETE CHAT
    if col2.button(
        "🗑️",
        key=f"del_{convo['session_id']}"
    ):

        st.session_state.db = [
            c for c in st.session_state.db
            if c["session_id"] != convo["session_id"]
        ]

        st.rerun()

# ==================================================
# NEW CHAT
# ==================================================
if st.sidebar.button("➕ New Chat"):

    st.session_state.session_id = str(uuid.uuid4())

    st.session_state.messages = []

    st.session_state.chat_history = []

    st.rerun()

# ==================================================
# DISPLAY CHAT
# ==================================================
st.subheader("Conversation")

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# ==================================================
# USER INPUT
# ==================================================
user_input = st.chat_input("Ask anything...")

if user_input:

    # ---------------- USER MESSAGE ---------------- #
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    st.chat_message("user").markdown(user_input)

    # ---------------- PAYLOAD ---------------- #
    payload = {
        "api_key": user_api_key,
        "messages": user_input,
        "chat_history": st.session_state.chat_history,
        "session_id": st.session_state.session_id
    }

    # ---------------- ASSISTANT UI ---------------- #
    placeholder = st.chat_message("assistant").empty()

    full_response = ""

    # ==================================================
    # STREAMING RESPONSE
    # ==================================================
    try:

        with st.spinner("Thinking... 🤖"):

            response = requests.post(
                f"{API_URL}/chat",
                json=payload,
                stream=True
            )

            response.raise_for_status()

            for line in response.iter_lines():

                if line:

                    decoded = (
                        line.decode("utf-8")
                        .replace("data: ", "")
                    )

                    data = json.loads(decoded)

                    # ---------------- TOKEN ---------------- #
                    if "token" in data:

                        token = data["token"]

                        full_response += token

                        placeholder.markdown(
                            full_response + "▌"
                        )

                    # ---------------- FINISHED ---------------- #
                    if data.get("done"):

                        placeholder.markdown(
                            full_response
                        )

                        break

    except Exception as e:

        full_response = f"⚠️ Error: {e}"

        placeholder.markdown(full_response)

    # ==================================================
    # SAVE CHAT
    # ==================================================
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": full_response
    })

    # ==================================================
    # SAVE CONVERSATION
    # ==================================================
    existing = next(
        (
            c for c in st.session_state.db
            if c["session_id"] == st.session_state.session_id
        ),
        None
    )

    if existing:

        existing["messages"] = (
            st.session_state.messages.copy()
        )

        existing["updated_at"] = str(uuid.uuid4())

    else:

        st.session_state.db.append({
            "session_id": st.session_state.session_id,
            "title": user_input[:30],
            "messages": st.session_state.messages.copy(),
            "updated_at": str(uuid.uuid4())
        })