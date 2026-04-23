# 🚀 AI Chatbot System

### FastAPI + Streamlit + Memory + MLflow + CI/CD

A production-style AI chatbot platform built with modern backend engineering, LLM integration, long-term memory, experiment tracking, and automated CI/CD.

---

# 🔥 Project Highlights

✅ FastAPI Backend API  
✅ Streamlit Frontend UI  
✅ Long-Term Memory Retrieval  
✅ Session-Based Conversations  
✅ NVIDIA / External LLM Support  
✅ MLflow Experiment Tracking  
✅ GitHub Actions CI/CD  
✅ Docker Ready  
✅ Deployable Full-Stack AI App

---

# 🧠 Architecture

```text id="0f18c0"
User
 ↓
Streamlit Frontend
 ↓
FastAPI Backend
 ↓
LLM Engine + Embeddings
 ↓
Memory Retrieval
 ↓
MLflow Tracking


⚙️ Tech Stack
Layer	    Technology
Frontend	Streamlit
Backend	    FastAPI
AI Model	NVIDIA
Memory	    FAISS / Local Vector DB
Tracking	MLflow
CI/CD	    GitHub Actions
Deployment	Render / Railway / VPS


project/
│── app/
│   ├── main.py
│   ├── models.py
│   ├── services/
│   └── utils/
│
│── streamlit_run_app.py
│── requirements.txt
│── Dockerfile
│── .github/workflows/deploy.yml
│── README.md
```

Features
✅ FastAPI Backend

Handles:

Chat requests
Session memory
Prompt construction
LLM responses
Error handling
Metrics logging
✅ Streamlit Frontend

Provides:

ChatGPT-style UI
Conversation history
Searchable memory
Session switching
API key input
✅ MLflow Tracking

Tracks:

Request latency
Prompt length
Response size
Errors
Session runs
✅ CI/CD Automation

Every push to main automatically:

Validates code
Starts FastAPI
Starts Streamlit
Runs health checks
Confirms deploy readiness

# Run Locally

```
pip install -r requirements.txt
pip install streamlit
```

# FastAPI

```
uvicorn app.main:app --reload
```

# Strealit

```
streamlit run streamlit_run_app.py
```

# URLs

```
FastAPI   → http://127.0.0.1:8000
Streamlit → http://127.0.0.1:8501
```

# env variables

```
API_URL=http://127.0.0.1:8000/chat
MLFLOW_URI=http://127.0.0.1:5000
```

# clone Repository

bash``` git clone https://github.com/Steve0101-pro/production-ai-chatbot.git

```
# Why This Project Matters

- This project demonstrates real industry skills:

✅ Backend API Engineering
✅ LLM Application Development
✅ Retrieval-Augmented Memory
✅ Frontend Integration
✅ MLOps Tracking
✅ CI/CD Automation
✅ Deployment Readiness

---

# 🏆 Best Fit Roles

AI Engineer
ML Engineer
MLOps Engineer
GenAI Developer
Backend AI Engineer

# 👤 Author

---

OLANREWAJU STEPHEN
AI/ML Engineer | FastAPI | Streamlit | GenAI

---
```
