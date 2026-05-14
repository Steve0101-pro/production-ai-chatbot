FROM python:3.13.11

WORKDIR /app

COPY . .
RUN pip install streamlit

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_run_app.py", "--server.address=0.0.0.0", "--server.port=8501"]