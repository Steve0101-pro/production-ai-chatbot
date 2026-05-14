FROM python:3.13.11

WORKDIR /app

COPY . .
RUN pip install streamlit

COPY . .

EXPOSE 10000

CMD ["sh", "-c", "streamlit run streamlit_run_app.py --server.port=$PORT --server.address=0.0.0.0"]
