FROM bitnami/pytorch:2.0.1

WORKDIR /gen/

COPY src/ src/
COPY checkpoints/ checkpoints/
COPY tokenizers/ tokenizers/
COPY requirements.txt .
COPY app.py .

RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]