# Render 배포용 Dockerfile (Python 3.10)
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD uvicorn rag.webapp:app --host 0.0.0.0 --port $PORT  