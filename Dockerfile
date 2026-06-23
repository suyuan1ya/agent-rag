FROM python:3.11-slim

LABEL description="RAG Research Agent — ReAct Agent 学术论文智能问答系统"

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
