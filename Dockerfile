FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir requests numpy

COPY . .

ENV OLLAMA_URL=http://ollama:11434

CMD ["python", "rag.py"]
