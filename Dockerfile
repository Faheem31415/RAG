FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_URL=http://ollama:11434

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir requests numpy

COPY . .

CMD ["python", "rag.py"]
