# Brew Haven Café - RAG Chat Application

A local Retrieval-Augmented Generation (RAG) assistant for **Brew Haven Café**. This application indexes the café's official handbook and answers customer questions using local Large Language Models and embedding models powered by Ollama.

---

## Architecture & Features

- **Document Ingestion & Heading-Aware Chunking**: Intelligently chunks markdown files (`docs/handbook.md`) keeping section headings, menus, pricing tables, and Q&A blocks together for superior context retrieval.
- **Local Embeddings**: Generates document and query embeddings using `nomic-embed-text` via Ollama.
- **Vector Search**: Fast cosine similarity retrieval via normalized dot products to find top matching context chunks.
- **Streaming LLM Generation**: Streams tokens in real time from `qwen3:4b` using Ollama's Chat API with strict system guardrails ("answer ONLY using context").
- **Docker Compose Ready**: Full containerized orchestration running Ollama, model initialization, and the interactive RAG CLI application.

---

## Quick Start with Docker (Recommended)

Run the entire stack with Docker Compose:

```bash
docker compose up --build
```

Docker Compose will:
1. Start the Ollama server container.
2. Run `rag-model-init` to automatically pull `qwen3:4b` and `nomic-embed-text`.
3. Launch the `rag-app` container attached to interactive stdin/tty.

To attach to the interactive chat session:
```bash
docker attach rag-app
```

---

## Local Setup (Without Docker)

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
or install the package in editable mode:
```bash
pip install -e .
```

### 3. Pull Required Models
Make sure Ollama is running (`ollama serve`), then pull the models:
```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

### 4. Run the Application
You can run the application directly:
```bash
python rag.py
```
Or use the installed console script:
```bash
rag
```
Or run as a module:
```bash
python -m rag
```

---

## Configuration

The application can be configured via environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | URL of the Ollama server |
| `CHAT_MODEL` | `qwen3:4b` | Ollama model name used for chat responses |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model name used for embeddings |
| `DOCS_DIR` | `./docs` | Path to the directory containing markdown files |
| `TOP_K` | `3` | Number of context chunks retrieved per question |

---

## Project Structure

```text
├── Dockerfile              # Container definition for the Python RAG app
├── README.md               # Documentation and usage guide
├── compose.yaml            # Docker Compose orchestration (Ollama + Models + App)
├── docs/
│   └── handbook.md         # Brew Haven Café knowledge base handbook
├── pyproject.toml          # Project metadata and package configuration
├── rag.py                  # Root CLI runner / backward compatibility entrypoint
├── requirements.txt        # Python package dependencies
├── scripts/
│   └── init-models.sh      # Automated script to pull models into Ollama
└── src/
    └── rag/
        ├── __init__.py     # Package exports
        ├── __main__.py     # Entrypoint for python -m rag
        └── app.py          # Core RAG implementation (chunking, embed, retrieve, LLM)
```
