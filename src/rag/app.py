import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import requests

# --------------------------------
# CONFIGURATION
# --------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:4b")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "3"))


def get_docs_dir() -> Path:
    """Resolve documents directory from env var or standard project locations."""
    if "DOCS_DIR" in os.environ:
        return Path(os.environ["DOCS_DIR"])
    
    # Try current working directory
    cwd_docs = Path.cwd() / "docs"
    if cwd_docs.exists() and cwd_docs.is_dir():
        return cwd_docs
    
    # Try relative to this file
    file_docs = Path(__file__).resolve().parent.parent.parent / "docs"
    if file_docs.exists() and file_docs.is_dir():
        return file_docs
    
    return cwd_docs


# --------------------------------
# OLLAMA HELPERS
# --------------------------------

def check_ollama_health(timeout: float = 3.0) -> Tuple[bool, str]:
    """Check if Ollama server is reachable."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        if r.status_code == 200:
            return True, "Ollama is ready"
        return False, f"Ollama returned HTTP status {r.status_code}"
    except requests.exceptions.RequestException as e:
        return False, str(e)


def wait_for_ollama(timeout_seconds: int = 60, retry_interval: float = 2.0) -> bool:
    """Poll Ollama until it is responsive or timeout expires."""
    print(f"Connecting to Ollama at {OLLAMA_URL}...", end="", flush=True)
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        healthy, _ = check_ollama_health(timeout=2.0)
        if healthy:
            print(" Connected!")
            return True
        print(".", end="", flush=True)
        time.sleep(retry_interval)
    print(" Timeout!")
    return False


def embed(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for a list of texts using Ollama embed API.
    Batches requests to prevent timeouts on large inputs.
    """
    if not texts:
        return np.empty((0, 0))

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/embed",
                json={
                    "model": EMBED_MODEL,
                    "input": batch,
                },
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(data["error"])
            all_embeddings.extend(data["embeddings"])
        except requests.exceptions.HTTPError as e:
            if r.status_code == 404:
                raise RuntimeError(
                    f"Embedding model '{EMBED_MODEL}' was not found in Ollama. "
                    f"Please run: ollama pull {EMBED_MODEL}"
                ) from e
            raise RuntimeError(f"Ollama embedding request failed: {e}") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Could not connect to Ollama at {OLLAMA_URL}: {e}"
            ) from e

    return np.array(all_embeddings, dtype=np.float32)


# --------------------------------
# MARKDOWN CHUNKING
# --------------------------------

def chunk_markdown(text: str, target_size: int = 600) -> List[str]:
    """
    Chunk markdown documents preserving heading context, tables, and Q&A blocks.
    Avoids orphaned headings and broken tables.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    merged_chunks: List[str] = []
    pending_headers: List[str] = []
    current_chunk = ""

    for p in paragraphs:
        # Detect markdown headers
        if re.match(r"^#{1,6}\s+", p):
            if current_chunk:
                merged_chunks.append(current_chunk.strip())
                current_chunk = ""
            pending_headers.append(p)
        else:
            if pending_headers:
                header_prefix = "\n".join(pending_headers)
                combined = f"{header_prefix}\n\n{p}"
                pending_headers = []
                current_chunk = combined
            elif current_chunk:
                if len(current_chunk) + len(p) + 2 <= target_size:
                    current_chunk += f"\n\n{p}"
                else:
                    merged_chunks.append(current_chunk.strip())
                    current_chunk = p
            else:
                current_chunk = p

    if pending_headers:
        header_text = "\n".join(pending_headers)
        if current_chunk:
            merged_chunks.append(current_chunk.strip())
            current_chunk = header_text
        else:
            current_chunk = header_text

    if current_chunk:
        merged_chunks.append(current_chunk.strip())

    return merged_chunks


def load_documents(docs_dir: Optional[Path] = None) -> List[str]:
    """Load and chunk all markdown files from documents directory."""
    dir_path = docs_dir or get_docs_dir()
    if not dir_path.exists() or not dir_path.is_dir():
        raise RuntimeError(f"Documents directory not found at: {dir_path}")

    all_chunks: List[str] = []
    md_files = list(dir_path.glob("*.md"))

    if not md_files:
        raise RuntimeError(f"No markdown (.md) documents found in: {dir_path}")

    for file in sorted(md_files):
        text = file.read_text(encoding="utf-8")
        file_chunks = chunk_markdown(text)
        all_chunks.extend(file_chunks)

    if not all_chunks:
        raise RuntimeError(f"No text content could be extracted from: {dir_path}")

    return all_chunks


# --------------------------------
# RAG INDEX
# --------------------------------

class RAGIndex:
    def __init__(self, chunks: List[str], chunk_vectors: np.ndarray):
        self.chunks = chunks
        # Normalize chunk vectors for cosine similarity via dot product
        norms = np.linalg.norm(chunk_vectors, axis=1, keepdims=True) + 1e-10
        self.normalized_vectors = chunk_vectors / norms

    @classmethod
    def from_documents(cls, docs_dir: Optional[Path] = None) -> "RAGIndex":
        chunks = load_documents(docs_dir)
        vectors = embed(chunks)
        return cls(chunks, vectors)

    def retrieve(self, question: str, k: int = DEFAULT_TOP_K) -> List[str]:
        if not self.chunks:
            return []

        q_vec = embed([question])[0]
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)

        scores = self.normalized_vectors @ q_norm
        k_actual = min(k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:k_actual]

        return [self.chunks[i] for i in top_indices]


# --------------------------------
# ASK LLM WITH STREAMING
# --------------------------------

def ask_llm(prompt: str, context: Optional[str] = None) -> None:
    """Stream response from Ollama chat API."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an accurate assistant for Brew Haven Café. "
                "Answer the user's question using ONLY the provided context. "
                "If the answer is not available in the context, say 'I don't know.' "
                "Do not invent facts not present in the context."
            ),
        }
    ]

    if context:
        user_content = f"Context:\n{context}\n\nQuestion: {prompt}"
    else:
        user_content = prompt

    messages.append({"role": "user", "content": user_content})

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
                "stream": True,
                "messages": messages,
            },
            stream=True,
            timeout=(10, None),
        )
        r.raise_for_status()

        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(data["error"])

                content = data.get("message", {}).get("content", "")
                print(content, end="", flush=True)

                if data.get("done"):
                    break

        print()

    except requests.exceptions.HTTPError as e:
        if r.status_code == 404:
            raise RuntimeError(
                f"Chat model '{CHAT_MODEL}' not found. Please run: ollama pull {CHAT_MODEL}"
            ) from e
        raise


# --------------------------------
# MAIN APPLICATION
# --------------------------------

def run_chat_query(index: RAGIndex, question: str) -> None:
    """Execute retrieval and ask LLM for a single question."""
    print("\nAnswer: ", end="", flush=True)
    try:
        relevant_chunks = index.retrieve(question, k=DEFAULT_TOP_K)
        context = "\n\n".join(relevant_chunks)
        ask_llm(prompt=question, context=context)
    except requests.exceptions.RequestException as e:
        print(f"\nOllama connection error: {e}")
    except RuntimeError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    print()


def main() -> None:
    print("================================")
    print("       RAG CHAT APPLICATION     ")
    print("================================")
    print(f"Ollama URL   : {OLLAMA_URL}")
    print(f"Chat Model   : {CHAT_MODEL}")
    print(f"Embed Model  : {EMBED_MODEL}")
    print("Type 'exit' or 'quit' to quit.\n")

    # Wait for Ollama to become responsive
    if not wait_for_ollama(timeout_seconds=60):
        print(f"\n[!] Error: Could not connect to Ollama at {OLLAMA_URL} after 60 seconds.")
        print("    If using Docker, check container status with: docker compose logs ollama")
        print("    If running locally, start: ollama serve\n")
        sys.exit(1)

    # Load and index documents
    print("Loading documents and generating embeddings...")
    try:
        index = RAGIndex.from_documents()
        print(f"Successfully loaded and indexed {len(index.chunks)} chunks!\n")
    except Exception as e:
        print(f"\n[!] Failed to initialize knowledge base: {e}")
        sys.exit(1)

    # Check if running in non-interactive / piped mode
    if not sys.stdin.isatty():
        # Read lines from piped input if any
        for line in sys.stdin:
            question = line.strip()
            if not question:
                continue
            if question.lower() in ["exit", "quit"]:
                break
            print(f"Ask: {question}")
            run_chat_query(index, question)
        return

    # Interactive TTY loop
    while True:
        try:
            question = input("Ask: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        run_chat_query(index, question)


if __name__ == "__main__":
    main()
