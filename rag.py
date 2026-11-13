import json
import requests
import numpy as np
from pathlib import Path
import os

# --------------------------------
# CONFIGURATION
# --------------------------------
OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"

DOCS_DIR = Path(__file__).parent / "docs"

# --------------------------------
# GENERATE EMBEDDINGS
# --------------------------------

def embed(texts):
    r = requests.post(
        f"{OLLAMA}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts
        },
        timeout=120
    )

    r.raise_for_status()

    return np.array(r.json()["embeddings"])


# --------------------------------
# ASK LLM WITH STREAMING
# --------------------------------

def ask_llm(prompt):
    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        },
        stream=True,
        timeout=(10, None)
    )

    r.raise_for_status()

    for line in r.iter_lines():
        if line:
            data = json.loads(line)

            content = data.get("message", {}).get("content", "")

            print(content, end="", flush=True)

            if data.get("done"):
                break

    print()


# --------------------------------
# LOAD DOCUMENTS AND CREATE CHUNKS
# --------------------------------

chunks = []

for file in DOCS_DIR.glob("*.md"):

    text = file.read_text(encoding="utf-8")

    chunks += [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

if not chunks:
    raise RuntimeError("No documents found inside docs/")

print(f"Loaded {len(chunks)} chunks")

# --------------------------------
# GENERATE CHUNK EMBEDDINGS
# --------------------------------

chunk_vectors = embed(chunks)

print(f"Indexed {len(chunks)} chunks successfully!\n")


# --------------------------------
# RETRIEVE RELEVANT CHUNKS
# --------------------------------

def retrieve(question, k=3):

    q = embed([question])[0]

    scores = chunk_vectors @ q / (
        np.linalg.norm(chunk_vectors, axis=1)
        * np.linalg.norm(q)
        + 1e-10
    )

    top_indices = np.argsort(scores)[::-1][:k]

    return [
        chunks[i]
        for i in top_indices
    ]


# --------------------------------
# MAIN RAG CHAT LOOP
# --------------------------------

def main():

    print("================================")
    print("       RAG CHAT APPLICATION     ")
    print("================================")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("Ask: ").strip()

        if not question:
            continue

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Retrieve relevant context
        context = "\n\n".join(retrieve(question))

        # Create prompt
        prompt = f"""
Answer the question using ONLY the context below.

If the answer is not available in the context,
say "I don't know."

Context:
{context}

Question: {question}

Answer:
"""

        # Generate streaming response
        print("\nAnswer: ", end="", flush=True)

        try:
            ask_llm(prompt)

        except requests.exceptions.RequestException as e:
            print(f"\nOllama connection error: {e}")

        except (ValueError, KeyError) as e:
            print(f"\nResponse parsing error: {e}")

        print()


# --------------------------------
# RUN APPLICATION
# --------------------------------

if __name__ == "__main__":
    main()
