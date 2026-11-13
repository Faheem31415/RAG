"""
Root entrypoint for the RAG application.
Provides backward compatibility for direct execution (python rag.py)
and exports core components.
"""

import sys
from pathlib import Path

# Add src/ to sys.path to ensure src/rag is importable when executed directly
_src_dir = Path(__file__).resolve().parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from rag.app import (
    CHAT_MODEL,
    EMBED_MODEL,
    OLLAMA_URL,
    RAGIndex,
    ask_llm,
    chunk_markdown,
    embed,
    get_docs_dir,
    load_documents,
    main,
)

__all__ = [
    "CHAT_MODEL",
    "EMBED_MODEL",
    "OLLAMA_URL",
    "RAGIndex",
    "ask_llm",
    "chunk_markdown",
    "embed",
    "get_docs_dir",
    "load_documents",
    "main",
]

if __name__ == "__main__":
    main()
