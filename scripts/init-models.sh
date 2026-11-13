#!/bin/sh

set -e

echo "Waiting for Ollama..."

until ollama list >/dev/null 2>&1; do
    sleep 3
done

echo "Ollama is ready!"

echo "Downloading Qwen3..."
ollama pull qwen3:4b

echo "Downloading Nomic embeddings..."
ollama pull nomic-embed-text

echo "All models downloaded successfully!"
