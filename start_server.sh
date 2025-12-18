#!/bin/bash
# Startup script for the RAG Chatbot backend

set -e

echo "Starting RAG Chatbot Backend..."

# Check if virtual environment exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if requirements.txt is newer than the installation
if [ ! -f requirements_installed ] || [ requirements.txt -nt requirements_installed ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch requirements_installed
fi

# Check if environment variables are set
if [ -z "$OPENAI_API_KEY" ] || [ -z "$QDRANT_URL" ] || [ -z "$QDRANT_API_KEY" ]; then
    if [ -f .env ]; then
        echo "Loading environment variables from .env file..."
        export $(grep -v '^#' .env | xargs)
    else
        echo "Warning: Environment variables not set. Please set OPENAI_API_KEY, QDRANT_URL, and QDRANT_API_KEY."
        echo "Or create a .env file with these variables."
    fi
fi

# Run the FastAPI server
echo "Starting FastAPI server on port 8000..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

echo "Server stopped."