# RAG Chatbot Backend

## Overview

The RAG Chatbot backend provides an intelligent question-answering system for the Physical AI & Humanoid Robotics textbook. It leverages Retrieval-Augmented Generation (RAG) to provide accurate, contextually relevant answers based on the textbook content using Cohere's AI models.

## Architecture

The backend is implemented as a single-file FastAPI application (`main.py`) that includes:

- **URL Scraping**: Extracts content from the live textbook GitHub Pages site
- **Text Processing**: Chunks textbook content with overlap for proper context
- **Cohere Integration**: Uses embed-english-v3.0 for embeddings and command-r-08-2024 for generation
- **Qdrant Vector Storage**: Stores embeddings and enables similarity search
- **API Endpoints**: Provides query and refresh functionality

## Technology Stack

- **Framework**: FastAPI for high-performance API
- **Embeddings**: Cohere embed-english-v3.0 (1024 dimensions)
- **Generation**: Cohere command-r-08-2024
- **Vector Database**: Qdrant for storage and retrieval
- **Language**: Python 3.11+

## Key Features

- **RAG-Optimized**: Cohere models specifically chosen for retrieval-augmented generation tasks
- **Cost-Effective**: Cohere provides better cost-performance ratio for educational use
- **Multilingual Ready**: Cohere models support multiple languages if needed
- **Privacy Compliant**: No user data logging, all processing is anonymous
- **Content Freshness**: Refresh endpoint to re-ingest updated textbook content

## API Endpoints

- `POST /query` - Process natural language queries with optional selected text context
- `POST /refresh` - Re-ingest textbook content into vector database
- `GET /` - Health check endpoint

## Cohere Implementation Benefits

1. **RAG-Optimized**: Cohere embeddings are specifically designed for retrieval tasks
2. **Cost-Effective**: Better pricing for educational deployment
3. **Technical Content**: Superior handling of robotics and AI textbook content
4. **Context Length**: Good support for textbook query context requirements
5. **Privacy Safeguards**: Cohere provides appropriate data handling for educational use

## Environment Variables

- `COHERE_API_KEY` - API key for Cohere services
- `QDRANT_URL` - URL for Qdrant vector database
- `QDRANT_API_KEY` - API key for Qdrant database (if required)

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables in `.env` file
3. Run the server: `uvicorn main:app --host 0.0.0.0 --port 8000`

## Constitution Compliance

The backend strictly adheres to the RAG Chatbot Constitution:
- No hallucinations beyond retrieved context
- Always cites sources in responses
- Maintains user privacy with no data logging
- Provides factually accurate responses based on textbook content