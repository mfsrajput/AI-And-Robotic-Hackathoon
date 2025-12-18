# RAG Chatbot Backend – COMPLETE ✅

- **Live Backend**: https://[YOUR-RENDER-APP-NAME].onrender.com (placeholder - deployed to Render.com)
- **Frontend Repo**: https://github.com/mfsrajput/AI-And-Robotic-Hackathoon

The RAG Chatbot backend provides an intelligent question-answering system for the Physical AI & Humanoid Robotics textbook. It leverages Retrieval-Augmented Generation (RAG) to provide accurate, contextually relevant answers based on the textbook content using Cohere's AI models.

## Deployment

- **Platform**: Render.com free tier
- **Runtime**: Python 3.11+
- **Environment**: Automatically deployed with environment variable configuration

## Key Features

- **RAG-Optimized**: Cohere models specifically chosen for retrieval-augmented generation tasks
- **Streaming Responses**: Real-time response generation with Server-Sent Events
- **Selected-Text Queries**: Supports context from selected text in frontend
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Constitution-Compliant**: Strictly follows RAG Chatbot Constitution (accuracy, source fidelity, privacy)
- **Content Freshness**: Refresh endpoint to re-ingest updated textbook content

## API Endpoints

- `POST /chat` - Main chat endpoint (test with: `curl -X POST [BACKEND-URL]/chat -H "Content-Type: application/json" -d '{"query":"test"}'`)
- `POST /query` - Alternative query endpoint with source citations
- `POST /refresh` - Re-ingest textbook content into vector database
- `GET /` - Health check endpoint

## Required Environment Variables

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