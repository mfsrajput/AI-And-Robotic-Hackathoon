# Physical AI & Humanoid Robotics Textbook with RAG Chatbot

**Project Status**: COMPLETE ✅
**Live URL**: https://mfsrajput.github.io/AI-And-Robotic-Hackathoon/
**Backend Repo**: https://github.com/mfsrajput/Backend-AI-And-Robotic-Hackathoon

This comprehensive textbook site features an integrated Retrieval-Augmented Generation (RAG) chatbot using Cohere's embedding and language models for intelligent question-answering about Physical AI & Humanoid Robotics.

## Features

- **RAG Chatbot**: Intelligent question-answering system with streaming responses
- **Cohere Integration**: Uses Cohere's embed-english-v3.0 for embeddings and command-r-08-2024 for generation
- **Qdrant Vector Database**: Stores document embeddings for efficient retrieval
- **Selected-Text Queries**: Supports "Ask about this" functionality for selected text context
- **Content Freshness**: API endpoint to refresh content from the live textbook site
- **Constitution-Compliant**: Strictly follows RAG Chatbot Constitution (accuracy, source fidelity, privacy)
- **Streaming Responses**: Real-time response generation with typing indicators
- **Source Citations**: Automatic source attribution with links to original content

## How to Run Locally

### Frontend (Textbook Site)

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run start
```

The site will be available at `http://localhost:3000`

### Backend (RAG Chatbot API)

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:
```bash
cp .env.example .env
# Edit .env with your actual API keys
```

### Environment Variables

- `COHERE_API_KEY`: Your Cohere API key
- `QDRANT_URL`: Your Qdrant Cloud URL
- `QDRANT_API_KEY`: Your Qdrant API key

## Usage

### 1. Ingest Content into Vector Database

Run the initial content ingestion:

```bash
python backend/main.py
```

This will scrape the live textbook site, chunk the content, generate embeddings with Cohere, and store them in Qdrant.

### 2. Start the Backend Server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## API Endpoints

- `GET /` - Health check
- `POST /query` - Query the RAG system with Cohere
- `POST /refresh` - Refresh content from the live site

### Query Endpoint

Request:
```json
{
  "query": "Your question here",
  "selected_text": "Optional selected text for context"
}
```

Response:
```json
{
  "response": "The answer from Cohere",
  "sources": [
    {
      "title": "Page title",
      "url": "Page URL"
    }
  ]
}
```

## Cohere Models Used

- **Embeddings**: `embed-english-v3.0` with input_type "search_document" for documents and "search_query" for queries
- **Generation**: `command-r-08-2024` with strict system message enforcing RAG Chatbot Constitution

## Content Refresh

To refresh content when the textbook is updated:

```bash
# Command line
python backend/main.py refresh

# Or via API
curl -X POST http://localhost:8000/refresh
```

## Deployment Notes

### Frontend Deployment
- Deployed to GitHub Pages at: https://mfsrajput.github.io/AI-And-Robotic-Hackathoon/
- Built with Docusaurus and automatically deployed via GitHub Actions

### Backend Deployment
- Deployed to Render.com for the RAG Chatbot API
- Environment variables configured securely on Render
- Auto-scaling enabled for handling variable load
- Cold start handling implemented for free tier performance

## Troubleshooting

- Make sure your Cohere API key is valid and has sufficient quota
- Check that the Qdrant collection name is `rag_embedding` with dimension 1024
- Ensure the Cohere embed-english-v3.0 model is accessible with your API key
- For local development, ensure both frontend and backend are running simultaneously
- If experiencing slow responses, this may be due to Render.com free tier cold starts