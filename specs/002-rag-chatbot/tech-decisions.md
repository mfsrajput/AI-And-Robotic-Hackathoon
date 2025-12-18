# Technical Decisions for Integrated RAG Chatbot

## Locked Technical Decisions

- **Backend**: FastAPI (async, modern, easy deployment)
- **Vector DB**: Qdrant Cloud Free Tier (managed, generous limits)
- **Embeddings**: OpenAI text-embedding-3-large (best semantic accuracy)
- **LLM**: gpt-4o-mini primary, fallback to gpt-4o (cost + speed balance)
- **Chunking**: 800 tokens with 200 overlap (balances context and precision)
- **Frontend**: Custom React components in Docusaurus (static GitHub Pages compatible)
- **Selected-text handling**: Client-side text selection detection + send to backend
- **Deployment**: Backend on Render.com or Fly.io free tier