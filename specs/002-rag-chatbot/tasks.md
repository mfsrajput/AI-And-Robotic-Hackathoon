# Detailed Tasks – Integrated RAG Chatbot

## Prioritized Task List

### Phase 1: Single-File Ingestion Pipeline

- [ ] Implement get_all_urls() → hardcode or scrape list of chapter URLs from the live site
- [ ] Implement extract_text_from_url(url) → use requests + BeautifulSoup
- [ ] Implement chunk_text(text) → ~800 tokens with overlap
- [ ] Implement embed(texts) → OpenAI text-embedding-3-large
- [ ] Implement create_collection() → dimension 3072, COSINE
- [ ] Implement save_chunks_to_qdrant(chunks with metadata: url, title)
- [ ] Add main() function that runs full pipeline when executed
- [ ] Test locally with Qdrant Cloud credentials

### Phase 2: Backend RAG Query Endpoint

- [ ] Implement query endpoint with similarity search
- [ ] Add response formatting with source citations
- [ ] Implement selected-text query handling
- [ ] Add error handling and validation

### Phase 3: Frontend Integration

- [ ] Create floating chat UI component
- [ ] Implement text selection detection
- [ ] Add "Ask about this" button functionality
- [ ] Integrate with backend API endpoints

### Phase 4: Testing & Deployment

- [ ] Write integration tests
- [ ] Deploy backend to Render/Fly.io
- [ ] Integrate frontend with live backend
- [ ] End-to-end testing