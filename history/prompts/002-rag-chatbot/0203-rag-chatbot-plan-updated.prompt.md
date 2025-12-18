---
id: 203
title: Implementation Plan – Integrated RAG Chatbot (Updated for URL-Based Ingestion)
stage: planning
date_iso: 2025-12-17
surface: agent
model: claude-sonnet-4-5-20251201
feature: 002-rag-chatbot
branch: main
user: mfsrajput
command: /sp.plan
labels: ["plan","rag-chatbot","roadmap","url-ingestion","single-file","hackathon"]
files_yaml:
  - specs/002-rag-chatbot/plan.md
  - specs/002-rag-chatbot/tasks.md
  - specs/002-rag-chatbot/architecture.md
  - backend/main.py
  - history/prompts/002-rag-chatbot/0203-rag-chatbot-plan-updated.prompt.md
---
## Prompt

You are the Chief Engineer and Project Planner.

Update the implementation plan for the **Integrated RAG Chatbot** to reflect the user's latest requirement:
All ingestion logic (scrape live book pages from URLs, extract text, chunk, embed, create Qdrant collection, upsert with metadata) must be in **ONE single file: backend/main.py**

The live book is deployed at: https://mfsrajput.github.io/AI-And-Robotic-Hackathoon/

Generate these files:

1. **plan.md**
   Updated phased roadmap (24–48 hour hackathon timeline):
   • Phase 1: Single-File Ingestion Pipeline (main.py with all functions + execution)
   • Phase 2: Backend RAG Query Endpoint (extend main.py if needed)
   • Phase 3: Frontend Integration
   • Phase 4: Testing & Deployment
   Table: Phase | Estimated Time | Deliverables | Status

2. **tasks.md**
   Detailed prioritized tasks focusing on single-file design:
   - Implement get_all_urls() → hardcode or scrape list of chapter URLs from the live site
   - Implement extract_text_from_url(url) → use requests + BeautifulSoup
   - Implement chunk_text(text) → ~800 tokens with overlap
   - Implement embed(texts) → OpenAI text-embedding-3-large
   - Implement create_collection() → dimension 3072, COSINE
   - Implement save_chunks_to_qdrant(chunks with metadata: url, title)
   - Add main() function that runs full pipeline when executed
   - Test locally with Qdrant Cloud credentials

3. **architecture.md**
   - Mermaid diagram: Live GitHub Pages URLs → main.py scrape → chunks → OpenAI embeddings → Qdrant Cloud
   - Single-file system design explanation
   - Metadata stored: page_url, page_title, chunk_index

4. **backend/main.py**
   Create the complete, runnable single file with:
   - All required imports (requests, bs4, openai, qdrant_client, tiktoken, os, dotenv)
   - Functions: get_all_urls(), extract_text_from_url(), chunk_text(), embed(), create_collection(), save_chunks_to_qdrant()
   - Collection name: "rag_embedding"
   - Embedding model: "text-embedding-3-large" (dimension 3072)
   - Hardcode base URL and list major chapter URLs (from site analysis: intro + 4 modules)
   - Load env vars: OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY
   - if __name__ == "__main__": run full pipeline

5. **history/prompts/002-rag-chatbot/0203-rag-chatbot-plan-updated.prompt.md** → store this exact prompt

After creating all files, commit with message:
"chore: update RAG plan for single-file URL-based ingestion pipeline in main.py"

Output ONLY the contents of the five files listed in files_yaml.
No additional text.

Execute /sp.plan