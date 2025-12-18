---
id: 202
title: Integrated RAG Chatbot with Selected-Text Support
stage: feature-spec
date_iso: 2025-12-17
surface: agent
model: claude-sonnet-4-5-20251201
feature: 002-rag-chatbot
branch: main
user: mfsrajput
command: /sp.specify
labels: ["rag","chatbot","selected-text","openai","fastapi","qdrant","constitution-compliant"]
files_yaml:
  - specs/002-rag-chatbot/specification.md
  - specs/002-rag-chatbot/user-stories.md
  - specs/002-rag-chatbot/tech-decisions.md
  - history/prompts/002-rag-chatbot/0202-rag-chatbot-specify.prompt.md
---
## Prompt

You are the Product Owner and Lead Architect.

Create a complete, enforceable feature specification for the **Integrated RAG Chatbot** — the second core hackathon deliverable.

This feature MUST fully adhere to the newly ratified RAG Chatbot Constitution (accuracy first, source fidelity, selected-text precision, privacy, etc.).

Generate these files:

1. **specification.md**
   - Feature Overview
   - Core Functional Requirements (numbered)
   - Non-Functional Requirements (performance, privacy, reliability)
   - Success Criteria for Hackathon Judging (what makes it "complete")

2. **user-stories.md**
   Write as numbered user stories:
   - As a student, I can open a floating chatbot and ask questions about any part of the book so I get accurate explanations.
   - As a student, I can highlight text in a chapter and click "Ask about this" to get an explanation based only on that selection + relevant context.
   - As a reader, I see inline source citations (chapter links) in every answer.
   - As a maintainer, I can run a script to re-ingest updated book content into the vector database.

3. **tech-decisions.md**
   Document these locked decisions with brief rationale:
   - Backend: FastAPI (async, modern, easy deployment)
   - Vector DB: Qdrant Cloud Free Tier (managed, generous limits)
   - Embeddings: OpenAI text-embedding-3-large (best semantic accuracy)
   - LLM: gpt-4o-mini primary, fallback to gpt-4o (cost + speed balance)
   - Chunking: 800 tokens with 200 overlap (balances context and precision)
   - Frontend: Custom React components in Docusaurus (static GitHub Pages compatible)
   - Selected-text handling: Client-side text selection detection + send to backend
   - Deployment: Backend on Render.com or Fly.io free tier

Also create:
- `history/prompts/002-rag-chatbot/0202-rag-chatbot-specify.prompt.md` → store this exact prompt for full provenance

After creating all files, commit with message:
"docs: add full specification for RAG chatbot – user stories, requirements, and tech decisions"

Output ONLY the contents of the files listed in files_yaml.
No additional text.

Execute /sp.specify now.