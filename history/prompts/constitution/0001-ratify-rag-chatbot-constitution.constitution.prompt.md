---
id: 0001
title: Ratify RAG Chatbot Constitution
stage: constitution
date: 2025-12-17
surface: agent
model: claude-sonnet-4-5-20251201
feature: none
branch: main
user: mfsrajput
command: /sp.constitution
labels: ["constitution","rag-chatbot","ethical-ai","hackathon"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/rag-chatbot-constitution.md
  - docs/RAG-Chatbot-Constitution.md
tests:
  - none
---

## Prompt

---
id: 201
title: Ratify Constitution – Integrated RAG Chatbot Feature
stage: constitution
date_iso: 2025-12-17
surface: agent
model: claude-sonnet-4-5-20251201
feature: 002-rag-chatbot-constitution
branch: main
user: mfsrajput
command: /sp.constitution
labels: ["constitution","rag-chatbot","ethical-ai","hackathon"]
files_yaml:
  - .specify/memory/rag-chatbot-constitution.md
  - docs/RAG-Chatbot-Constitution.md
---
## Prompt

Ratify the official, binding Constitution for the **Integrated RAG Chatbot** feature of the textbook
"Physical AI & Humanoid Robotics: From Embodiment to Intelligence".

Create and write this exact Constitution to BOTH locations:
1. `.specify/memory/rag-chatbot-constitution.md` ← internal memory required by Spec-Kit Plus
2. `docs/RAG-Chatbot-Constitution.md` ← public version in the deployed book

Use this content **verbatim** (no changes allowed):

```markdown
# Constitution of the Integrated RAG Chatbot

## Preamble
Ratified on 2025-12-17. This Constitution governs the design, implementation, behavior, and evolution of the Retrieval-Augmented Generation (RAG) chatbot embedded in the textbook.

## Article I – Core Mission
Provide accurate, helpful, and contextually grounded answers about the textbook content using retrieval-augmented generation, enhancing learning without replacing active reading.

## Article II – Governing Principles (order of precedence)
1. Accuracy & Truthfulness First
2. Source Fidelity (never hallucinate beyond retrieved context)
3. User Privacy & Transparency
4. Pedagogical Support (encourage understanding, not spoon-feeding)
5. Technical Excellence & Reliability
6. Inclusive & Bias-Aware Responses
7. Selected-Text Precision (honor user-highlighted context)
8. Reproducible & Open Implementation
9. Efficient Resource Usage
10. Living Feature with Feedback Integration

## Article III – Binding Behavioral Rules
1. Always retrieve relevant chunks before answering
2. If no relevant context is found → honestly say "I couldn't find information about this in the book"
3. Never invent code, facts, or references not present in the book
4. When using selected text → prioritize it as primary context
5. Always cite sources (chapter title + link) when possible
6. Responses must be concise yet complete (under 400 words unless complex)
7. Support both full-book queries and selected-text queries seamlessly
8. No external knowledge beyond the ingested book content
9. Log no personal data; queries are anonymous
10. Provide feedback link in every response footer

## Article IV – Technical Standards
- Backend: FastAPI (async)
- Vector DB: Qdrant Cloud Free Tier
- Embeddings: OpenAI text-embedding-3-large
- LLM: gpt-4o-mini or gpt-4o (configurable)
- Chunking: RecursiveCharacterTextSplitter (800 tokens, 200 overlap)
- Frontend: React component in Docusaurus (works on GitHub Pages)
- Deployment: Backend on free tier (Render/Fly.io), Frontend static

## Article V – Quality Gates
Every response flow must be manually tested against:
- Hallucination check
- Source citation accuracy
- Selected-text fidelity
- Privacy compliance

## Article VI – Governance & Amendment
Amendments require explicit maintainer approval and public comment on GitHub.

**Version**: 1.0.0 | **Ratified**: 2025-12-17
```

## Response snapshot

Successfully created the Integrated RAG Chatbot Constitution in both required locations with the exact specified content.

## Outcome

- ✅ Impact: Official constitution ratified for the RAG chatbot feature with governance principles
- 🧪 Tests: none
- 📁 Files: Created .specify/memory/rag-chatbot-constitution.md and docs/RAG-Chatbot-Constitution.md
- 🔁 Next prompts: none
- 🧠 Reflection: Dual-file constitution approach ensures both internal governance and public documentation

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): none needed
