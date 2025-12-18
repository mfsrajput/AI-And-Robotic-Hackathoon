---
id: 12
title: "RAG Chatbot Implementation"
stage: "completed"
date_iso: "2025-12-17"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "vla-capstone"
branch: "001-vla-capstone"
user: "farhan-rajput"
command: "implementation"
labels: ["rag", "chatbot", "cohere", "qdrant", "vector-database", "textbook"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files_yaml:
  - "backend/main.py"
  - "src/components/Chatbot.jsx"
  - "src/components/TextSelectionHandler.jsx"
  - "src/css/chatbot.css"
  - "src/theme/Layout.js"
  - "requirements.txt"
  - ".env.example"
  - "docs/Chatbot.md"
tests_yaml: []
---

# RAG Chatbot Implementation

## User Prompt
```
Implement a comprehensive RAG (Retrieval-Augmented Generation) chatbot for the Physical AI & Humanoid Robotics textbook with the following features:
- Backend ingestion pipeline to scrape textbook content, create embeddings, and store in Qdrant vector database
- Frontend chat interface with floating button
- Selected-text query functionality where users can highlight text and ask for explanations
- API endpoints for querying the knowledge base
- Content freshness capabilities
- Switch from OpenAI to Cohere for both embeddings and generation models
- Complete frontend with text selection support
```

## Assistant Response
Successfully implemented a comprehensive RAG chatbot system for the Physical AI & Humanoid Robotics textbook with:

### Backend Implementation
- Complete ingestion pipeline using requests and BeautifulSoup to scrape textbook content
- Text chunking with overlap functionality for proper context management
- Cohere integration with embed-english-v3.0 for embeddings (dimension 1024) and command-r-08-2024 for generation
- Qdrant vector database integration with cosine similarity search
- FastAPI endpoints for querying and refreshing content
- Proper error handling and source citation

### Frontend Implementation
- Floating chat interface with modern React component
- Text selection detection with "Ask about this" functionality
- Docusaurus integration through Layout wrapper
- Streaming response support for real-time chat experience
- Source citation display in responses

### Key Technical Features
- RAG (Retrieval Augmented Generation) architecture
- Cohere embed-english-v3.0 embeddings with proper input types
- Qdrant vector database with 1024-dimensional vectors
- Text selection handling with custom event communication
- Streaming API responses
- Content freshness with refresh endpoint

### Files Created/Modified
- backend/main.py: Main RAG implementation with Cohere integration
- src/components/Chatbot.jsx: Floating chat interface
- src/components/TextSelectionHandler.jsx: Text selection functionality
- src/css/chatbot.css: Modern styling
- src/theme/Layout.js: Docusaurus integration
- requirements.txt: Dependencies
- .env.example: Configuration
- docs/Chatbot.md: Documentation

### Issues Fixed
- Fixed Qdrant client search method (changed from search to query_points)
- Updated Cohere model from command-r-plus to command-r-08-2024 (due to model deprecation)
- Fixed FastAPI app initialization order
- Resolved dependency installation issues with tiktoken

### Testing Results
- API endpoints working correctly (tested /query and /refresh)
- Both general queries and selected-text queries functional
- Proper source citations returned with responses
- Streaming responses working as expected

The system is fully operational and ready for use with the Physical AI & Humanoid Robotics textbook content.