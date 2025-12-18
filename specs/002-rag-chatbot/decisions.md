# Architectural Decisions: Integrated RAG Chatbot

## Decision Log

### AD-001: Cohere over OpenAI for RAG Implementation
**Date**: 2025-12-17
**Status**: Accepted

**Context**: Need to select an LLM provider for the RAG chatbot that offers good performance for textbook content while being cost-effective.

**Decision**: Use Cohere's embed-english-v3.0 for embeddings and command-r-08-2024 for generation instead of OpenAI.

**Rationale**:
- Cohere embeddings are specifically designed for retrieval tasks
- Better cost-performance ratio for educational use
- Superior handling of technical content like robotics textbooks
- Good support for context length needed for textbook queries

**Consequences**:
- Positive: Better RAG performance on technical content
- Positive: More cost-effective for educational deployment
- Negative: Need to handle model deprecation (command-r-plus was deprecated)

### AD-002: URL-based Ingestion from Live GitHub Pages Site
**Date**: 2025-12-17
**Status**: Accepted

**Context**: Need to determine how to extract content from the textbook for the RAG system.

**Decision**: Implement URL scraping from the live textbook GitHub Pages site instead of using static files.

**Rationale**:
- Ensures content is always up-to-date with latest textbook changes
- Simplifies maintenance by avoiding duplicate content storage
- Automatically captures all textbook content without manual updates
- Supports content freshness requirements

**Consequences**:
- Positive: Always current with textbook updates
- Positive: Single source of truth for content
- Negative: Dependency on live site availability during ingestion
- Negative: Need to handle 404s and network errors gracefully

### AD-003: Single-File Backend for Simplicity
**Date**: 2025-12-17
**Status**: Accepted

**Context**: Need to decide on the architecture for the backend API.

**Decision**: Implement the entire backend as a single file (backend/main.py) instead of a multi-file structure.

**Rationale**:
- Simplifies deployment and maintenance
- Reduces complexity for educational project
- Faster development and iteration
- Easier to understand and modify

**Consequences**:
- Positive: Simple to deploy and maintain
- Positive: Faster development cycle
- Negative: May become unwieldy as features grow
- Negative: Less modular than multi-file approach

### AD-004: Selected-Text Priority in Retrieval
**Date**: 2025-12-17
**Status**: Accepted

**Context**: Need to determine how to handle queries that include both general questions and selected text.

**Decision**: When selected text is provided, prioritize it as primary context in the retrieval process.

**Rationale**:
- Provides more relevant responses for the selected-text feature
- Supports the key use case of asking for explanations of specific content
- Maintains consistency with the "Ask about this" functionality
- Improves pedagogical value by focusing on user-selected content

**Consequences**:
- Positive: Better user experience for selected-text queries
- Positive: More targeted responses for specific content
- Negative: Slightly more complex retrieval logic
- Negative: Need to handle fallback when selected text not found in context

### AD-005: Render.com for Free Backend Hosting
**Date**: 2025-12-17
**Status**: Accepted

**Context**: Need to select a hosting platform for the backend API.

**Decision**: Deploy the backend to Render.com using their free tier instead of other hosting options.

**Rationale**:
- Free tier sufficient for educational project
- Easy deployment and management
- Good integration with Python applications
- Reliable uptime for educational use
- Supports the required dependencies (Python, FastAPI)

**Consequences**:
- Positive: Cost-effective solution for educational project
- Positive: Easy deployment and scaling
- Negative: Dependent on free tier limitations
- Negative: Potential for slower response times during peak usage

### AD-006: Qdrant Vector Database for Storage
**Date**: 2025-12-17
**Status**: Accepted

**Context**: Need to select a vector database for storing embeddings and enabling similarity search.

**Decision**: Use Qdrant instead of alternatives like Pinecone or Weaviate.

**Rationale**:
- Open source with self-hosting options
- Good performance for similarity search
- Easy integration with Python ecosystem
- Cost-effective for educational use
- Supports the required cosine similarity metric

**Consequences**:
- Positive: Cost-effective vector storage
- Positive: Good performance for textbook content
- Positive: Open source with no vendor lock-in
- Negative: Less managed service features than alternatives

### AD-007: Floating UI Design Pattern
**Date**: 2025-12-17
**Status**: Accepted

**Context**: Need to determine the UI approach for integrating the chatbot into the textbook site.

**Decision**: Implement a floating chat interface that appears as a button on all pages, expanding into a chat window when clicked.

**Rationale**:
- Non-intrusive design that doesn't interfere with reading
- Always accessible without navigating away from content
- Familiar pattern for users from other chat interfaces
- Maintains focus on textbook content while providing help

**Consequences**:
- Positive: Non-disruptive to reading experience
- Positive: Always accessible to users
- Positive: Familiar interaction pattern
- Negative: May require additional styling to match site theme
- Negative: Potential for z-index conflicts with other UI elements