# Integrated RAG Chatbot Specification

## Feature Overview

The Integrated RAG Chatbot is an intelligent question-answering system embedded within the "Physical AI & Humanoid Robotics" textbook website. It leverages Retrieval-Augmented Generation (RAG) to provide accurate, contextually relevant answers based on the textbook content. The system prioritizes accuracy, source fidelity, and pedagogical support while maintaining user privacy. It supports both general book queries and selected-text queries, enabling students to get explanations based on specific content they're reading.

## Core Features

1. **Floating Chat Interface**: Provides a floating chat interface that remains accessible on all textbook pages without interfering with reading experience.

2. **General Book Queries**: Accepts natural language questions about the textbook content and provides accurate answers based solely on the book's information, with proper source citations.

3. **Selected-Text Queries**: Detects text selection on any page and provides an "Ask about this" button that generates context-aware responses based on the selected text plus relevant surrounding context.

4. **Source Citations**: Provides inline source citations (chapter titles with links) in every response, allowing users to verify information and explore original content.

5. **Content Freshness**: Provides a mechanism for maintainers to re-ingest updated book content into the vector database when the textbook is updated.

6. **Constitution Compliance**: Strictly adheres to the RAG Chatbot Constitution, never hallucinating information, always citing sources, and maintaining privacy.

## Technical Implementation

- **Backend**: FastAPI with Cohere integration
- **Vector Database**: Qdrant for embedding storage and retrieval
- **Embeddings**: Cohere embed-english-v3.0 (1024 dimensions)
- **Generation**: Cohere command-r-08-2024
- **Frontend**: React components integrated with Docusaurus
- **Deployment**: Backend on Render.com, frontend on GitHub Pages

## Artifacts

- `spec.md`: Feature specification and requirements
- `plan.md`: Implementation plan and architecture
- `tasks.md`: Implementation tasks and progress tracking
- `decisions.md`: Architectural decisions and rationale

## Success Criteria

1. **Constitution Compliance**: The chatbot fully adheres to the ratified RAG Chatbot Constitution (accuracy, source fidelity, privacy, etc.)

2. **Selected-Text Functionality**: Text selection detection and "Ask about this" feature works seamlessly across all textbook pages

3. **Response Quality**: Answers are accurate, concise (under 400 words), and always include proper source citations

4. **User Experience**: The floating chat interface is intuitive, unobtrusive, and enhances rather than disrupts the reading experience

5. **Technical Implementation**: The solution demonstrates proper use of RAG architecture with FastAPI backend, Qdrant vector database, and Cohere integration

6. **Deployment**: The system is successfully deployed and accessible on the live textbook website