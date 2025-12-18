# Integrated RAG Chatbot Specification

## Feature Overview

The Integrated RAG Chatbot is an intelligent question-answering system embedded within the "Physical AI & Humanoid Robotics" textbook website. It leverages Retrieval-Augmented Generation (RAG) to provide accurate, contextually relevant answers based on the textbook content. The system prioritizes accuracy, source fidelity, and pedagogical support while maintaining user privacy. It supports both general book queries and selected-text queries, enabling students to get explanations based on specific content they're reading.

## Core Functional Requirements

1. **Floating Chat Interface**: The system shall provide a floating chat interface that remains accessible on all textbook pages without interfering with reading experience.

2. **General Book Queries**: The system shall accept natural language questions about the textbook content and provide accurate answers based solely on the book's information, with proper source citations.

3. **Selected-Text Queries**: The system shall detect text selection on any page and provide an "Ask about this" button that generates context-aware responses based on the selected text plus relevant surrounding context.

4. **Source Citations**: The system shall provide inline source citations (chapter titles with links) in every response, allowing users to verify information and explore original content.

5. **Content Freshness**: The system shall provide a mechanism for maintainers to re-ingest updated book content into the vector database when the textbook is updated.

6. **Constitution Compliance**: The system shall strictly adhere to the RAG Chatbot Constitution, never hallucinating information, always citing sources, and maintaining privacy.

## Non-Functional Requirements

1. **Performance**: The system shall respond to simple queries within 2 seconds and complex queries (requiring multiple context chunks) within 4 seconds under normal load conditions.

2. **Privacy**: The system shall log no personal data; all queries shall be processed anonymously without storing user information.

3. **Reliability**: The system shall maintain 99% uptime during normal operating hours.

4. **Accuracy**: The system shall provide factually accurate responses based only on retrieved context, with less than 1% hallucination rate.

5. **Scalability**: The system shall support up to 100 concurrent users without performance degradation.

6. **Accessibility**: The chat interface shall be accessible to users with disabilities, following WCAG 2.1 guidelines.

7. **Security**: The system shall implement proper input sanitization to prevent injection attacks.

8. **Error Resilience**
   - The system must gracefully handle Cohere API errors (TooManyTokens, RateLimit, Authentication, network timeout)
   - Backend must return meaningful JSON error responses
   - Frontend must display user-friendly messages instead of generic "encountered an error"
   - Support retry for transient errors

9. **Cold Start & Performance**
   - Handle Render.com free tier cold starts (first request slow)
   - Frontend shows "Thinking..." with longer timeout
   - Reduce default retrieved chunks to 5 to prevent context overflow

10. **Debugging Support**
   - Backend logs detailed Cohere errors
   - Frontend console logs API errors during development

## Success Criteria for Hackathon Judging

1. **Constitution Compliance**: The chatbot fully adheres to the ratified RAG Chatbot Constitution (accuracy, source fidelity, privacy, etc.)

2. **Selected-Text Functionality**: Text selection detection and "Ask about this" feature works seamlessly across all textbook pages

3. **Response Quality**: Answers are accurate, concise (under 400 words), and always include proper source citations

4. **User Experience**: The floating chat interface is intuitive, unobtrusive, and enhances rather than disrupts the reading experience

5. **Technical Implementation**: The solution demonstrates proper use of RAG architecture with FastAPI backend, Qdrant vector database, and Cohere integration

6. **Deployment**: The system is successfully deployed and accessible on the live textbook website

## Clarifications

### Session 2025-12-17

- Q: What are the specific response time targets for different query types? → A: Define specific response time targets: 2s for simple queries, 4s for complex queries requiring multiple context chunks
- Q: How should the system handle different types of errors and failure scenarios? → A: Define specific error responses: API timeout, vector DB unavailable, LLM error, no relevant context found