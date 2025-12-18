# DEPRECATED: Module 4 - Vision-Language-Action (VLA) + Capstone

**STATUS: ARCHIVED** This directory contains deprecated specification artifacts.

## Project Pivot Notice

This directory originally contained artifacts for the Vision-Language-Action (VLA) + Capstone module. However, the project direction was pivoted to implement the Integrated RAG Chatbot feature instead, which better aligned with immediate needs for an interactive textbook experience.

## Current Implementation

The actual implemented feature is the **Integrated RAG Chatbot** with:
- Floating chat interface integrated with the textbook
- Selected-text query functionality ("Ask about this")
- Source citations for all responses
- Cohere-powered RAG backend with Qdrant vector storage
- Privacy-compliant design with no user data logging

## New Artifacts Location

The properly aligned specification artifacts for the implemented RAG Chatbot feature are located at:
- `/specs/002-rag-chatbot/spec.md`
- `/specs/002-rag-chatbot/plan.md`
- `/specs/002-rag-chatbot/tasks.md`
- `/specs/002-rag-chatbot/decisions.md`

## Decision Summary

The project team decided to pivot from the VLA module to implement the Integrated RAG Chatbot feature to provide immediate value to textbook users through an interactive Q&A system. The RAG Chatbot better serves the core mission of enhancing the educational experience with AI-powered assistance.

All future development and reference should use the artifacts in `/specs/002-rag-chatbot/`.