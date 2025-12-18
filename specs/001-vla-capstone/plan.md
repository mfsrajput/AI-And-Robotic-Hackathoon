# DEPRECATED: Module 4 - Vision-Language-Action (VLA) + Capstone

**STATUS: ARCHIVED** This plan has been superseded by the Integrated RAG Chatbot implementation.

**Original Date**: 2025-12-10 | **Superseded by**: [specs/002-rag-chatbot/](specs/002-rag-chatbot/)

## Summary

This plan document for the Vision-Language-Action (VLA) + Capstone module has been archived. The project direction was pivoted to implement the Integrated RAG Chatbot feature instead, which better aligned with the immediate needs for an interactive textbook experience.

The RAG Chatbot feature (see specs/002-rag-chatbot/) provides:
- Floating chat interface integrated with the textbook
- Selected-text query functionality ("Ask about this")
- Source citations for all responses
- Cohere-powered RAG backend with Qdrant vector storage
- Privacy-compliant design with no user data logging

## Original Technical Context (Deprecated)

**Language/Version**: Python 3.11, Markdown, YAML, C++ (ROS 2 Humble/Jazzy compatible)
**Primary Dependencies**: ROS 2 Humble/Jazzy, Whisper.cpp, Ollama/Llama, PyAudio, NumPy, OpenCV, Pandoc (for citations)
**Storage**: File-based (YAML configs, Markdown docs, Python scripts), no database required
**Testing**: pytest for Python scripts, colcon test for ROS 2 packages, manual verification for educational content
**Target Platform**: Linux (Ubuntu 22.04 LTS for ROS 2 Humble), compatible with ROS 2 Humble and Jazzy
**Project Type**: Educational textbook module with runnable code examples
**Performance Goals**: <2 seconds for voice-to-text conversion, <5 seconds for LLM response, <10 seconds for action sequence generation
**Constraints**: <500MB memory for LLM operations, offline-capable (no external API calls), WCAG 2.1 AA compliant
**Scale/Scope**: 4 textbook chapters, ~20,000 words, 50+ runnable code examples, 20+ Mermaid diagrams

## Project Structure (Deprecated)

This structure was planned but not implemented. The actual implementation uses the RAG Chatbot structure defined in specs/002-rag-chatbot/plan.md.

## Decision

The project team decided to pivot from the VLA module to implement the Integrated RAG Chatbot feature to provide immediate value to textbook users through an interactive Q&A system. The RAG Chatbot better serves the core mission of enhancing the educational experience with AI-powered assistance.
