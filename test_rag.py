import pytest
import asyncio
from backend.main import get_all_urls, extract_text_from_url, chunk_text, embed, retrieve_relevant_chunks
import os
from dotenv import load_dotenv

load_dotenv()

def test_get_all_urls():
    """Test that URLs are retrieved correctly"""
    urls = get_all_urls()
    assert len(urls) > 0
    assert "https://mfsrajput.github.io/AI-And-Robotic-Hackathoon/" in urls

def test_extract_text_from_url():
    """Test text extraction from a sample URL"""
    # Test with the main page
    url = "https://mfsrajput.github.io/AI-And-Robotic-Hackathoon/"
    text, title = extract_text_from_url(url)

    # Should have some content and a title
    assert len(text) > 0
    assert len(title) > 0

def test_chunk_text():
    """Test text chunking functionality"""
    sample_text = "This is a sample text. " * 100  # Create a longer text
    chunks = chunk_text(sample_text, chunk_size=100, overlap=20)

    assert len(chunks) > 0
    assert all(len(chunk) > 0 for chunk in chunks)

def test_embed():
    """Test embedding functionality"""
    sample_texts = ["This is a test sentence.", "Another test sentence."]
    embeddings = embed(sample_texts)

    assert len(embeddings) == len(sample_texts)
    assert all(len(embedding) > 0 for embedding in embeddings)

# Note: The following tests require valid API keys and may take time to run
# They are marked as integration tests

@pytest.mark.integration
def test_retrieve_relevant_chunks():
    """Test RAG retrieval functionality"""
    if not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY"):
        pytest.skip("Qdrant credentials not set")

    query = "What is ROS 2?"
    results = retrieve_relevant_chunks(query)

    # Should return a list of results
    assert isinstance(results, list)

if __name__ == "__main__":
    pytest.main([__file__])