import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
import cohere
from dotenv import load_dotenv
import time
import logging

# Load environment variables
load_dotenv()

# Initialize Cohere client
cohere_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

def embed(texts, input_type="search_document"):
    """
    Generate embeddings for a list of texts using Cohere embed-english-v3.0.
    """
    if not texts:
        return []

    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            response = cohere_client.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type=input_type
            )
            return response.embeddings
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error generating embeddings (attempt {attempt + 1}/{max_retries}): {error_msg}")
            print(f"Error generating embeddings (attempt {attempt + 1}/{max_retries}): {error_msg}")

            # Check if it's a rate limit error
            if "429" in error_msg or "TooManyRequests" in error_msg or "rate limit" in error_msg.lower():
                print(f"Rate limit hit. Waiting {retry_delay * (attempt + 1)} seconds before retry...")
                time.sleep(retry_delay * (attempt + 1))
            elif attempt == max_retries - 1:  # Last attempt
                logging.error(f"Failed to generate embeddings after {max_retries} attempts")
                print(f"Failed to generate embeddings after {max_retries} attempts")
                return []
            else:
                # Wait before retrying for other errors
                time.sleep(retry_delay)

    return []

def create_collection():
    """
    Create a Qdrant collection for storing embeddings.
    """
    # Initialize Qdrant client
    if os.getenv("QDRANT_API_KEY"):
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=10
        )
    else:
        client = QdrantClient(
            host='localhost',
            port=6333,
            timeout=10
        )

    # Create collection if it doesn't exist
    try:
        client.get_collection("rag_embedding")
        print("Collection rag_embedding already exists")
    except:
        print("Creating collection rag_embedding...")
        client.create_collection(
            collection_name="rag_embedding",
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)  # Cohere embeddings are 1024-dim
        )
        print("Collection created")

    return client

def test_ingestion():
    """
    Test ingestion with a few sample chunks to verify the system works.
    """
    print("Starting test ingestion...")

    # Sample chunks from the textbook
    sample_chunks = [
        ("Physical AI & Humanoid Robotics is an interdisciplinary field combining physics, artificial intelligence, and robotics to create humanoid robots capable of interacting with the physical world.",
         {"page_url": "http://localhost:3003/intro/", "page_title": "Introduction", "chunk_index": 0}),
        ("ROS 2 (Robot Operating System 2) provides a flexible framework for writing robot software, including tools, libraries, and conventions for building robot applications.",
         {"page_url": "http://localhost:3003/ros2/ros2-and-embodied-control/", "page_title": "ROS 2 and Embodied Control", "chunk_index": 0}),
        ("Replicas in robotics refer to multiple instances or copies of robotic systems used for testing, validation, or parallel execution of tasks.",
         {"page_url": "http://localhost:3003/vla/llm-task-and-motion-planning/", "page_title": "LLM Task & Motion Planning", "chunk_index": 1})
    ]

    client = create_collection()
    collection_name = "rag_embedding"

    # Prepare points for upsert
    points = []
    for idx, (chunk, metadata) in enumerate(sample_chunks):
        print(f"Processing sample chunk {idx + 1}/3...")

        # Generate embedding for the chunk
        embeddings = embed([chunk], "search_document")
        if embeddings and len(embeddings) > 0:
            point = models.PointStruct(
                id=idx,  # Use a unique ID
                vector=embeddings[0],
                payload={
                    "content": chunk,
                    "page_url": metadata["page_url"],
                    "page_title": metadata["page_title"],
                    "chunk_index": metadata["chunk_index"]
                }
            )
            points.append(point)
            print(f"  Generated embedding for: {chunk[:50]}...")
        else:
            print(f"  Failed to generate embedding for chunk {idx + 1}")

        # Add delay to prevent rate limiting
        time.sleep(0.6)  # 600ms delay between chunks to stay under rate limit

    # Upsert points to collection
    if points:
        try:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"Successfully saved {len(points)} sample chunks to Qdrant collection {collection_name}")
        except Exception as e:
            print(f"Error saving to Qdrant: {str(e)}")
    else:
        print("No points to save")

if __name__ == "__main__":
    test_ingestion()