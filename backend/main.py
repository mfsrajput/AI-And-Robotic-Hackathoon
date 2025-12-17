import os
import requests
from bs4 import BeautifulSoup
import openai
from qdrant_client import QdrantClient
from qdrant_client.http import models
import tiktoken
from dotenv import load_dotenv
import time
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_all_urls():
    """
    Get all URLs from the live book site.
    Hardcoded list of major chapter URLs from the live site.
    """
    base_url = "https://mfsrajput.github.io/AI-And-Robotic-Hackathoon"
    urls = [
        f"{base_url}/",
        f"{base_url}/docs/intro",
        f"{base_url}/docs/preface",
        f"{base_url}/docs/contributors",
        f"{base_url}/docs/getting-started",
        f"{base_url}/docs/01-ros2/",
        f"{base_url}/docs/02-digital-twin/",
        f"{base_url}/docs/03-nvidia-isaac/",
        f"{base_url}/docs/04-vla/",
        f"{base_url}/docs/troubleshooting",
        f"{base_url}/docs/local-verification-checklist",
        # Add more specific chapter pages if needed
        f"{base_url}/docs/01-ros2/01-ros2-and-embodied-control",
        f"{base_url}/docs/01-ros2/02-nodes-topics-services-actions",
        f"{base_url}/docs/01-ros2/03-urdf-xacro-for-humanoids",
        f"{base_url}/docs/01-ros2/04-python-rclpy-bridge",
        f"{base_url}/docs/02-digital-twin/01-gazebo-physics-and-world-building",
        f"{base_url}/docs/02-digital-twin/02-simulating-sensors-lidar-imu-depth",
        f"{base_url}/docs/02-digital-twin/03-unity-for-high-fidelity-hri",
        f"{base_url}/docs/02-digital-twin/04-creating-complete-digital-twins",
        f"{base_url}/docs/03-nvidia-isaac/01-isaac-sim-synthetic-data",
        f"{base_url}/docs/03-nvidia-isaac/02-isaac-ros-vslam-perception",
        f"{base_url}/docs/03-nvidia-isaac/03-nav2-bipedal-locomotion",
        f"{base_url}/docs/03-nvidia-isaac/04-sim-to-real-transfer",
        f"{base_url}/docs/04-vla/01-voice-to-action-with-whisper",
        f"{base_url}/docs/04-vla/02-llm-task-and-motion-planning",
        f"{base_url}/docs/04-vla/03-multi-modal-integration",
        f"{base_url}/docs/04-vla/04-capstone-autonomous-humanoid"
    ]
    return urls

def extract_text_from_url(url):
    """
    Extract text content from a given URL using requests and BeautifulSoup.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get the title
        title = soup.find('title')
        page_title = title.get_text().strip() if title else "No Title"

        # Extract main content - focus on article/main content areas
        content_selectors = [
            'main', 'article', '.main-content', '.content', '.markdown',
            '.doc-content', '.docs-content', '.container', '[role="main"]'
        ]

        content_text = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    content_text += element.get_text(separator=' ', strip=True) + "\n\n"
                break

        # If no specific content area found, get all text
        if not content_text.strip():
            content_text = soup.get_text(separator=' ', strip=True)

        # Clean up the text
        content_text = re.sub(r'\s+', ' ', content_text).strip()

        return content_text, page_title
    except Exception as e:
        print(f"Error extracting text from {url}: {str(e)}")
        return "", "Error Page"

def chunk_text(text, chunk_size=800, overlap=200):
    """
    Chunk text into pieces of approximately chunk_size tokens with overlap.
    """
    if not text.strip():
        return []

    # Use tiktoken to estimate token count
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # Split text into sentences to avoid breaking in the middle
    sentences = re.split(r'[.!?]+\s+', text)

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        # Estimate token count for the sentence
        sentence_tokens = len(encoding.encode(sentence))

        # If adding this sentence would exceed chunk size
        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())

            # Apply overlap by taking the last part of the current chunk
            if overlap > 0:
                overlap_tokens = encoding.encode(current_chunk)
                if len(overlap_tokens) > overlap:
                    overlap_text = encoding.decode(overlap_tokens[-overlap:])
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
                current_tokens = len(encoding.encode(current_chunk))
            else:
                current_chunk = sentence
                current_tokens = sentence_tokens
        else:
            current_chunk += " " + sentence
            current_tokens += sentence_tokens

    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def embed(texts):
    """
    Generate embeddings for a list of texts using OpenAI text-embedding-3-large.
    """
    if not texts:
        return []

    try:
        response = openai_client.embeddings.create(
            input=texts,
            model="text-embedding-3-large"
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        return []

def create_collection():
    """
    Create a Qdrant collection for storing embeddings.
    """
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    collection_name = "rag_embedding"

    # Check if collection already exists
    try:
        client.get_collection(collection_name)
        print(f"Collection {collection_name} already exists")
        return client
    except:
        pass

    # Create new collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=3072,  # text-embedding-3-large dimension
            distance=models.Distance.COSINE
        )
    )

    print(f"Created collection {collection_name} with dimension 3072")
    return client

def save_chunks_to_qdrant(chunks_with_metadata):
    """
    Save chunks with metadata to Qdrant collection.
    """
    client = create_collection()
    collection_name = "rag_embedding"

    # Prepare points for upsert
    points = []
    for idx, (chunk, metadata) in enumerate(chunks_with_metadata):
        # Generate embedding for the chunk
        embeddings = embed([chunk])
        if embeddings and len(embeddings) > 0:
            point = models.PointStruct(
                id=idx,
                vector=embeddings[0],
                payload={
                    "content": chunk,
                    "page_url": metadata["page_url"],
                    "page_title": metadata["page_title"],
                    "chunk_index": metadata["chunk_index"]
                }
            )
            points.append(point)

    # Upsert points to collection
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Saved {len(points)} chunks to Qdrant collection {collection_name}")

    return len(points)

def main():
    """
    Main function to run the full ingestion pipeline.
    """
    print("Starting RAG chatbot ingestion pipeline...")

    # Get all URLs from the live site
    print("Fetching URLs...")
    urls = get_all_urls()
    print(f"Found {len(urls)} URLs to process")

    all_chunks_with_metadata = []

    # Process each URL
    for i, url in enumerate(urls):
        print(f"Processing ({i+1}/{len(urls)}): {url}")

        # Extract text from URL
        content, title = extract_text_from_url(url)

        if content.strip():
            # Chunk the content
            chunks = chunk_text(content)

            # Add chunks with metadata
            for chunk_idx, chunk in enumerate(chunks):
                metadata = {
                    "page_url": url,
                    "page_title": title,
                    "chunk_index": chunk_idx
                }
                all_chunks_with_metadata.append((chunk, metadata))

            print(f"  Extracted {len(chunks)} chunks from {title}")
        else:
            print(f"  No content extracted from {url}")

        # Be respectful to the server
        time.sleep(0.5)

    print(f"Total chunks to process: {len(all_chunks_with_metadata)}")

    # Save all chunks to Qdrant
    saved_count = save_chunks_to_qdrant(all_chunks_with_metadata)
    print(f"Pipeline completed. Saved {saved_count} chunks to Qdrant.")

if __name__ == "__main__":
    main()