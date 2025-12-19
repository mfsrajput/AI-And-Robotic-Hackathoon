import os
import requests
from bs4 import BeautifulSoup
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
import tiktoken
from dotenv import load_dotenv
import time
import re
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging

# Load environment variables
load_dotenv()

# Initialize Cohere client
cohere_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

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

def embed(texts, input_type="search_document"):
    """
    Generate embeddings for a list of texts using Cohere embed-english-v3.0.
    """
    if not texts:
        return []

    try:
        response = cohere_client.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type=input_type
        )
        return response.embeddings
    except cohere.CohereError as e:
        logging.error(f"Cohere error generating embeddings: {str(e)}")
        print(f"Cohere error generating embeddings: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Error generating embeddings: {str(e)}")
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
            size=1024,  # embed-english-v3.0 dimension
            distance=models.Distance.COSINE
        )
    )

    print(f"Created collection {collection_name} with dimension 1024")
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
        embeddings = embed([chunk], "search_document")
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

# Data models for API
class QueryRequest(BaseModel):
    query: str
    selected_text: Optional[str] = None

class Source(BaseModel):
    title: str
    url: str
    content: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    sources: List[Source]


def retrieve_relevant_chunks(query: str, top_k: int = 5):
    """
    Retrieve relevant chunks from Qdrant based on the query.
    """
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    collection_name = "rag_embedding"

    # Generate embedding for the query
    query_embedding = embed([query], "search_query")
    if not query_embedding or len(query_embedding) == 0:
        return []

    # Search for similar chunks in Qdrant
    try:
        search_response = client.query_points(
            collection_name=collection_name,
            query=query_embedding[0],
            limit=top_k,
            with_payload=True
        )

        # Extract the points from the response
        # query_points returns a QueryResponse object, and we need the points from it
        return search_response.points if hasattr(search_response, 'points') else []
    except Exception as e:
        logging.error(f"Error retrieving chunks from Qdrant: {str(e)}")
        print(f"Error retrieving chunks from Qdrant: {str(e)}")
        return []


def generate_response_with_rag(query: str, context_chunks: list, selected_text: Optional[str] = None):
    """
    Generate a response using Cohere LLM with retrieved context.
    """
    if not context_chunks:
        # If no context found, generate a response indicating so
        return "I couldn't find relevant information in the textbook to answer your question. Please try rephrasing your question or ask about a different topic from the Physical AI & Humanoid Robotics textbook.", []

    # Construct context from retrieved chunks
    context_text = "\n\n".join([chunk.payload["content"] for chunk in context_chunks])

    # Extract unique sources
    sources = []
    seen_urls = set()
    for chunk in context_chunks:
        url = chunk.payload["page_url"]
        title = chunk.payload["page_title"]
        if url not in seen_urls:
            sources.append(Source(title=title, url=url))
            seen_urls.add(url)

    # Create a prompt for the LLM
    if selected_text:
        # If selected text is provided, use it as context for explanation
        prompt = f"""
        You are an AI assistant for the Physical AI & Humanoid Robotics textbook.
        The user has selected specific text and asked for an explanation.
        Explain the selected text based on the provided context from the textbook.

        Selected text to explain: "{selected_text}"

        Additional context from textbook:
        {context_text}

        Question: {query}

        Provide a clear explanation of the selected text, using the context to enhance understanding.
        If the context doesn't contain enough information to fully explain the selected text,
        explain what you can and mention what additional information might be needed.
        """
    else:
        # Standard query processing
        prompt = f"""
        You are an AI assistant for the Physical AI & Humanoid Robotics textbook.
        Answer the user's question based on the provided context from the textbook.
        If the context doesn't contain enough information to answer the question,
        say so and suggest the user try rephrasing their question.

        Context:
        {context_text}

        Question: {query}

        Answer:
        """

    try:
        response = cohere_client.chat(
            model="command-r-08-2024",
            message=prompt,
            preamble="You are an AI assistant for the Physical AI & Humanoid Robotics textbook. Provide helpful, accurate answers based on the textbook content. Always be honest about the limitations of your knowledge if the provided context doesn't contain the answer. Cite sources when possible. Your responses must strictly follow the RAG Chatbot Constitution: prioritize accuracy, avoid hallucinations, and always cite sources with page_url when possible.",
            chat_history=[]
        )

        return response.text.strip(), sources
    except cohere.CohereError as e:
        logging.error(f"Cohere error generating response: {str(e)}")
        print(f"Cohere error generating response: {str(e)}")
        return "Sorry, I encountered an error with the AI service. Please try again later.", []
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        print(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error while generating the response. Please try again.", []


def refresh_content():
    """
    Refresh the vector database with updated content by clearing the existing collection
    and re-ingesting all content from the live site.
    """
    print("Starting content refresh process...")

    # Get the Qdrant client
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    collection_name = "rag_embedding"

    # Delete the existing collection to start fresh
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception as e:
        print(f"Collection {collection_name} may not have existed: {str(e)}")

    # Run the main ingestion pipeline to re-ingest all content
    main_ingestion()


def main_ingestion():
    """
    Main ingestion function that can be called independently for re-ingestion.
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

    return saved_count


def main():
    """
    Main function that can handle command-line arguments to either run the initial
    ingestion or refresh the content.
    """
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "refresh":
            refresh_content()
        else:
            print(f"Usage: python main.py [refresh]")
            print("  refresh: Refresh the vector database with updated content")
            print("  (no args): Run the initial ingestion pipeline")
    else:
        main_ingestion()


# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Print all registered routes at startup
print("Registered routes:")
for route in app.routes:
    print(route.path, route.methods)

# Add a refresh endpoint to the API
@app.post("/refresh")
def refresh_endpoint():
    """
    Endpoint to trigger content refresh/re-ingestion.
    """
    try:
        refresh_content()
        return {"message": "Content refresh completed successfully"}
    except Exception as e:
        print(f"Error during content refresh: {str(e)}")
        raise HTTPException(status_code=500, detail="Error refreshing content")

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "https://mfsrajput.github.io",
        "https://mfsrajput.github.io/AI-And-Robotic-Hackathoon",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Critical — allows POST, OPTIONS
    allow_headers=["*"],
)
# Full CORS for frontend POST requests – fixes 405 pre-flight


@app.get("/")
def read_root():
    """
    Health check endpoint. The main query endpoints are POST /query and POST /chat.
    """
    return {
        "message": "RAG Chatbot API is running!",
        "endpoints": {
            "query": "POST /query - Main endpoint for chat queries (streaming)",
            "chat": "POST /chat - Compatibility endpoint for chat queries (streaming)",
            "refresh": "POST /refresh - Refresh content in vector database",
            "health": "GET / - This health check endpoint"
        }
    }


def query_stream_generator(request: QueryRequest):
    """
    Generator function to stream the query response.
    """
    try:
        print("Query streaming started")
        # Use selected text as context if provided, otherwise use the query
        search_query = request.selected_text if request.selected_text else request.query

        # Retrieve relevant chunks from vector database
        relevant_chunks = retrieve_relevant_chunks(search_query)

        # Generate response using RAG with streaming
        if not relevant_chunks:
            yield f"data: I couldn't find relevant information in the textbook to answer your question. Please try rephrasing your question or ask about a different topic from the Physical AI & Humanoid Robotics textbook.\n\n"
            return

        # Construct context from retrieved chunks
        context_text = "\n\n".join([chunk.payload["content"] for chunk in relevant_chunks])

        # Extract unique sources
        sources = []
        seen_urls = set()
        for chunk in relevant_chunks:
            url = chunk.payload["page_url"]
            title = chunk.payload["page_title"]
            if url not in seen_urls:
                sources.append({"title": title, "url": url})
                seen_urls.add(url)

        # Create a prompt for the LLM
        if request.selected_text:
            # If selected text is provided, use it as context for explanation
            prompt = f"""
            You are an AI assistant for the Physical AI & Humanoid Robotics textbook.
            The user has selected specific text and asked for an explanation.
            Explain the selected text based on the provided context from the textbook.

            Selected text to explain: "{request.selected_text}"

            Additional context from textbook:
            {context_text}

            Question: {request.query}

            Provide a clear explanation of the selected text, using the context to enhance understanding.
            If the context doesn't contain enough information to fully explain the selected text,
            explain what you can and mention what additional information might be needed.
            """
        else:
            # Standard query processing
            prompt = f"""
            You are an AI assistant for the Physical AI & Humanoid Robotics textbook.
            Answer the user's question based on the provided context from the textbook.
            If the context doesn't contain enough information to answer the question,
            say so and suggest the user try rephrasing their question.

            Context:
            {context_text}

            Question: {request.query}

            Answer:
            """

        # Stream the response from Cohere
        response = cohere_client.chat_stream(
            model="command-r-08-2024",
            message=prompt,
            preamble="You are an AI assistant for the Physical AI & Humanoid Robotics textbook. Provide helpful, accurate answers based on the textbook content. Always be honest about the limitations of your knowledge if the provided context doesn't contain the answer. Cite sources when possible. Your responses must strictly follow the RAG Chatbot Constitution: prioritize accuracy, avoid hallucinations, and always cite sources with page_url when possible.",
            chat_history=[]
        )

        full_response = ""
        for event in response:
            if event.event_type == "text-generation":
                chunk = event.text
                full_response += chunk
                # Send chunk as plain text Server-Sent Events
                yield f"data: {chunk}\n\n"
                print(f"Query streaming chunk: {chunk}")

        # Send sources at the end
        if sources:
            yield f"data: \n\nSources:\n"
            for source in sources:
                yield f"data: - {source['title']} ({source['url']})\n"

    except cohere.CohereError as e:
        logging.error(f"Cohere error generating query response: {str(e)}")
        print(f"Cohere error generating query response: {str(e)}")
        yield f"data: Sorry, I encountered an error with the AI service. Please try again later.\n\n"
    except Exception as e:
        logging.error(f"Error generating query response: {str(e)}")
        print(f"Error generating query response: {str(e)}")
        yield f"data: Sorry, I encountered an error while generating the response. Please try again.\n\n"


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    Streaming endpoint to handle user queries and return RAG-enhanced responses.
    """
    return StreamingResponse(query_stream_generator(request), media_type="text/event-stream")


def chat_stream_generator(request: QueryRequest):
    """
    Generator function to stream the chat response.
    """
    try:
        print("Streaming started")
        # Use selected text as context if provided, otherwise use the query
        search_query = request.selected_text if request.selected_text else request.query

        # Retrieve relevant chunks from vector database
        relevant_chunks = retrieve_relevant_chunks(search_query)

        # Generate response using RAG with streaming
        if not relevant_chunks:
            yield f"data: I couldn't find relevant information in the textbook to answer your question. Please try rephrasing your question or ask about a different topic from the Physical AI & Humanoid Robotics textbook.\n\n"
            return

        # Construct context from retrieved chunks
        context_text = "\n\n".join([chunk.payload["content"] for chunk in relevant_chunks])

        # Extract unique sources
        sources = []
        seen_urls = set()
        for chunk in relevant_chunks:
            url = chunk.payload["page_url"]
            title = chunk.payload["page_title"]
            if url not in seen_urls:
                sources.append({"title": title, "url": url})
                seen_urls.add(url)

        # Create a prompt for the LLM
        if request.selected_text:
            # If selected text is provided, use it as context for explanation
            prompt = f"""
            You are an AI assistant for the Physical AI & Humanoid Robotics textbook.
            The user has selected specific text and asked for an explanation.
            Explain the selected text based on the provided context from the textbook.

            Selected text to explain: "{request.selected_text}"

            Additional context from textbook:
            {context_text}

            Question: {request.query}

            Provide a clear explanation of the selected text, using the context to enhance understanding.
            If the context doesn't contain enough information to fully explain the selected text,
            explain what you can and mention what additional information might be needed.
            """
        else:
            # Standard query processing
            prompt = f"""
            You are an AI assistant for the Physical AI & Humanoid Robotics textbook.
            Answer the user's question based on the provided context from the textbook.
            If the context doesn't contain enough information to answer the question,
            say so and suggest the user try rephrasing their question.

            Context:
            {context_text}

            Question: {request.query}

            Answer:
            """

        # Stream the response from Cohere
        response = cohere_client.chat_stream(
            model="command-r-08-2024",
            message=prompt,
            preamble="You are an AI assistant for the Physical AI & Humanoid Robotics textbook. Provide helpful, accurate answers based on the textbook content. Always be honest about the limitations of your knowledge if the provided context doesn't contain the answer. Cite sources when possible. Your responses must strictly follow the RAG Chatbot Constitution: prioritize accuracy, avoid hallucinations, and always cite sources with page_url when possible.",
            chat_history=[]
        )

        full_response = ""
        for event in response:
            if event.event_type == "text-generation":
                chunk = event.text
                full_response += chunk
                # Send chunk as plain text Server-Sent Events
                yield f"data: {chunk}\n\n"
                print(f"Streaming chunk: {chunk}")

        # Sources are kept internally for accuracy but not sent to frontend

    except cohere.CohereError as e:
        logging.error(f"Cohere error generating response: {str(e)}")
        print(f"Cohere error generating response: {str(e)}")
        yield f"data: Sorry, I encountered an error with the AI service. Please try again later.\n\n"
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        print(f"Error generating response: {str(e)}")
        yield f"data: Sorry, I encountered an error while generating the response. Please try again.\n\n"


@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    """
    Streaming endpoint for chat queries with proper Server-Sent Events.
    This endpoint exists to support frontend configurations that use /chat.
    # POST /chat — verified in startup log
    """
    return StreamingResponse(chat_stream_generator(request), media_type="text/event-stream")

@app.options("/chat")
def chat_options():
    """
    Handle pre-flight requests for /chat endpoint.
    """
    return {"message": "OK"}


if __name__ == "__main__":
    main()