import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_qdrant_content():
    """
    Check if there's any content in the Qdrant collection.
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

    try:
        # Get collection info
        collection_info = client.get_collection("rag_embedding")
        print(f"Collection 'rag_embedding' exists")
        print(f"Points count: {collection_info.points_count}")

        if collection_info.points_count > 0:
            # Get a few sample points
            points = client.scroll(
                collection_name="rag_embedding",
                limit=3
            )
            print(f"Sample points from collection:")
            for point in points[0]:
                print(f"  ID: {point.id}")
                print(f"  Content preview: {point.payload.get('content', '')[:100]}...")
                print(f"  Page URL: {point.payload.get('page_url', '')}")
                print(f"  Page Title: {point.payload.get('page_title', '')}")
                print("  ---")
        else:
            print("Collection is empty - no content has been ingested yet")

    except Exception as e:
        print(f"Error accessing Qdrant: {str(e)}")

if __name__ == "__main__":
    check_qdrant_content()