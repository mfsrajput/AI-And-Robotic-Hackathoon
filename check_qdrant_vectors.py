import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_qdrant_vectors():
    """
    Check if the Qdrant collection has vectors stored.
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

        # Get one sample point to check if it has vectors
        if collection_info.points_count > 0:
            points = client.scroll(
                collection_name="rag_embedding",
                limit=1
            )
            if points[0]:
                sample_point = points[0][0]
                print(f"Sample point ID: {sample_point.id}")
                print(f"Has vector: {sample_point.vector is not None}")
                if sample_point.vector:
                    print(f"Vector length: {len(sample_point.vector) if isinstance(sample_point.vector, list) else 'N/A'}")
                print(f"Payload keys: {list(sample_point.payload.keys())}")
            else:
                print("No points returned from scroll")
        else:
            print("Collection is empty")

    except Exception as e:
        print(f"Error accessing Qdrant: {str(e)}")

if __name__ == "__main__":
    check_qdrant_vectors()