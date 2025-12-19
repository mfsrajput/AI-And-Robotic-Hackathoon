import requests
import json

# Test the chatbot endpoint with different queries
def test_chatbot():
    url = "http://localhost:8000/chat"

    # Test queries
    test_queries = [
        {"query": "What is ROS 2?", "description": "Ask about ROS 2"},
        {"query": "Explain humanoid robotics", "description": "Ask about humanoid robotics"},
        {"query": "What is the Physical AI & Humanoid Robotics textbook about?", "description": "Ask about the textbook itself"},
        {"query": "What is embodied control?", "description": "Ask about embodied control"},
    ]

    headers = {
        "Content-Type": "application/json"
    }

    for i, query_data in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {query_data['description']} ---")
        payload = {
            "query": query_data["query"],
            "selected_text": None
        }

        try:
            print(f"Sending query: {query_data['query']}")
            response = requests.post(url, data=json.dumps(payload), headers=headers)

            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                # Handle streaming response
                response_text = response.text
                # Remove "data: " prefixes and join the content
                lines = response_text.split('\n')
                actual_response = ""
                for line in lines:
                    if line.startswith('data: '):
                        content = line[6:].strip()  # Remove "data: " prefix
                        if content and content != '[DONE]':
                            actual_response += content + " "

                print(f"Response: {actual_response.strip()}")
            else:
                print(f"ERROR: Status code {response.status_code}")

        except Exception as e:
            print(f"ERROR: Could not connect to chatbot - {str(e)}")

if __name__ == "__main__":
    test_chatbot()