import requests
import json

# Test the chatbot endpoint
def test_chatbot():
    url = "http://localhost:8000/chat"

    # Test query about replicas (the original issue)
    payload = {
        "query": "Explain this: 'replicas'",
        "selected_text": None
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        print("Sending test query to chatbot...")
        response = requests.post(url, data=json.dumps(payload), headers=headers)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        if response.status_code == 200:
            print("SUCCESS: Chatbot responded successfully!")
            # Try to parse the JSON response
            try:
                data = response.json()
                print(f"Response data: {json.dumps(data, indent=2)}")
            except:
                print(f"Response text: {response.text}")
        else:
            print("ERROR: Chatbot returned an error")

    except Exception as e:
        print(f"ERROR: Could not connect to chatbot - {str(e)}")

if __name__ == "__main__":
    test_chatbot()