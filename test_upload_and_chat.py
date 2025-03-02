import requests
import json
import os
import time

# Configuration
BASE_URL = "http://localhost:8000"
TEST_FILE_PATH = "test_document.txt"

def test_upload():
    """Test the document upload functionality"""
    print("\n=== Testing Document Upload ===")
    
    # Check if test file exists
    if not os.path.exists(TEST_FILE_PATH):
        print(f"Error: Test file {TEST_FILE_PATH} not found")
        return False
    
    # Prepare the file for upload
    files = {'files': open(TEST_FILE_PATH, 'rb')}
    
    # Send the upload request
    print(f"Uploading file: {TEST_FILE_PATH}")
    response = requests.post(f"{BASE_URL}/upload", files=files)
    
    # Close the file
    files['files'].close()
    
    # Check the response
    if response.status_code == 200:
        result = response.json()
        print(f"Upload successful: {json.dumps(result, indent=2)}")
        return True
    else:
        print(f"Upload failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_chat(use_rag=True):
    """Test the chat functionality with or without RAG"""
    rag_status = "with RAG" if use_rag else "without RAG"
    print(f"\n=== Testing Chat {rag_status} ===")
    
    # Prepare the chat request
    chat_data = {
        "message": "What are the key points in the document I uploaded?",
        "max_tokens": 1000,
        "use_rag": use_rag
    }
    
    # Send the chat request
    print(f"Sending chat request: {json.dumps(chat_data, indent=2)}")
    response = requests.post(f"{BASE_URL}/chat", json=chat_data)
    
    # Check the response
    if response.status_code == 200:
        result = response.json()
        print(f"Chat response successful")
        print(f"Response: {result['response']}")
        return True
    else:
        print(f"Chat request failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return False

if __name__ == "__main__":
    # Run the tests
    upload_success = test_upload()
    
    if upload_success:
        # Wait a moment for processing to complete
        print("\nWaiting for document processing to complete...")
        time.sleep(2)
        
        # Test chat with RAG
        test_chat(use_rag=True)
        
        # Test chat without RAG
        test_chat(use_rag=False)
    else:
        print("\nSkipping chat tests due to upload failure")
