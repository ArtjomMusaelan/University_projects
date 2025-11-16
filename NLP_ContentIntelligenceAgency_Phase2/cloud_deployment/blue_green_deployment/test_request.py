import requests
import json

# Example placeholder values — replace with your real endpoint and key or load from environment variables
scoring_uri = "https://example.com/api/v1/endpoint/emotion-endpoint/score"
api_key = "YOUR_API_KEY_HERE"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

sample_input = {
    "text": "That is very sad!"
}

try:
    print("📡 Sending request to endpoint...")
    response = requests.post(scoring_uri, headers=headers, json=sample_input, timeout=120)  # timeout in seconds

    print("🔄 Status code:", response.status_code)
    
    try:
        print("✅ Response JSON:", json.dumps(response.json(), indent=2))
    except json.JSONDecodeError:
        print("❌ Could not decode JSON: possibly a timeout or internal error")
        print("🪵 Raw response:\n", response.text)

except requests.exceptions.Timeout:
    print("⏰ Request timed out after 120 seconds.")
except requests.exceptions.RequestException as e:
    print(f"❌ Request failed: {e}")
