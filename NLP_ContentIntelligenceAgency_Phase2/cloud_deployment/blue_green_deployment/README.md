## Deployed Kubernetes Endpoint (Simple deployment)

**Endpoint Name:** `emotion-endpoint1`  
**Deployment Name:** `emotion-deploy`  
**Compute Target:** `adsai-lambda-2` (Kubernetes)

# Example placeholder values — replace with your real endpoint and key or load from environment variables
scoring_uri = "https://example.com/api/v1/endpoint/emotion-endpoint/score"
api_key = "YOUR_API_KEY_HERE"

---
### Retrieve URI and Key

Use `uri_key.py` to extract the **scoring URI** and **primary key** from the deployed endpoint.

```bash
python uri_key.py
```

---
### Test the Deployment
Use test_request.py to send a test emotion classification request to the endpoint.

```bash
python test_request.py
```