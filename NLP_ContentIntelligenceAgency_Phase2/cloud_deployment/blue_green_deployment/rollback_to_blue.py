from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential

# Auth
credential = InteractiveBrowserCredential()
ml_client = MLClient.from_config(credential=credential)

# Config
endpoint_name = "emotion-endpoint2"
blue_deployment = "emotion-deploy"

# Get current endpoint
endpoint = ml_client.online_endpoints.get(endpoint_name)

# Switch 100% traffic back to blue
endpoint.traffic = {blue_deployment: 100}
ml_client.begin_create_or_update(endpoint).result()

print(f"Rolled back traffic to deployment '{blue_deployment}' ")
