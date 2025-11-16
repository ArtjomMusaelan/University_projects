from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential

credential = InteractiveBrowserCredential()
ml_client = MLClient.from_config(credential=credential)

endpoint_name = "emotion-endpoint2"

endpoint = ml_client.online_endpoints.get(endpoint_name)
endpoint.traffic = {
    "emotion-deploy": 80,
    "emotion-deploy-green": 20
}
ml_client.begin_create_or_update(endpoint).result()
print("✅ Traffic routed: 80% blue / 20% green")
