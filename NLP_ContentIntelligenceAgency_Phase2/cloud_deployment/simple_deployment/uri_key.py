from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential

credential = InteractiveBrowserCredential()
ml_client = MLClient.from_config(credential=credential)

endpoint_name = "emotion-endpoint5"

# Get scoring URI
endpoint = ml_client.online_endpoints.get(endpoint_name)
print("Scoring URI:", endpoint.scoring_uri)

# Get access key
keys = ml_client.online_endpoints.get_keys(endpoint_name)
print("Primary key:", keys.primary_key)

