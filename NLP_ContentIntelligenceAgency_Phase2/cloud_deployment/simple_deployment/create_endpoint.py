from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    KubernetesOnlineEndpoint,
    KubernetesOnlineDeployment,
    CodeConfiguration,
    OnlineRequestSettings
)
from azure.identity import InteractiveBrowserCredential
from azure.core.exceptions import HttpResponseError

# Auth
credential = InteractiveBrowserCredential()
ml_client = MLClient.from_config(credential=credential)

# Config
endpoint_name = "emotion-endpoint5"
deployment_name = "emotion-deploy"
model_path = "azureml:nlp8_model:1"
environment_path = "azureml:nlp8-env2:6"
compute_target = "adsai-lambda-0"

# Create Kubernetes Endpoint (if not exists)
try:
    ml_client.online_endpoints.get(endpoint_name)
    print(f"✅ Endpoint '{endpoint_name}' already exists.")
except Exception:
    print(f"🚀 Creating new public endpoint '{endpoint_name}'...")
    endpoint = KubernetesOnlineEndpoint(
        name=endpoint_name,
        description="Emotion classification endpoint on Kubernetes",
        auth_mode="key",
        compute=compute_target
    )
    ml_client.begin_create_or_update(endpoint).result()
    print("✅ Endpoint created.")

# Create Deployment
print("🚀 Creating deployment...")
deployment = KubernetesOnlineDeployment(
    name=deployment_name,
    endpoint_name=endpoint_name,
    model=model_path,
    environment=environment_path,
    code_configuration=CodeConfiguration(code="./", scoring_script="score.py"),
    instance_type="defaultinstancetype",
    instance_count=1,
    request_settings=OnlineRequestSettings(
        request_timeout_ms=120000,  # 2 minutes
        max_concurrent_requests_per_instance=1,
        max_queue_wait_ms=60000     # 1 minute
    )
)

try:
    ml_client.begin_create_or_update(deployment).result()
    print("✅ Deployment created successfully.")
except HttpResponseError as e:
    print("❌ Deployment failed. Fetching logs...")
    logs = ml_client.online_deployments.get_logs(
        name=deployment_name,
        endpoint_name=endpoint_name,
        lines=250,
        container_type="inference-server"
    )
    print(logs)
    raise e

# Route 100% Traffic to Deployment
print("🚦 Routing 100% traffic to deployment...")
endpoint = ml_client.online_endpoints.get(endpoint_name)
endpoint.traffic = {deployment_name: 100}
ml_client.begin_create_or_update(endpoint).result()

print("✅ Simple deployment complete.")
