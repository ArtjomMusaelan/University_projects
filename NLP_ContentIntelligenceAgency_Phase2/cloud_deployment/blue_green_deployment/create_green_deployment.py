from azure.ai.ml import MLClient
from azure.ai.ml.entities import KubernetesOnlineDeployment, CodeConfiguration
from azure.identity import InteractiveBrowserCredential
from azure.core.exceptions import HttpResponseError

# Auth
credential = InteractiveBrowserCredential()
ml_client = MLClient.from_config(credential=credential)

# Config
endpoint_name = "emotion-endpoint2"  # EXISTING endpoint
green_deployment_name = "emotion-deploy-green"  # NEW deployment
model_path = "azureml:nlp8_model:1"
environment_path = "azureml:nlp8-env2:6"
compute_target = "adsai-lambda-0"

# Create Green Deployment 
print("Creating Green Deployment...")
green_deployment = KubernetesOnlineDeployment(
    name=green_deployment_name,
    endpoint_name=endpoint_name,
    model=model_path,
    environment=environment_path,
    code_configuration=CodeConfiguration(code="./", scoring_script="score.py"),
    instance_type="defaultinstancetype",
    instance_count=1
)

try:
    ml_client.begin_create_or_update(green_deployment).result()
    print("Green deployment created successfully.")
except HttpResponseError as e:
    print("Green deployment failed.")
    logs = ml_client.online_deployments.get_logs(
        name=green_deployment_name,
        endpoint_name=endpoint_name,
        lines=250,
        container_type="inference-server"
    )
    print(logs)
    raise e

# Shift Traffic to Green
print("Routing 100% traffic to Green deployment...")
endpoint = ml_client.online_endpoints.get(endpoint_name)
endpoint.traffic = {green_deployment_name: 100}
ml_client.begin_create_or_update(endpoint).result()

print("Blue-Green deployment complete.")

