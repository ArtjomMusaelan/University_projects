from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential

credential = InteractiveBrowserCredential()
ml_client = MLClient.from_config(credential=credential)

endpoint_name = "emotion-endpoint3"  
deployment_name = "emotion-deploy" 

logs = ml_client.online_deployments.get_logs(
    name=deployment_name,
    endpoint_name=endpoint_name,
    lines=250,
    container_type="inference-server"
)

print(logs)
