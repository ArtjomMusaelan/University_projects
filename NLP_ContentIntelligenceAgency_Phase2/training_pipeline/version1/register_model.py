"""Register a Hugging Face model in Azure Machine Learning workspace.

This script connects to an Azure ML workspace and registers a model located in the 'model' directory.
Sensitive values are loaded from environment variables with example placeholders.
"""

import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model

# Load sensitive info from environment variables (set these in your environment / CI)
SUBSCRIPTION_ID = os.environ.get("AZ_SUBSCRIPTION_ID", "your-subscription-id")
RESOURCE_GROUP = os.environ.get("AZ_RESOURCE_GROUP", "your-resource-group")
WORKSPACE_NAME = os.environ.get("AZ_WORKSPACE_NAME", "your-workspace-name")

# Optional: model metadata from env
MODEL_PATH = os.environ.get("MODEL_PATH", "model")
MODEL_NAME = os.environ.get("MODEL_NAME", "transformer-roberta-model")
MODEL_DESC = os.environ.get("MODEL_DESCRIPTION", "Hugging Face model for emotion detection")

# Connect to your Azure ML workspace using DefaultAzureCredential
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

# Register the model by uploading the contents of the model directory
model = ml_client.models.create_or_update(
    Model(
        path=MODEL_PATH,
        name=MODEL_NAME,
        description=MODEL_DESC,
        type="custom_model",
    )
)

print("Model registered:", model.name, "version:", model.version)
