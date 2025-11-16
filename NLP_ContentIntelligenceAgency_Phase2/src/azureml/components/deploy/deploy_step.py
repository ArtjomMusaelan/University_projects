"""
Deploy a registered model to an existing online endpoint if flag == true.
"""

import os
import pathlib
import argparse
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def _get(var, *fallbacks):
    for name in (var, *fallbacks):
        val = os.getenv(name)
        if val:
            return val
    raise RuntimeError(f"Missing env var: tried {var,*fallbacks}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy_flag", type=pathlib.Path, required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--endpoint_name", required=True)
    args = ap.parse_args()

    if args.deploy_flag.read_text().strip().lower() != "true":
        print("No deployment requested; exiting")
        return

    ml_client = MLClient(
        DefaultAzureCredential(),
        _get("AZURE_SUBSCRIPTION_ID", "SUBSCRIPTION_ID", "AZUREML_ARM_SUBSCRIPTION_ID"),
        _get("AZURE_RESOURCE_GROUP", "RESOURCE_GROUP", "AZUREML_ARM_RESOURCEGROUP"),
        _get("AZURE_WORKSPACE_NAME", "WORKSPACE_NAME", "AZUREML_ARM_WORKSPACE_NAME"),
    )

    latest_model = next(
        ml_client.models.list(name=args.model_name, order_by="version desc")
    )

    # Simple blue/green update: assumes single deployment named 'blue'
    ml_client.online_deployments.begin_create_or_update(
        name="blue",
        endpoint_name=args.endpoint_name,
        model=latest_model.id,
        instance_type="Standard_DS3_v2",  # keep as current
        instance_count=1,
    ).result()

    print(f"✔ Endpoint {args.endpoint_name} now serves model v{latest_model.version}")


if __name__ == "__main__":
    main()
