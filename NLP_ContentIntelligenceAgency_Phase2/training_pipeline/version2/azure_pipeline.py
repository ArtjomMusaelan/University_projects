import os
from azureml.core import Workspace, Dataset, Experiment, Environment
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.authentication import InteractiveLoginAuthentication

# ----------- CONFIGURATION (replace with your values or set env vars) -----------
# Example env var names: AZ_SUBSCRIPTION_ID, AZ_RESOURCE_GROUP, AZ_WORKSPACE_NAME, AZ_COMPUTE_NAME
subscription_id = os.environ.get("AZ_SUBSCRIPTION_ID", "00000000-0000-0000-0000-000000000000")
resource_group = os.environ.get("AZ_RESOURCE_GROUP", "my-resource-group")
workspace_name = os.environ.get("AZ_WORKSPACE_NAME", "my-workspace")
compute_name = os.environ.get("AZ_COMPUTE_NAME", "my-compute")

# ----------- AUTHENTICATION -----------
interactive_auth = InteractiveLoginAuthentication()
ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name, auth=interactive_auth)

compute_target = ComputeTarget(workspace=ws, name=compute_name)
env = Environment.get(workspace=ws, name="nlp8-env2", version="5")
run_config = RunConfiguration()
run_config.environment = env

train_data = Dataset.get_by_name(ws, name="dataset_train")
val_data = Dataset.get_by_name(ws, name="dataset_val")

train_step = PythonScriptStep(
    name="Train and Evaluate Model",
    script_name="train_script.py",
    arguments=[
        "--train_data", train_data.as_named_input("train_data").as_mount(),
        "--val_data", val_data.as_named_input("val_data").as_mount(),
        "--model_dir", "transformer-roberta-model",  # Pass ONLY the model name here!
        "--output_dir", "outputs"
    ],
    compute_target=compute_target,
    runconfig=run_config,
    allow_reuse=False
)

pipeline = Pipeline(workspace=ws, steps=[train_step])
experiment = Experiment(workspace=ws, name="evaluate_model_experiment")
pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)