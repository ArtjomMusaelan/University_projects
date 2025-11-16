"""Azure ML Pipeline Script (Version 2)

This script sets up and runs an Azure Machine Learning pipeline for training and evaluating a model.
It loads the workspace, datasets, environment, and compute target, then defines and submits a pipeline step.
"""

from azureml.core import Workspace, Dataset, Experiment, Environment
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep

# Load Azure ML workspace from config file
ws = Workspace.from_config()

# Get the compute target by name
compute_target = ComputeTarget(workspace=ws, name="adsai-lambda-2")

# Get the environment by name and version
env = Environment.get(workspace=ws, name="nlp8-env2", version="4")

# Set up the run configuration with the specified environment
run_config = RunConfiguration()
run_config.environment = env

# Retrieve the training and validation datasets by name
train_data = Dataset.get_by_name(ws, name="dataset_train")
val_data = Dataset.get_by_name(ws, name="dataset_val")

# Define the training step using a Python script
train_step = PythonScriptStep(
    name="Train and Evaluate Model",
    script_name="train_script_v1.py",
    source_directory=(
        "C:/Users/Mohon/OneDrive/Документы/GitHub/"
        "2024-25d-fai2-adsai-group-nlp8/data/upload_temp/script_folder"
    ),
    arguments=[
        "--train_data", train_data.as_named_input("train_data").as_mount(),
        "--val_data", val_data.as_named_input("val_data").as_mount(),
        "--model_dir", "transformer-roberta-model",  # Pass ONLY the model name here!
        "--output_dir", "outputs",
    ],
    compute_target=compute_target,
    runconfig=run_config,
    allow_reuse=False,
)

# Create the pipeline with the defined step
pipeline = Pipeline(workspace=ws, steps=[train_step])

# Create an experiment and submit the pipeline run
experiment = Experiment(workspace=ws, name="evaluate_model_experiment_new_version_2")
pipeline_run = experiment.submit(pipeline)

# Wait for the pipeline run to complete and show output
pipeline_run.wait_for_completion(show_output=True)
