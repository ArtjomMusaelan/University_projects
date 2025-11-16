import os
from azureml.core import (
    Workspace,
    ScriptRunConfig,
    Experiment,
    Environment,
    ComputeTarget,
)

print("Azure ML submission script started")

# Replace placeholder values (UPPERCASE_EXAMPLE) with your actual Azure ML resources
ws = Workspace.from_config()  # uses local config.json by default
compute_target = ComputeTarget(workspace=ws, name="COMPUTE_TARGET_NAME_EXAMPLE")
env = Environment.get(workspace=ws, name="ENV_NAME_EXAMPLE", version="1")

experiment_name = "example-experiment"
script_path = "example_script.py"
input_csv = "example_dataset.csv"

if not os.path.exists(input_csv):
    raise FileNotFoundError(f"Could not find {input_csv}")

src = ScriptRunConfig(
    source_directory=".",
    script=script_path,
    arguments=[
        "--csv_path", input_csv,
        "--n_trials", "10",
        "--max_per_emotion", "5",
        "--seed", "42",
        "--study_path", "example_hpo_study.pkl"
    ],
    compute_target=compute_target,
    environment=env
)

experiment = Experiment(ws, experiment_name)
print("Submitting run to AzureML...")
run = experiment.submit(src)
print(f"Run submitted! Monitor here: {run.get_portal_url()}")
print(f"RunId: {run.id}")

run.wait_for_completion(show_output=True)
