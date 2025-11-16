import yaml
from pathlib import Path, re

PIPELINE_YAML = Path("azureml/pipeline_retrain.yml")


def test_pipeline_structure():
    data = yaml.safe_load(PIPELINE_YAML.read_text())
    jobs = data["jobs"]
    # new DAG: prep → train → compare → (deploy?)
    assert set(jobs) == {"merge_feedback", "prep", "train", "eval", "compare", "deploy"}

    # ensure deploy is conditional (safe rollout)
    deploy_job = jobs["deploy"]
    assert "condition" in deploy_job
    assert re.search(r"==\s*'true'", deploy_job["condition"])

    pattern = r"\$\{\{parent\.jobs\.[^.]+\.outputs\.eval_metrics\}\}"
    assert re.search(pattern, str(jobs["compare"]))
