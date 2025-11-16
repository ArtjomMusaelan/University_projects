"""
Compare new model metrics with the latest production model.
If better, register the new model and emit a deploy flag.
"""

import argparse
import json
import pathlib
import os


def _get(var, *fallbacks):
    """
    Returns the first found environment from the list.
    Throws a RuntimeError if none is specified.
    """
    for name in (var, *fallbacks):
        val = os.getenv(name)
        if val:
            return val
    raise RuntimeError(f"Missing env var: tried {var,*fallbacks}")


try:
    # Heavy SDK – present in runtime image, but not in CI test environment
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
except ModuleNotFoundError:  # pragma: no cover
    from types import SimpleNamespace  # lightweight stub

    class _StubMLClient(SimpleNamespace):  # type: ignore
        pass

    MLClient = _StubMLClient  # type: ignore

    def _noop_credential(*_args, **_kwargs):  # noqa: D401
        """Stub that mimics DefaultAzureCredential but does nothing."""

        return None

    DefaultAzureCredential = _noop_credential  # type: ignore


def load_json(path):
    with open(path) as fp:
        return json.load(fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_model_dir", type=pathlib.Path, required=True)
    ap.add_argument("--new_metrics", type=pathlib.Path, required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--metric_key", default="f1")
    ap.add_argument("--min_improvement", type=float, default=0.0)
    ap.add_argument("--deploy_flag", type=pathlib.Path, required=True)
    args = ap.parse_args()

    ml_client = MLClient(
        DefaultAzureCredential(),
        _get("AZURE_SUBSCRIPTION_ID", "SUBSCRIPTION_ID", "AZUREML_ARM_SUBSCRIPTION_ID"),
        _get("AZURE_RESOURCE_GROUP", "RESOURCE_GROUP", "AZUREML_ARM_RESOURCEGROUP"),
        _get("AZURE_WORKSPACE_NAME", "WORKSPACE_NAME", "AZUREML_ARM_WORKSPACE_NAME"),
    )

    new_score = float(load_json(args.new_metrics).get(args.metric_key, 0))

    # Fetch latest registered model
    try:
        latest_model = next(
            ml_client.models.list(name=args.model_name, order_by="version desc")
        )
        bas_score = float(latest_model.tags.get(args.metric_key, 0))
    except StopIteration:
        latest_model, bas_score = None, -1  # No baseline yet

    improved = new_score > bas_score + args.min_improvement

    if improved:
        reg = ml_client.models.create_or_update(
            {
                "name": args.model_name,
                "path": str(args.new_model_dir),
                "tags": {args.metric_key: str(new_score)},
            }
        )
        deploy_value = "true"
        print(f"✔ Registered model v{reg.version} (better {bas_score} → {new_score})")
    else:
        deploy_value = "false"
        print(f"✖ New model not better ({new_score} ≤ {bas_score}); skip deploy")

    args.deploy_flag.parent.mkdir(exist_ok=True, parents=True)
    args.deploy_flag.write_text(deploy_value)


if __name__ == "__main__":
    main()
