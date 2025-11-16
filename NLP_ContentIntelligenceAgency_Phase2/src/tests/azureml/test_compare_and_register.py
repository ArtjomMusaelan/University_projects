import json
import importlib
import sys

import pytest


class FakeModel:
    """Single fake model object with mutable F1 tag."""
    def __init__(self, f1: float, version: int = 1):
        self.tags = {"f1": str(f1)}
        self.version = version


class _FakeModels:
    """Emulates .list() and .create_or_update()."""

    def __init__(self, baseline: bool):
        self._baseline = baseline

    # iterator for ml_client.models.list()
    def list(self, *_, **__):
        if self._baseline:
            yield FakeModel(0.8)
        else:
            # empty iterator → StopIteration
            if False:  # pragma: no cover
                yield

    # register new model
    @staticmethod
    def create_or_update(_payload):
        return FakeModel(0.9, version=2)


class FakeMLClient:  # pylint: disable=too-few-public-methods
    """Stub for azure.ai.ml.MLClient, baseline flag passed in ctor."""

    def __init__(self, baseline: bool):
        self.models = _FakeModels(baseline)


def _noop_credential(*_a, **_k):  # noqa: D401
    return None


@pytest.mark.parametrize(
    "new_f1, baseline_present, expect_flag",
    [
        (0.9, True, "true"),   # better than baseline 0.8  → deploy
        (0.7, True, "false"),  # worse                    → no deploy
        (0.9, False, "true"),  # no baseline (StopIter)   → deploy
    ],
)
def test_compare_register(tmp_path, monkeypatch, new_f1, baseline_present, expect_flag):
    """Run compare_and_register.main() via CLI for three scenarios."""

    # import target AFTER stubs set up
    mod = importlib.import_module("src.compare_and_register")
    monkeypatch.setattr(
        mod,
        "MLClient",
        lambda *_, **__: FakeMLClient(baseline_present)
    )
    monkeypatch.setattr(mod, "DefaultAzureCredential", _noop_credential)

    # -------- prepare fake CLI files ----------
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    metrics = tmp_path / "m.json"
    metrics.write_text(json.dumps({"f1": new_f1}))
    flag = tmp_path / "flag.txt"

    cli = [
        "compare_and_register.py",
        "--new_model_dir",
        str(model_dir),
        "--new_metrics",
        str(metrics),
        "--model_name",
        "emotion-model",
        "--deploy_flag",
        str(flag),
    ]
    monkeypatch.setitem(
        sys.modules["__main__"].__dict__,
        "__file__",
        "compare_and_register.py"
    )
    monkeypatch.setattr(sys, "argv", cli)

    # dummy env vars for MLClient
    monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", "sub")
    monkeypatch.setenv("AZURE_RESOURCE_GROUP", "rg")
    monkeypatch.setenv("AZUREML_WORKSPACE_NAME", "ws")

    # --------------- run -----------------
    mod.main()

    # --------------- assert --------------
    assert flag.read_text() == expect_flag
