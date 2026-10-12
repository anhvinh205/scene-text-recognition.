from __future__ import annotations

import os
import sys
from types import ModuleType

import pytest

from scenetext.tracking import log_metrics, log_params, start_tracking, tracking_enabled


def test_tracking_is_disabled_by_default():
    assert not tracking_enabled({})
    assert not tracking_enabled({"tracking": {"enabled": False}})


def test_start_tracking_disabled_does_not_import_mlflow():
    with start_tracking({}, "test-experiment") as tracker:
        assert tracker is None


def test_start_tracking_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(RuntimeError, match=r'pip install -e "\.\[tracking\]"'):
        with start_tracking({"tracking": {"enabled": True}}, "test-experiment"):
            pass


def test_start_tracking_configures_and_restores_environment(monkeypatch):
    calls = []
    mlflow = ModuleType("mlflow")
    mlflow.set_tracking_uri = lambda uri: calls.append(("uri", uri))
    mlflow.set_experiment = lambda name: calls.append(("experiment", name))

    class Run:
        def __enter__(self):
            calls.append(("start", "test-run"))

        def __exit__(self, *args):
            calls.append(("end", None))

    mlflow.start_run = lambda run_name=None: Run()
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "previous-uri")
    with start_tracking(
        {
            "tracking": {
                "enabled": True,
                "uri": "file:./test-mlruns",
                "experiment_name": "test-experiment",
                "run_name": "test-run",
            }
        },
        "unused-default",
    ) as tracker:
        assert tracker is mlflow
        assert calls[:3] == [
            ("uri", "file:./test-mlruns"),
            ("experiment", "test-experiment"),
            ("start", "test-run"),
        ]
        assert os.environ["MLFLOW_TRACKING_URI"] == "file:./test-mlruns"
    assert calls[-1] == ("end", None)
    assert os.environ["MLFLOW_TRACKING_URI"] == "previous-uri"


def test_log_helpers_flatten_params_and_ignore_non_numeric_metrics():
    class Tracker:
        params = None
        metrics = None

        def log_params(self, values):
            self.params = values

        def log_metrics(self, values, step):
            self.metrics = (values, step)

    tracker = Tracker()
    log_params(
        tracker,
        {"model": {"backbone": "resnet18"}, "tracking": {"enabled": True}, "seed": 0},
    )
    log_metrics(tracker, {"loss": 0.25, "note": "skip", "missing": None}, step=4)

    assert tracker.params == {"model.backbone": "resnet18", "seed": "0"}
    assert tracker.metrics == ({"loss": 0.25}, 4)
