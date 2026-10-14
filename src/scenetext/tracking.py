"""Optional MLflow experiment tracking helpers."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def tracking_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("tracking", {}).get("enabled", False))


@contextmanager
def start_tracking(cfg: dict[str, Any], experiment_name: str) -> Iterator[Any | None]:
    tracking_cfg = cfg.get("tracking", {})
    if not tracking_cfg.get("enabled", False):
        yield None
        return

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow tracking is enabled but MLflow is not installed. "
            'Install it with: pip install -e ".[tracking]"'
        ) from exc

    tracking_uri = tracking_cfg.get("uri", "sqlite:///mlflow.db")
    experiment_name = tracking_cfg.get("experiment_name", experiment_name)
    run_name = tracking_cfg.get("run_name")
    previous_uri = os.environ.get("MLFLOW_TRACKING_URI")
    previous_experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    previous_run = os.environ.get("MLFLOW_RUN")

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    if run_name:
        os.environ["MLFLOW_RUN"] = run_name

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            yield mlflow
    finally:
        for key, value in (
            ("MLFLOW_TRACKING_URI", previous_uri),
            ("MLFLOW_EXPERIMENT_NAME", previous_experiment),
            ("MLFLOW_RUN", previous_run),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def log_params(tracker: Any, params: dict[str, Any], prefix: str = "") -> None:
    flattened: dict[str, str] = {}

    def flatten(values: dict[str, Any], parent: str = "") -> None:
        for key, value in values.items():
            if key == "tracking":
                continue
            name = f"{parent}.{key}" if parent else key
            if isinstance(value, dict):
                flatten(value, name)
            elif value is not None:
                serialized = (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, list | tuple | bool)
                    else str(value)
                )
                flattened[name] = serialized[:500]

    flatten(params, prefix)
    if flattened:
        tracker.log_params(flattened)


def log_metrics(tracker: Any, metrics: dict[str, Any], step: int) -> None:
    numeric = {}
    for key, value in metrics.items():
        if value is None:
            continue
        try:
            numeric[key] = float(value)
        except (TypeError, ValueError):
            continue
    if numeric:
        tracker.log_metrics(numeric, step=step)


@contextmanager
def ultralytics_mlflow_setting(enabled: bool) -> Iterator[None]:
    """Control Ultralytics' built-in MLflow callback for this training call."""
    from ultralytics.utils import SETTINGS

    original = SETTINGS["mlflow"]
    SETTINGS.update({"mlflow": enabled})
    try:
        yield
    finally:
        SETTINGS.update({"mlflow": original})
