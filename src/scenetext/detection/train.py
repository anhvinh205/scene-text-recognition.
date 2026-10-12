"""Fine-tune YOLO26 to detect a single class: ``text``."""

from __future__ import annotations

from scenetext.tracking import log_params, start_tracking, ultralytics_mlflow_setting
from scenetext.utils import resolve_device, seed_everything


def train_detector(cfg: dict, device: str = "auto"):
    seed_everything(cfg.get("seed", 0))
    with start_tracking(cfg, "scene-text-detection") as tracker:
        with ultralytics_mlflow_setting(tracker is not None):
            from ultralytics import YOLO

            if tracker is not None:
                log_params(
                    tracker,
                    {
                        "model": cfg["model"],
                        "data": cfg["data"],
                        "epochs": cfg["epochs"],
                        "imgsz": cfg["imgsz"],
                        "patience": cfg["patience"],
                        "batch": cfg.get("batch", 16),
                        "seed": cfg.get("seed", 0),
                        "device": resolve_device(device),
                    },
                    prefix="config",
                )
            model = YOLO(cfg["model"])
            return model.train(
                data=cfg["data"],
                epochs=cfg["epochs"],
                imgsz=cfg["imgsz"],
                patience=cfg["patience"],
                batch=cfg.get("batch", 16),
                cache=cfg.get("cache", True),
                seed=cfg.get("seed", 0),
                project=cfg.get("project", "runs/detect"),
                name=cfg.get("name", "train"),
                device=resolve_device(device) if device != "auto" else None,
                plots=True,
            )


def evaluate_detector(weights: str, data_yaml: str, split: str = "test", imgsz: int = 640) -> dict:
    from ultralytics import YOLO

    m = YOLO(weights).val(data=data_yaml, split=split, imgsz=imgsz)
    return {
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
    }
