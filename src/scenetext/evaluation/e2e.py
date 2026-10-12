"""End-to-end evaluation: a prediction is correct if IoU >= thr AND the text matches."""

from __future__ import annotations

import json
from pathlib import Path


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(
    preds: list[dict], gts: list[dict], iou_thr: float = 0.5, require_text: bool = True
) -> int:
    """Greedy one-to-one matching by descending IoU. Returns number of true positives.

    ``preds``/``gts`` are dicts with ``xyxy`` and ``text`` keys.
    """
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            v = iou(p["xyxy"], g["xyxy"])
            if v >= iou_thr and (not require_text or p["text"].lower() == g["text"].lower()):
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_p, used_g, tp = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            tp += 1
    return tp


def prf(tp: int, n_pred: int, n_gt: int) -> dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def evaluate_e2e(pipeline, yolo_dir: str | Path, split: str = "test", iou_thr: float = 0.5) -> dict:
    """Run the full pipeline over a split written by ``prepare-data``."""
    yolo_dir = Path(yolo_dir)
    gt = json.loads((yolo_dir / "annotations.json").read_text())[split]
    tp_e2e = tp_det = n_pred = n_gt = 0
    for item in gt:
        regions = pipeline.predict(yolo_dir / item["image"], sort=False)
        preds = [{"xyxy": r.bbox, "text": r.text} for r in regions]
        n_pred += len(preds)
        n_gt += len(item["boxes"])
        tp_e2e += match_boxes(preds, item["boxes"], iou_thr, require_text=True)
        tp_det += match_boxes(preds, item["boxes"], iou_thr, require_text=False)
    return {
        "split": split,
        "images": len(gt),
        "iou_threshold": iou_thr,
        "detection": prf(tp_det, n_pred, n_gt),
        "end_to_end": prf(tp_e2e, n_pred, n_gt),
    }
