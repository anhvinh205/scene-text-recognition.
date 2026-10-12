from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from scenetext.data.icdar import ImageAnnotation, WordBox
from scenetext.data.splits import split_items


def to_yolo_line(box: WordBox, img_w: int, img_h: int, class_id: int = 0) -> str:
    """Top-left (x, y, w, h) in pixels -> ``class cx cy w h`` normalised to [0, 1]."""
    cx = (box.x + box.w / 2) / img_w
    cy = (box.y + box.h / 2) / img_h
    w, h = box.w / img_w, box.h / img_h

    def clip(v: float) -> float:
        return min(max(v, 0.0), 1.0)

    return f"{class_id} {clip(cx):.6f} {clip(cy):.6f} {clip(w):.6f} {clip(h):.6f}"


def _write_split(anns: list[ImageAnnotation], out_dir: Path) -> list[dict]:
    """Copy images, write YOLO label files; return ground truth (with text) for e2e evaluation."""
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    gt = []
    for ann in anns:
        name = f"{ann.uid}{ann.path.suffix.lower()}"
        shutil.copy(ann.path, out_dir / "images" / name)
        lines = [to_yolo_line(b, ann.width, ann.height) for b in ann.boxes]
        (out_dir / "labels" / f"{ann.uid}.txt").write_text("\n".join(lines) + "\n")
        gt.append(
            {
                "image": f"{out_dir.name}/images/{name}",
                "boxes": [
                    {"xyxy": [b.x, b.y, b.x + b.w, b.y + b.h], "text": b.text} for b in ann.boxes
                ],
            }
        )
    return gt


def build_yolo_dataset(
    annotations: list[ImageAnnotation],
    out_dir: str | Path,
    val: float = 0.2,
    test: float = 0.1,
    seed: int = 0,
) -> dict[str, int]:
    """Write ``train/val/test`` image+label folders and ``data.yaml``. Returns split sizes."""
    out_dir = Path(out_dir).resolve()
    annotations = sorted(annotations, key=lambda ann: ann.uid)
    train, val_s, test_s = split_items(annotations, val=val, test=test, seed=seed)
    ground_truth = {}
    for name, split in (("train", train), ("val", val_s), ("test", test_s)):
        ground_truth[name] = _write_split(split, out_dir / name)
    (out_dir / "annotations.json").write_text(json.dumps(ground_truth, indent=1))

    data_yaml = {
        "train": str((out_dir / "train").resolve()),
        "val": str((out_dir / "val").resolve()),
        "test": str((out_dir / "test").resolve()),
        "nc": 1,
        "names": ["text"],
    }
    with open(out_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    return {"train": len(train), "val": len(val_s), "test": len(test_s)}
