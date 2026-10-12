"""Build the recognition dataset by cropping every word box out of its scene image."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scenetext.data.icdar import ImageAnnotation


def crop_is_usable(
    crop: Image.Image, min_size: int = 10, brightness_range: tuple[float, float] = (35, 220)
) -> bool:
    if crop.size[0] < min_size or crop.size[1] < min_size:
        return False
    mean = float(np.asarray(crop.convert("L")).mean())
    return brightness_range[0] <= mean <= brightness_range[1]


def build_recognition_dataset(
    annotations: list[ImageAnnotation],
    out_dir: str | Path,
    min_size: int = 10,
    brightness_range: tuple[float, float] = (35, 220),
    scene_splits: dict[str, list[str]] | None = None,
) -> int:
    """Write ``images/*.jpg`` and ``labels.txt`` (``relpath<TAB>label<TAB>scene_uid``).

    The scene uid lets the train/val/test split be done by scene, not by crop.
    """
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for ann in annotations:
        with Image.open(ann.path) as im:
            im = im.convert("RGB")
            for i, b in enumerate(ann.boxes):
                x1, y1 = max(0, int(b.x)), max(0, int(b.y))
                x2, y2 = min(im.width, int(b.x + b.w)), min(im.height, int(b.y + b.h))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = im.crop((x1, y1, x2, y2))
                if not crop_is_usable(crop, min_size, brightness_range):
                    continue
                rel = f"images/{ann.uid}_{i:03d}.jpg"
                crop.save(out_dir / rel, quality=95)
                lines.append(f"{rel}\t{b.text}\t{ann.uid}")
    (out_dir / "labels.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if scene_splits is not None:
        (out_dir / "scene_splits.json").write_text(json.dumps(scene_splits, indent=1))
    return len(lines)


def read_labels(ocr_dir: str | Path) -> list[tuple[str, str, str]]:
    """Return ``(abs_image_path, label, scene_uid)`` triples."""
    ocr_dir = Path(ocr_dir).resolve()
    rows = []
    for line in (ocr_dir / "labels.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            fields = line.split("\t")
            if len(fields) != 3:
                raise ValueError("each labels.txt row must contain path, label, and scene uid")
            rel, label, uid = fields
            image_path = (ocr_dir / rel).resolve()
            if not image_path.is_relative_to(ocr_dir):
                raise ValueError(f"label image path escapes dataset root: {rel!r}")
            rows.append((str(image_path), label, uid))
    return rows
