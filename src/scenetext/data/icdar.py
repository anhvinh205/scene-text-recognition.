"Parse the ICDAR2003 ``words.xml`` annotation file."

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

_VALID = re.compile(r"[0-9a-z]+")


@dataclass
class WordBox:
    x: float  # top-left
    y: float
    w: float
    h: float
    text: str  # lower-cased, [0-9a-z]+


@dataclass
class ImageAnnotation:
    path: Path
    width: int
    height: int
    boxes: list[WordBox] = field(default_factory=list)

    @property
    def uid(self) -> str:
        """Unique, filesystem-safe name (folder + file stem)."""
        return f"{self.path.parent.name}__{self.path.stem}"


def is_valid_label(text: str | None) -> bool:
    """Keep only words made of ASCII digits / letters (course spec: 0-9, a-z)."""
    return bool(text) and _VALID.fullmatch(text.strip().lower()) is not None


def parse_icdar_xml(root_dir: str | Path, keep_empty_images: bool = False) -> list[ImageAnnotation]:
    root_dir = Path(root_dir).resolve()
    tree = ET.parse(root_dir / "words.xml")
    annotations: list[ImageAnnotation] = []

    for img in tree.getroot().findall("image"):
        name = img.findtext("imageName", default="").strip()
        res = img.find("resolution")
        if not name or res is None:
            continue
        image_path = (root_dir / name).resolve()
        if not image_path.is_relative_to(root_dir):
            raise ValueError(f"annotation image path escapes dataset root: {name!r}")
        ann = ImageAnnotation(
            path=image_path,
            width=int(float(res.attrib["x"])),
            height=int(float(res.attrib["y"])),
        )
        rects = img.find("taggedRectangles")
        for rect in rects if rects is not None else []:
            text = rect.findtext("tag")
            if not is_valid_label(text):
                continue
            x1 = max(0.0, float(rect.attrib["x"]))
            y1 = max(0.0, float(rect.attrib["y"]))
            x2 = min(ann.width, float(rect.attrib["x"]) + float(rect.attrib["width"]))
            y2 = min(ann.height, float(rect.attrib["y"]) + float(rect.attrib["height"]))
            if x2 <= x1 or y2 <= y1:
                continue
            ann.boxes.append(
                WordBox(
                    x=x1,
                    y=y1,
                    w=x2 - x1,
                    h=y2 - y1,
                    text=text.strip().lower(),
                )
            )
        if ann.boxes or keep_empty_images:
            annotations.append(ann)
    return annotations
