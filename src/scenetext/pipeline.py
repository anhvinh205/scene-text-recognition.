"""End-to-end OCR pipeline: YOLO26 text detector -> CRNN recogniser."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from scenetext.data.transforms import build_transforms
from scenetext.recognition.checkpoint import load_recognizer
from scenetext.recognition.decode import WordLexicon, beam_search_decode, greedy_decode
from scenetext.utils import resolve_device

ImageLike = str | Path | Image.Image | np.ndarray


@dataclass
class TextRegion:
    bbox: list[float]  # x1, y1, x2, y2 in pixels
    det_conf: float
    text: str
    rec_conf: float

    def to_dict(self) -> dict:
        return asdict(self)


def to_pil(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return Image.open(image).convert("RGB")


def reading_order(regions: list[TextRegion], line_tol: float = 0.5) -> list[TextRegion]:
    """Sort regions top-to-bottom, left-to-right.

    Two boxes share a line when their vertical centres differ by less than
    ``line_tol`` x the height of the taller one.
    """
    remaining = sorted(regions, key=lambda r: (r.bbox[1] + r.bbox[3]) / 2)
    lines: list[list[TextRegion]] = []
    for r in remaining:
        cy, h = (r.bbox[1] + r.bbox[3]) / 2, r.bbox[3] - r.bbox[1]
        for line in lines:
            ref = line[0]
            ref_cy, ref_h = (ref.bbox[1] + ref.bbox[3]) / 2, ref.bbox[3] - ref.bbox[1]
            if abs(cy - ref_cy) < line_tol * max(h, ref_h):
                line.append(r)
                break
        else:
            lines.append([r])
    return [r for line in lines for r in sorted(line, key=lambda r: r.bbox[0])]


class OCRPipeline:
    def __init__(
        self,
        detector_weights: str,
        recognizer_weights: str,
        device: str = "auto",
        det_conf: float = 0.25,
        det_imgsz: int = 640,
        crop_padding: float = 0.05,
        decode_mode: str = "greedy",
        beam_width: int = 10,
        lexicon: WordLexicon | list[str] | str | Path | None = None,
        lm_weight: float = 0.5,
    ):
        from ultralytics import YOLO

        self.device = resolve_device(device)
        self.detector = YOLO(detector_weights)
        self.recognizer, self.vocab, self.img_size = load_recognizer(
            recognizer_weights, self.device
        )
        self.transform = build_transforms(self.img_size, train=False)
        self.det_conf, self.det_imgsz = det_conf, det_imgsz
        self.crop_padding = crop_padding
        if decode_mode not in ("greedy", "beam"):
            raise ValueError(f"decode_mode must be 'greedy' or 'beam', got {decode_mode!r}")
        self.decode_mode = decode_mode
        self.beam_width = beam_width
        if isinstance(lexicon, str | Path):
            self.lexicon = WordLexicon.from_file(lexicon)
        elif lexicon is not None and not isinstance(lexicon, WordLexicon):
            self.lexicon = WordLexicon(lexicon)
        else:
            self.lexicon = lexicon
        self.lm_weight = lm_weight

    @classmethod
    def from_config(cls, cfg: dict) -> OCRPipeline:
        return cls(
            cfg["detector_weights"],
            cfg["recognizer_weights"],
            device=cfg.get("device", "auto"),
            det_conf=cfg.get("det_conf", 0.25),
            det_imgsz=cfg.get("det_imgsz", 640),
            crop_padding=cfg.get("crop_padding", 0.05),
            decode_mode=cfg.get("decode_mode", "greedy"),
            beam_width=cfg.get("beam_width", 10),
            lexicon=cfg.get("lexicon_path") or cfg.get("lexicon"),
            lm_weight=cfg.get("lm_weight", 0.5),
        )

    def detect(self, img: Image.Image) -> list[tuple[list[float], float]]:
        res = self.detector.predict(
            img,
            conf=self.det_conf,
            imgsz=self.det_imgsz,
            verbose=False,
            device=self.device,
        )[0]
        return list(zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), strict=True))

    def _crop(self, img: Image.Image, box: list[float]) -> Image.Image:
        x1, y1, x2, y2 = box
        pw, ph = (x2 - x1) * self.crop_padding, (y2 - y1) * self.crop_padding
        return img.crop(
            (max(0, x1 - pw), max(0, y1 - ph), min(img.width, x2 + pw), min(img.height, y2 + ph))
        )

    @torch.no_grad()
    def recognize(self, crops: list[Image.Image], batch_size: int = 64) -> list[tuple[str, float]]:
        """Recognise many crops with batched forward passes."""
        out: list[tuple[str, float]] = []
        for i in range(0, len(crops), batch_size):
            batch = torch.stack([self.transform(c) for c in crops[i : i + batch_size]]).to(
                self.device
            )
            log_probs = self.recognizer(batch).float().cpu().numpy()
            if self.decode_mode == "beam":
                out.extend(
                    beam_search_decode(
                        log_probs,
                        self.vocab,
                        beam_width=self.beam_width,
                        lexicon=self.lexicon,
                        lm_weight=self.lm_weight,
                    )
                )
            else:
                out.extend(greedy_decode(log_probs, self.vocab))
        return out

    def predict(
        self, image: ImageLike, sort: bool = True, min_rec_conf: float = 0.0
    ) -> list[TextRegion]:
        img = to_pil(image)
        dets = self.detect(img)
        if not dets:
            return []
        texts = self.recognize([self._crop(img, b) for b, _ in dets])
        regions = [
            TextRegion(
                bbox=[round(v, 1) for v in b], det_conf=round(c, 4), text=t, rec_conf=round(rc, 4)
            )
            for (b, c), (t, rc) in zip(dets, texts, strict=True)
            if t and rc >= min_rec_conf
        ]
        return reading_order(regions) if sort else regions


def draw_regions(image: ImageLike, regions: list[TextRegion]) -> Image.Image:
    img = to_pil(image).copy()
    d = ImageDraw.Draw(img)
    for r in regions:
        x1, y1, x2, y2 = r.bbox
        d.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
        label = f"{r.text} ({r.det_conf:.2f})"
        tw = d.textlength(label)
        d.rectangle((x1, max(0, y1 - 14), x1 + tw + 4, max(0, y1 - 14) + 14), fill=(255, 0, 0))
        d.text((x1 + 2, max(0, y1 - 14)), label, fill=(255, 255, 255))
    return img
