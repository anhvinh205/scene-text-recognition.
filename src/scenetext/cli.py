from __future__ import annotations

import argparse
import json
from pathlib import Path

from scenetext.utils import get_logger, load_yaml

log = get_logger()


def _prepare(args: argparse.Namespace) -> None:
    from scenetext.data.crops import build_recognition_dataset
    from scenetext.data.icdar import parse_icdar_xml
    from scenetext.data.splits import split_items
    from scenetext.data.yolo_format import build_yolo_dataset

    cfg = load_yaml(args.config)
    anns = parse_icdar_xml(cfg["raw_dir"])
    scene_ids = sorted(ann.uid for ann in anns)
    train_scenes, val_scenes, test_scenes = split_items(
        scene_ids, cfg["split"]["val"], cfg["split"]["test"], cfg["seed"]
    )
    scene_splits = {"train": train_scenes, "val": val_scenes, "test": test_scenes}
    log.info("parsed %d images / %d word boxes", len(anns), sum(len(a.boxes) for a in anns))
    sizes = build_yolo_dataset(
        anns, cfg["yolo_dir"], cfg["split"]["val"], cfg["split"]["test"], cfg["seed"]
    )
    log.info("YOLO dataset -> %s %s", cfg["yolo_dir"], sizes)
    n = build_recognition_dataset(
        anns,
        cfg["ocr_dir"],
        cfg["min_crop_size"],
        tuple(cfg["brightness_range"]),
        scene_splits=scene_splits,
    )
    log.info("recognition dataset -> %s (%d crops)", cfg["ocr_dir"], n)


def _train_det(args: argparse.Namespace) -> None:
    from scenetext.detection.train import train_detector

    train_detector(load_yaml(args.config), args.device)


def _eval_det(args: argparse.Namespace) -> None:
    from scenetext.detection.train import evaluate_detector

    print(json.dumps(evaluate_detector(args.weights, args.data, args.split), indent=2))


def _train_rec(args: argparse.Namespace) -> None:
    from scenetext.recognition.train import train

    train(load_yaml(args.config), args.device)


def _eval_rec(args: argparse.Namespace) -> None:
    import torch.nn as nn

    from scenetext.data.vocab import Vocab
    from scenetext.recognition.checkpoint import load_recognizer
    from scenetext.recognition.evaluate import evaluate
    from scenetext.recognition.train import make_loaders
    from scenetext.utils import resolve_device

    cfg = load_yaml(args.config)
    device = resolve_device(args.device)
    model, vocab, _ = load_recognizer(args.weights, device)
    loaders, _ = make_loaders(cfg, vocab)
    res = evaluate(
        model,
        loaders[args.split],
        nn.CTCLoss(blank=Vocab.BLANK, zero_infinity=True),
        vocab,
        device,
        decode_mode=args.decode,
        beam_width=args.beam_width,
    )
    print(json.dumps({k: res[k] for k in ("loss", "accuracy", "cer")}, indent=2))


def _eval_e2e(args: argparse.Namespace) -> None:
    from scenetext.evaluation.e2e import evaluate_e2e
    from scenetext.pipeline import OCRPipeline

    pipe = OCRPipeline.from_config(load_yaml(args.config))
    print(json.dumps(evaluate_e2e(pipe, args.yolo_dir, args.split, args.iou), indent=2))


def _predict(args: argparse.Namespace) -> None:
    from scenetext.pipeline import OCRPipeline, draw_regions

    pipe = OCRPipeline.from_config(load_yaml(args.config))
    regions = pipe.predict(args.image)
    print(json.dumps([r.to_dict() for r in regions], indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        draw_regions(args.image, regions).save(args.output)
        log.info("saved visualisation to %s", args.output)


def _export(args: argparse.Namespace) -> None:
    from scenetext.export import export_recognizer_onnx

    print(export_recognizer_onnx(args.weights, args.output, verify=not args.no_verify))


def _serve(args: argparse.Namespace) -> None:
    import os

    import uvicorn

    os.environ.setdefault("PIPELINE_CONFIG", args.config)
    uvicorn.run("scenetext.serving.app:app", host=args.host, port=args.port, reload=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scenetext", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(func=fn)
        return sp

    sp = add("prepare-data", _prepare, "Parse ICDAR2003 -> YOLO dataset + recognition crops")
    sp.add_argument("--config", default="configs/data.yaml")

    sp = add("train-det", _train_det, "Train the YOLO26 text detector")
    sp.add_argument("--config", default="configs/detection.yaml")
    sp.add_argument("--device", default="auto")

    sp = add("eval-det", _eval_det, "Evaluate the detector (precision/recall/mAP)")
    sp.add_argument("--weights", default="runs/detect/train/weights/best.pt")
    sp.add_argument("--data", default="data/yolo_data/data.yaml")
    sp.add_argument("--split", default="test", choices=["val", "test"])

    sp = add("train-rec", _train_rec, "Train the CRNN recogniser")
    sp.add_argument("--config", default="configs/recognition.yaml")
    sp.add_argument("--device", default="auto")

    sp = add("eval-rec", _eval_rec, "Evaluate the recogniser (loss/accuracy/CER)")
    sp.add_argument("--config", default="configs/recognition.yaml")
    sp.add_argument("--weights", default="runs/recognition/best.pt")
    sp.add_argument("--split", default="test", choices=["val", "test"])
    sp.add_argument("--device", default="auto")
    sp.add_argument("--decode", default="greedy", choices=["greedy", "beam"])
    sp.add_argument("--beam-width", type=int, default=10)

    sp = add("eval-e2e", _eval_e2e, "End-to-end P/R/F1 of detector + recogniser")
    sp.add_argument("--config", default="configs/pipeline.yaml")
    sp.add_argument("--yolo-dir", default="data/yolo_data")
    sp.add_argument("--split", default="test", choices=["val", "test"])
    sp.add_argument("--iou", type=float, default=0.5)

    sp = add("predict", _predict, "Run OCR on one image")
    sp.add_argument("image")
    sp.add_argument("--config", default="configs/pipeline.yaml")
    sp.add_argument("--output", help="optional path for the annotated image")

    sp = add("export-onnx", _export, "Export recogniser to ONNX")
    sp.add_argument("--weights", default="runs/recognition/best.pt")
    sp.add_argument("--output", default="artifacts/recognizer.onnx")
    sp.add_argument("--no-verify", action="store_true")

    sp = add("serve", _serve, "Start the FastAPI inference server")
    sp.add_argument("--config", default="configs/pipeline.yaml")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8000)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
