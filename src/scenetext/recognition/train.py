"""Train the CRNN text recogniser with CTC loss."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scenetext.data.crops import read_labels
from scenetext.data.dataset import STRDataset
from scenetext.data.splits import split_items
from scenetext.data.transforms import build_transforms
from scenetext.data.vocab import Vocab
from scenetext.models.crnn import CRNN
from scenetext.recognition.checkpoint import load_recognizer, save_checkpoint
from scenetext.recognition.evaluate import evaluate
from scenetext.tracking import log_metrics, log_params, start_tracking
from scenetext.utils import get_logger, resolve_device, seed_everything

log = get_logger()


def make_loaders(cfg: dict, vocab: Vocab):
    rows = read_labels(cfg["data_dir"])
    bad = {c for _, lab, _ in rows for c in lab} - set(vocab.chars)
    if bad:
        raise ValueError(f"labels contain characters outside the vocabulary: {sorted(bad)}")

    scenes = sorted({uid for _, _, uid in rows})
    split_path = Path(cfg["data_dir"]) / "scene_splits.json"
    if split_path.is_file():
        scene_splits = json.loads(split_path.read_text(encoding="utf-8"))
        if set(scene_splits) != {"train", "val", "test"}:
            raise ValueError("scene_splits.json must define train, val, and test")
        bucket = {}
        for name, scene_ids in scene_splits.items():
            for uid in scene_ids:
                if uid in bucket:
                    raise ValueError(f"scene {uid!r} appears in multiple splits")
                bucket[uid] = name
        missing = set(scenes) - set(bucket)
        if missing:
            raise ValueError(
                f"labels contain scenes missing from scene_splits.json: {sorted(missing)}"
            )
    else:
        tr, va, te = split_items(scenes, cfg["split"]["val"], cfg["split"]["test"], cfg["seed"])
        bucket = {u: "train" for u in tr} | {u: "val" for u in va} | {u: "test" for u in te}
    parts: dict[str, tuple[list[str], list[str]]] = {k: ([], []) for k in ("train", "val", "test")}
    for path, label, uid in rows:
        parts[bucket[uid]][0].append(path)
        parts[bucket[uid]][1].append(label)

    max_len = max(len(lab) for _, lab, _ in rows)
    size = tuple(cfg["img_size"])
    bs = cfg["train"]["batch_size"]
    nw = cfg["train"]["num_workers"]
    loaders = {}
    for name, (paths, labels) in parts.items():
        ds = STRDataset(
            paths, labels, vocab, max_len, build_transforms(size, train=name == "train")
        )
        loaders[name] = DataLoader(
            ds,
            batch_size=bs if name == "train" else bs * 2,
            shuffle=name == "train",
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
        )
        n_scenes = sum(1 for split in bucket.values() if split == name)
        log.info("%s: %d crops from %d scenes", name, len(ds), n_scenes)
    return loaders, max_len


def train(cfg: dict, device: str = "auto") -> dict:
    device = resolve_device(device)
    with start_tracking(cfg, "scene-text-recognition") as tracker:
        if tracker is not None:
            log_params(
                tracker,
                {
                    "seed": cfg["seed"],
                    "img_size": cfg["img_size"],
                    "model": cfg["model"],
                    "train": cfg["train"],
                    "device": device,
                },
                prefix="config",
            )
        return _train(cfg, device, tracker)


def _train(cfg: dict, device: str, tracker) -> dict:
    seed_everything(cfg["seed"])
    tc, mc = cfg["train"], cfg["model"]
    vocab = Vocab()
    loaders, max_len = make_loaders(cfg, vocab)
    img_size = tuple(cfg["img_size"])

    model_cfg = {k: v for k, v in mc.items() if k != "backbone_lr_mult"}
    model = CRNN(num_classes=vocab.num_classes, **model_cfg).to(device)
    criterion = nn.CTCLoss(blank=Vocab.BLANK, zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameter_groups(tc["lr"], mc.get("backbone_lr_mult", 0.1)),
        weight_decay=tc["weight_decay"],
    )
    step = max(1, int(tc["epochs"] * tc["lr_step_ratio"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.1)
    use_amp = tc["amp"] and device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    out_dir = Path(tc["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_cer, bad_epochs = float("inf"), 0

    for epoch in range(1, tc["epochs"] + 1):
        start = time.time()
        model.train()
        losses = []
        for images, target_ids, target_lens, _ in loaders["train"]:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                log_probs = model(images)
            log_probs = log_probs.float()  # CTC in fp32
            t, b, _ = log_probs.shape
            input_lens = torch.full((b,), t, dtype=torch.long)
            loss = criterion(log_probs, target_ids.to(device), input_lens, target_lens)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()

        val = evaluate(model, loaders["val"], criterion, vocab, device)
        row = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "val_loss": val["loss"],
            "val_acc": val["accuracy"],
            "val_cer": val["cer"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - start, 1),
        }
        history.append(row)
        if tracker is not None:
            log_metrics(
                tracker,
                {
                    "train/loss": row["train_loss"],
                    "validation/loss": row["val_loss"],
                    "validation/word_accuracy": row["val_acc"],
                    "validation/cer": row["val_cer"],
                    "learning_rate": row["lr"],
                    "epoch_seconds": row["seconds"],
                },
                step=epoch,
            )
        log.info(
            "epoch %3d | train %.4f | val %.4f | acc %.3f | CER %.3f | %.0fs",
            epoch,
            row["train_loss"],
            row["val_loss"],
            row["val_acc"],
            row["val_cer"],
            row["seconds"],
        )

        save_checkpoint(out_dir / "last.pt", model, vocab, model_cfg, img_size, row)
        if val["cer"] < best_cer:
            best_cer, bad_epochs = val["cer"], 0
            save_checkpoint(out_dir / "best.pt", model, vocab, model_cfg, img_size, row)
        else:
            bad_epochs += 1
            if bad_epochs >= tc["early_stopping_patience"]:
                log.info("early stopping (no CER improvement for %d epochs)", bad_epochs)
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    _plot_history(history, out_dir / "curves.png")

    best_model, _, _ = load_recognizer(out_dir / "best.pt", device)
    test = evaluate(best_model, loaders["test"], criterion, vocab, device)
    metrics = {k: test[k] for k in ("loss", "accuracy", "cer")}
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=1))
    if tracker is not None:
        log_metrics(
            tracker,
            {
                "test/loss": metrics["loss"],
                "test/word_accuracy": metrics["accuracy"],
                "test/cer": metrics["cer"],
            },
            step=len(history),
        )
        for artifact in (
            out_dir / "best.pt",
            out_dir / "history.json",
            out_dir / "curves.png",
            out_dir / "test_metrics.json",
        ):
            tracker.log_artifact(str(artifact))
    log.info(
        "TEST | loss %.4f | acc %.3f | CER %.3f",
        metrics["loss"],
        metrics["accuracy"],
        metrics["cer"],
    )
    return metrics


def _plot_history(history: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set(title="CTC loss", xlabel="epoch")
    ax[0].legend()
    ax[1].plot(ep, [h["val_acc"] for h in history], color="tab:green")
    ax[1].set(title="Val word accuracy", xlabel="epoch")
    ax[2].plot(ep, [h["val_cer"] for h in history], color="tab:red")
    ax[2].set(title="Val CER", xlabel="epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
