from __future__ import annotations

from pathlib import Path

import torch

from scenetext.data.vocab import Vocab
from scenetext.models.crnn import CRNN


def save_checkpoint(
    checkpoint_path: str | Path,
    model: CRNN,
    vocab: Vocab,
    model_cfg: dict,
    img_size,
    metrics: dict,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "chars": vocab.chars,
            "model_cfg": model_cfg,
            "img_size": list(img_size),
            "metrics": metrics,
        },
        checkpoint_path,
    )


def load_recognizer(path: str | Path, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    vocab = Vocab(ckpt["chars"])
    cfg = dict(ckpt["model_cfg"])
    cfg["pretrained"] = False  # weights come from the checkpoint
    model = CRNN(num_classes=vocab.num_classes, **cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), vocab, tuple(ckpt["img_size"])
