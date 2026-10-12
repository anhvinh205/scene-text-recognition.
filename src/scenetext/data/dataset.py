from __future__ import annotations

import torch
from PIL import Image
from torch.utils.data import Dataset

from scenetext.data.vocab import Vocab


class STRDataset(Dataset):
    def __init__(
        self, paths: list[str], labels: list[str], vocab: Vocab, max_len: int, transform=None
    ):
        if len(paths) != len(labels):
            raise ValueError("paths and labels must have the same length")
        self.paths, self.labels, self.vocab = paths, labels, vocab
        self.max_len, self.transform = max_len, transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as source:
            img = source.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        text = self.labels[idx]
        ids = self.vocab.encode(text)
        if len(ids) > self.max_len:
            raise ValueError(f"label length {len(ids)} exceeds max_len {self.max_len}")
        target = torch.zeros(self.max_len, dtype=torch.long)
        target[: len(ids)] = torch.tensor(ids, dtype=torch.long)
        return img, target, len(ids), text
