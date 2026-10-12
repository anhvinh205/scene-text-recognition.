from __future__ import annotations

import torch
import torch.nn as nn

from scenetext.recognition.decode import greedy_decode
from scenetext.recognition.metrics import character_error_rate, sequence_accuracy


@torch.no_grad()
def evaluate(
    model, loader, criterion: nn.CTCLoss, vocab, device: str, max_examples: int = 10
) -> dict:
    model.eval()
    total_loss, n_batches = 0.0, 0
    preds: list[str] = []
    targets: list[str] = []
    for images, target_ids, target_lens, texts in loader:
        images = images.to(device)
        log_probs = model(images).float()  # (T, B, C)
        t, b, _ = log_probs.shape
        input_lens = torch.full((b,), t, dtype=torch.long)
        loss = criterion(log_probs, target_ids.to(device), input_lens, target_lens)
        total_loss += loss.item()
        n_batches += 1
        preds.extend(text for text, _ in greedy_decode(log_probs.cpu().numpy(), vocab))
        targets.extend(texts)
    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": sequence_accuracy(preds, targets),
        "cer": character_error_rate(preds, targets),
        "examples": list(zip(preds[:max_examples], targets[:max_examples], strict=True)),
    }
