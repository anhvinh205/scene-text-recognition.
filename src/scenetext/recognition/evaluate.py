from __future__ import annotations

import torch
import torch.nn as nn

from scenetext.recognition.decode import beam_search_decode, greedy_decode
from scenetext.recognition.metrics import character_error_rate, sequence_accuracy


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion: nn.CTCLoss,
    vocab,
    device: str,
    max_examples: int = 10,
    decode_mode: str = "greedy",
    beam_width: int = 10,
) -> dict:
    if decode_mode not in ("greedy", "beam"):
        raise ValueError(f"decode_mode must be 'greedy' or 'beam', got {decode_mode!r}")
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
        arr = log_probs.cpu().numpy()
        if decode_mode == "beam":
            batch_decoded = beam_search_decode(arr, vocab, beam_width=beam_width)
        else:
            batch_decoded = greedy_decode(arr, vocab)
        preds.extend(text for text, _ in batch_decoded)
        targets.extend(texts)
    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": sequence_accuracy(preds, targets),
        "cer": character_error_rate(preds, targets),
        "examples": list(zip(preds[:max_examples], targets[:max_examples], strict=True)),
    }
