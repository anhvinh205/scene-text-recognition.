"""CTC greedy decoding (collapse repeats, then drop blanks)."""

from __future__ import annotations

from collections.abc import Sequence

from scenetext.data.vocab import Vocab


def ctc_collapse(indices: Sequence[int], blank: int = 0) -> list[int]:
    """Collapse consecutive duplicates then remove blanks: [1,1,0,1,2,2] -> [1,1,2]."""
    out: list[int] = []
    prev = None
    for i in indices:
        if i != prev and i != blank:
            out.append(int(i))
        prev = i
    return out


def greedy_decode(log_probs, vocab: Vocab) -> list[tuple[str, float]]:
    """Decode a batch of CTC outputs.

    Args:
        log_probs: array/tensor shaped ``(T, B, C)`` (log-softmax or logits).
    Returns:
        list of ``(text, confidence)``; confidence is the mean max-probability over
        the non-blank frames (a cheap but useful proxy, in [0, 1]).
    """
    import numpy as np

    arr = np.asarray(log_probs, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected (T, B, C), got {arr.shape}")
    # normalise to probabilities in a numerically stable way
    arr = arr - arr.max(axis=2, keepdims=True)
    probs = np.exp(arr)
    probs /= probs.sum(axis=2, keepdims=True)

    best = probs.argmax(axis=2)  # (T, B)
    best_p = probs.max(axis=2)
    results = []
    for b in range(arr.shape[1]):
        seq = best[:, b].tolist()
        text = vocab.decode_indices(ctc_collapse(seq, Vocab.BLANK))
        keep = [best_p[t, b] for t, k in enumerate(seq) if k != Vocab.BLANK]
        conf = float(np.mean(keep)) if keep else 0.0
        results.append((text, conf))
    return results
