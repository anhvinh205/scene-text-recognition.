from __future__ import annotations

from collections.abc import Sequence


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (iterative DP, O(len(a)*len(b)) time, O(len(b)) memory)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sequence_accuracy(preds: Sequence[str], targets: Sequence[str]) -> float:
    """Exact-match (word) accuracy."""
    if not targets:
        return 0.0
    return sum(p == t for p, t in zip(preds, targets, strict=True)) / len(targets)


def character_error_rate(preds: Sequence[str], targets: Sequence[str]) -> float:
    """Total edit distance / total target characters."""
    total = sum(len(t) for t in targets)
    if total == 0:
        return 0.0
    dist = sum(edit_distance(p, t) for p, t in zip(preds, targets, strict=True))
    return dist / total
