from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def split_items(
    items: Sequence[T], val: float, test: float, seed: int = 0
) -> tuple[list[T], list[T], list[T]]:
    """Shuffle deterministically and split into (train, val, test).

    Splitting is done at *item* level: pass source images (not crops) to avoid
    leaking the same scene between train and evaluation sets.
    """
    if not (0 <= val < 1 and 0 <= test < 1 and val + test < 1):
        raise ValueError("val and test must be in [0,1) with val + test < 1")
    items = list(items)
    random.Random(seed).shuffle(items)
    n_test = round(len(items) * test)
    n_val = round(len(items) * val)
    return items[n_test + n_val :], items[n_test : n_test + n_val], items[:n_test]
