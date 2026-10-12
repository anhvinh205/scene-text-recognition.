from __future__ import annotations

import string

DEFAULT_CHARS = string.digits + string.ascii_lowercase  # 0-9a-z


class Vocab:
    """``blank=0``; characters are mapped to ``1..len(chars)``.

    ``num_classes`` (= len(chars) + 1) is the model output dimension.
    """

    BLANK = 0

    def __init__(self, chars: str = DEFAULT_CHARS):
        if len(set(chars)) != len(chars):
            raise ValueError("duplicate characters in vocabulary")
        self.chars = chars
        self.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

    @property
    def num_classes(self) -> int:
        return len(self.chars) + 1

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[c] for c in text.lower()]

    def decode_indices(self, indices: list[int]) -> str:
        """Plain index -> string (no CTC collapsing)."""
        return "".join(self.idx_to_char.get(i, "") for i in indices if i != self.BLANK)

    @classmethod
    def from_labels(cls, labels: list[str]) -> Vocab:
        return cls("".join(sorted({c for lab in labels for c in lab.lower()})))
