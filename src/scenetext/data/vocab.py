from __future__ import annotations

import string

DEFAULT_CHARS = string.digits + string.ascii_lowercase  # 0-9a-z
ALPHANUMERIC_CHARS = string.digits + string.ascii_lowercase + string.ascii_uppercase  # 0-9a-zA-Z
PUNCTUATION_CHARS = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
EXTENDED_CHARS = string.digits + string.ascii_letters + PUNCTUATION_CHARS + " "


class Vocab:
    """Character vocabulary mapping between characters and model output indices.

    ``blank=0``; characters are mapped to ``1..len(chars)``.
    ``num_classes`` (= len(chars) + 1) is the model output dimension.
    """

    BLANK = 0
    DEFAULT_CHARS = DEFAULT_CHARS
    ALPHANUMERIC_CHARS = ALPHANUMERIC_CHARS
    PUNCTUATION_CHARS = PUNCTUATION_CHARS
    EXTENDED_CHARS = EXTENDED_CHARS

    def __init__(
        self,
        chars: str = DEFAULT_CHARS,
        unk_char: str | None = None,
        case_sensitive: bool | None = None,
    ):
        if unk_char is not None and unk_char not in chars:
            chars = chars + unk_char
        if len(set(chars)) != len(chars):
            raise ValueError("duplicate characters in vocabulary")
        self.chars = chars
        self.unk_char = unk_char
        if case_sensitive is None:
            self.case_sensitive = any(c.isupper() for c in chars)
        else:
            self.case_sensitive = case_sensitive

        self.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.unk_idx = self.char_to_idx.get(unk_char) if unk_char is not None else None

    @property
    def num_classes(self) -> int:
        return len(self.chars) + 1

    def encode(self, text: str, strict: bool = False) -> list[int]:
        """Convert a string into a list of class indices."""
        if not self.case_sensitive:
            text = text.lower()
        encoded: list[int] = []
        for c in text:
            if c in self.char_to_idx:
                encoded.append(self.char_to_idx[c])
            elif strict:
                raise KeyError(f"character {c!r} not in vocabulary: {self.chars!r}")
            elif self.unk_idx is not None:
                encoded.append(self.unk_idx)
        return encoded

    def decode_indices(self, indices: list[int]) -> str:
        """Plain index -> string (no CTC collapsing)."""
        return "".join(self.idx_to_char.get(i, "") for i in indices if i != self.BLANK)

    @classmethod
    def alphanumeric_lower(cls, unk_char: str | None = None) -> Vocab:
        return cls(DEFAULT_CHARS, unk_char=unk_char, case_sensitive=False)

    @classmethod
    def alphanumeric(cls, unk_char: str | None = None) -> Vocab:
        return cls(ALPHANUMERIC_CHARS, unk_char=unk_char, case_sensitive=True)

    @classmethod
    def extended(cls, unk_char: str | None = None) -> Vocab:
        return cls(EXTENDED_CHARS, unk_char=unk_char, case_sensitive=True)

    @classmethod
    def from_labels(cls, labels: list[str], case_sensitive: bool = False) -> Vocab:
        chars = sorted({c if case_sensitive else c.lower() for lab in labels for c in lab})
        return cls("".join(chars), case_sensitive=case_sensitive)
