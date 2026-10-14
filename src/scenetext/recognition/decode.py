"""CTC greedy and beam-search decoding."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

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
    arr = np.asarray(log_probs, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected (T, B, C), got {arr.shape}")
    if arr.shape[2] != vocab.num_classes:
        raise ValueError(f"expected {vocab.num_classes} classes for vocabulary, got {arr.shape[2]}")
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


# ---------------------------------------------------------------------------
# Lexicon / Language Model
# ---------------------------------------------------------------------------


class WordLexicon:
    """Dictionary and language model lexicon for beam search scoring and re-ranking."""

    def __init__(self, words: set[str] | Sequence[str] | None = None):
        if words is not None:
            self.words: set[str] = {w.strip().lower() for w in words if w.strip()}
        else:
            self.words = set()
        self.prefixes: set[str] = {w[:i] for w in self.words for i in range(1, len(w) + 1)}

    def __contains__(self, word: str) -> bool:
        return word.lower() in self.words

    def is_word(self, word: str) -> bool:
        return word.lower() in self.words

    def is_prefix(self, prefix: str) -> bool:
        return prefix.lower() in self.prefixes

    def score(self, text: str) -> float:
        """Score candidate text against the lexicon (higher is better)."""
        if not text:
            return 0.0
        t = text.lower()
        if t in self.words:
            return 2.0  # Exact dictionary word bonus
        if t in self.prefixes:
            return 0.5  # Valid word prefix bonus
        return 0.0

    @classmethod
    def from_file(cls, path: str | Path) -> WordLexicon:
        words = Path(path).read_text(encoding="utf-8").split()
        return cls(words)

    @classmethod
    def from_words(cls, words: Sequence[str]) -> WordLexicon:
        return cls(words)


# ---------------------------------------------------------------------------
# CTC prefix beam search
# ---------------------------------------------------------------------------


def _beam_search_single(
    log_probs_t: np.ndarray,
    beam_width: int,
    blank: int,
) -> list[tuple[tuple[int, ...], float]]:
    """CTC prefix beam search for a single sequence.

    Args:
        log_probs_t: ``(T, C)`` array of **log-probabilities**.
        beam_width: number of hypotheses to keep at each step.
        blank: index of the CTC blank label.

    Returns:
        Top-``beam_width`` prefixes with their log-probabilities, sorted best
        first. Each entry is ``(prefix_tuple_of_label_ids, log_prob)``.
    """
    T, C = log_probs_t.shape  # noqa: N806

    NEG_INF = -float("inf")  # noqa: N806

    # Each beam entry stores two log-probabilities:
    #   p_b  = prob of all alignments ending in blank
    #   p_nb = prob of all alignments ending in non-blank
    # Total prob = log_sum_exp(p_b, p_nb)

    beams: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [NEG_INF, NEG_INF])
    beams[()][0] = 0.0  # empty prefix, ends in blank

    for t in range(T):
        new_beams: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [NEG_INF, NEG_INF])
        # Prune: keep only top-k by total prob
        scored = [(prefix, _log_sum_exp(pb, pnb)) for prefix, (pb, pnb) in beams.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        kept = scored[:beam_width]

        for prefix, _ in kept:
            p_b, p_nb = beams[prefix]
            lp = log_probs_t[t]  # (C,)

            # --- extend with blank ---
            new_p_b = _log_sum_exp(new_beams[prefix][0], p_b + lp[blank])
            new_p_nb_stay = new_beams[prefix][1]
            new_beams[prefix] = [
                _log_sum_exp(new_p_b, p_nb + lp[blank]),
                new_p_nb_stay,
            ]

            # --- extend with each non-blank label ---
            for c in range(C):
                if c == blank:
                    continue
                new_prefix = prefix + (c,)

                if prefix and c == prefix[-1]:
                    # same char as last in prefix:
                    # can extend only via a blank step (p_b contributes)
                    new_beams[new_prefix][1] = _log_sum_exp(new_beams[new_prefix][1], p_b + lp[c])
                    # also allow continuing the *current* prefix
                    # (repeated same char, no collapse)
                    new_beams[prefix][1] = _log_sum_exp(new_beams[prefix][1], p_nb + lp[c])
                else:
                    new_beams[new_prefix][1] = _log_sum_exp(
                        new_beams[new_prefix][1],
                        _log_sum_exp(p_b, p_nb) + lp[c],
                    )

        beams = new_beams

    # Final ranking
    results = [(prefix, _log_sum_exp(pb, pnb)) for prefix, (pb, pnb) in beams.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:beam_width]


def _log_sum_exp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == -float("inf"):
        return b
    if b == -float("inf"):
        return a
    mx = max(a, b)
    return mx + math.log(math.exp(a - mx) + math.exp(b - mx))


def beam_search_decode(
    log_probs,
    vocab: Vocab,
    beam_width: int = 10,
    lexicon: WordLexicon | set[str] | Sequence[str] | None = None,
    lm_weight: float = 0.5,
) -> list[tuple[str, float]]:
    """Decode a batch of CTC outputs using prefix beam search.

    This is more accurate than :func:`greedy_decode` because it considers
    multiple alignment hypotheses.  When an optional *lexicon* is provided,
    hypotheses are scored combining acoustic likelihood and language / dictionary
    matching to correct subtle OCR errors.

    Args:
        log_probs: array/tensor shaped ``(T, B, C)`` — log-softmax output.
        vocab: vocabulary mapping indices back to characters.
        beam_width: number of beams to keep at each time step.  Larger values
            are more accurate but slower.  10-25 is a good range.
        lexicon: optional set of valid words or :class:`WordLexicon` for language scoring.
        lm_weight: weight for language lexicon score relative to acoustic log-probability.

    Returns:
        list of ``(text, confidence)`` for each item in the batch.
        Confidence is ``exp(log_prob / T)`` — the geometric-mean frame
        probability of the best beam, which is a better calibrated score
        than the greedy mean-max proxy.
    """
    arr = np.asarray(log_probs, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected (T, B, C), got {arr.shape}")
    if arr.shape[2] != vocab.num_classes:
        raise ValueError(f"expected {vocab.num_classes} classes for vocabulary, got {arr.shape[2]}")
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width < 1:
        raise ValueError(f"beam_width must be a positive integer, got {beam_width!r}")
    if not math.isfinite(lm_weight) or lm_weight < 0:
        raise ValueError(f"lm_weight must be a finite non-negative number, got {lm_weight!r}")

    T, B, _C = arr.shape  # noqa: N806

    # Normalise to log-probabilities (handles both logits and log-softmax)
    arr = arr - arr.max(axis=2, keepdims=True)
    log_p = arr - np.log(np.exp(arr).sum(axis=2, keepdims=True))

    lex: WordLexicon | None = (
        WordLexicon(lexicon)
        if lexicon is not None and not isinstance(lexicon, WordLexicon)
        else lexicon
    )

    results: list[tuple[str, float]] = []
    for b in range(B):
        top = _beam_search_single(log_p[:, b, :], beam_width, Vocab.BLANK)
        if not top:
            results.append(("", 0.0))
            continue

        if lex is None:
            best_prefix, best_lp = top[0]
            text = vocab.decode_indices(list(best_prefix))
            conf = float(math.exp(best_lp / T)) if (T > 0 and text) else 0.0
        else:
            # Re-rank candidates combining acoustic log-prob and lexicon bonus
            candidates: list[tuple[float, str, float]] = []
            for prefix, lp in top:
                cand_text = vocab.decode_indices(list(prefix))
                norm_lp = lp / T if T > 0 else lp
                lm_bonus = lex.score(cand_text)
                combined_score = norm_lp + lm_weight * lm_bonus
                candidates.append((combined_score, cand_text, lp))
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, text, best_lp = candidates[0]
            conf = float(math.exp(best_lp / T)) if (T > 0 and text) else 0.0

        results.append((text, conf))
    return results
