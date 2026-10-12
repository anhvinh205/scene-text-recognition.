import numpy as np
import pytest

from scenetext.data.vocab import Vocab
from scenetext.recognition.decode import ctc_collapse, greedy_decode
from scenetext.recognition.metrics import character_error_rate, edit_distance, sequence_accuracy


def test_vocab_roundtrip():
    v = Vocab()
    assert v.num_classes == 37 and Vocab.BLANK == 0
    assert v.decode_indices(v.encode("abc123")) == "abc123"
    assert min(v.char_to_idx.values()) == 1  # 0 is reserved for blank


def test_vocab_rejects_duplicates():
    with pytest.raises(ValueError):
        Vocab("aab")


def test_ctc_collapse():
    assert ctc_collapse([1, 1, 0, 1, 2, 2]) == [1, 1, 2]
    assert ctc_collapse([0, 0, 0]) == []


def _logits_for(vocab, seq):
    lp = np.full((len(seq), 1, vocab.num_classes), -10.0)
    for t, i in enumerate(seq):
        lp[t, 0, i] = 0.0
    return lp


def test_greedy_decode_keeps_double_letters():
    """Regression: the course's decode() dropped the second 'l' of 'hello'."""
    v = Vocab()
    c = v.char_to_idx
    seq = [c["h"], c["h"], c["e"], c["l"], 0, c["l"], c["o"], c["o"]]
    ((text, conf),) = greedy_decode(_logits_for(v, seq), v)
    assert text == "hello" and 0.9 < conf <= 1.0


def test_greedy_decode_empty():
    v = Vocab()
    assert greedy_decode(_logits_for(v, [0, 0, 0]), v) == [("", 0.0)]


def test_greedy_decode_shape_check():
    with pytest.raises(ValueError):
        greedy_decode(np.zeros((3, 37)), Vocab())


def test_edit_distance_and_rates():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3 and edit_distance("abc", "abc") == 0
    assert character_error_rate(["abc"], ["abd"]) == pytest.approx(1 / 3)
    assert sequence_accuracy(["a", "b"], ["a", "c"]) == 0.5
