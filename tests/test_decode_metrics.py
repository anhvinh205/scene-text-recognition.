import numpy as np
import pytest

from scenetext.data.vocab import Vocab
from scenetext.recognition.decode import (
    WordLexicon,
    beam_search_decode,
    ctc_collapse,
    greedy_decode,
)
from scenetext.recognition.metrics import character_error_rate, edit_distance, sequence_accuracy


def test_vocab_roundtrip():
    v = Vocab()
    assert v.num_classes == 37 and Vocab.BLANK == 0
    assert v.decode_indices(v.encode("abc123")) == "abc123"
    assert min(v.char_to_idx.values()) == 1  # 0 is reserved for blank


def test_vocab_rejects_duplicates():
    with pytest.raises(ValueError):
        Vocab("aab")


def test_vocab_presets_and_case_sensitivity():
    v_upper = Vocab.alphanumeric()
    assert v_upper.num_classes == 63  # 62 chars + 1 blank
    assert v_upper.case_sensitive is True
    assert v_upper.decode_indices(v_upper.encode("Hello123")) == "Hello123"

    v_ext = Vocab.extended()
    assert v_ext.num_classes == 96  # 94 chars + space + 1 blank
    assert v_ext.decode_indices(v_ext.encode("Price: $10.99!")) == "Price: $10.99!"


def test_vocab_unk_token():
    v = Vocab("abc", unk_char="?")
    assert v.unk_idx == v.char_to_idx["?"]
    # Unknown characters are mapped to unk_idx
    assert v.decode_indices(v.encode("axbyc")) == "a?b?c"
    # Strict mode raises KeyError
    with pytest.raises(KeyError):
        v.encode("axb", strict=True)


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


def test_beam_search_decode_keeps_double_letters():
    v = Vocab()
    c = v.char_to_idx
    seq = [c["h"], c["h"], c["e"], c["l"], 0, c["l"], c["o"], c["o"]]
    ((text, conf),) = beam_search_decode(_logits_for(v, seq), v, beam_width=5)
    assert text == "hello" and 0.5 < conf <= 1.0


def test_beam_search_decode_empty():
    v = Vocab()
    assert beam_search_decode(_logits_for(v, [0, 0, 0]), v) == [("", 0.0)]


def test_beam_search_decode_shape_check():
    with pytest.raises(ValueError):
        beam_search_decode(np.zeros((3, 37)), Vocab())


@pytest.mark.parametrize("beam_width", [0, -1])
def test_beam_search_rejects_non_positive_width(beam_width):
    with pytest.raises(ValueError, match="beam_width"):
        beam_search_decode(_logits_for(Vocab(), [1]), Vocab(), beam_width=beam_width)


def test_beam_search_rejects_invalid_lm_weight():
    with pytest.raises(ValueError, match="lm_weight"):
        beam_search_decode(_logits_for(Vocab(), [1]), Vocab(), lm_weight=-0.1)


def test_decoders_reject_vocabulary_shape_mismatch():
    with pytest.raises(ValueError, match="classes for vocabulary"):
        greedy_decode(np.zeros((3, 1, 36)), Vocab())
    with pytest.raises(ValueError, match="classes for vocabulary"):
        beam_search_decode(np.zeros((3, 1, 36)), Vocab())


def test_word_lexicon_scoring():
    lex = WordLexicon(["hello", "world"])
    assert "hello" in lex
    assert lex.is_word("HELLO")
    assert lex.is_prefix("hel")
    assert not lex.is_prefix("xyz")
    assert lex.score("hello") == 2.0
    assert lex.score("hel") == 0.5
    assert lex.score("random") == 0.0


def test_beam_search_with_lexicon_corrects_ocr_error():
    v = Vocab()
    c = v.char_to_idx
    # Construct sequence where 'hell0' has slightly higher acoustic score than 'hello'
    seq = [c["h"], c["e"], c["l"], 0, c["l"]]
    lp = np.full((6, 1, v.num_classes), -10.0)
    for t, idx in enumerate(seq):
        lp[t, 0, idx] = 0.0
    lp[5, 0, c["0"]] = 0.0
    lp[5, 0, c["o"]] = -0.05

    res_no_lex = beam_search_decode(lp, v, beam_width=5)
    assert res_no_lex[0][0] == "hell0"

    lex = WordLexicon(["hello"])
    res_with_lex = beam_search_decode(lp, v, beam_width=5, lexicon=lex, lm_weight=0.5)
    assert res_with_lex[0][0] == "hello"


def test_edit_distance_and_rates():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3 and edit_distance("abc", "abc") == 0
    assert character_error_rate(["abc"], ["abd"]) == pytest.approx(1 / 3)
    assert sequence_accuracy(["a", "b"], ["a", "c"]) == 0.5
