import pytest

from scenetext.evaluation.e2e import iou, match_boxes, prf


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_match_requires_text_when_asked():
    gts = [{"xyxy": [0, 0, 10, 10], "text": "Hello"}]
    good = [{"xyxy": [0, 0, 10, 10], "text": "hello"}]
    wrong = [{"xyxy": [0, 0, 10, 10], "text": "hellp"}]
    assert match_boxes(good, gts) == 1
    assert match_boxes(wrong, gts) == 0
    assert match_boxes(wrong, gts, require_text=False) == 1


def test_matching_is_one_to_one():
    gts = [{"xyxy": [0, 0, 10, 10], "text": "a"}]
    preds = [{"xyxy": [0, 0, 10, 10], "text": "a"}, {"xyxy": [0, 0, 10, 9], "text": "a"}]
    assert match_boxes(preds, gts) == 1


def test_prf():
    r = prf(tp=2, n_pred=4, n_gt=2)
    assert r["precision"] == 0.5 and r["recall"] == 1.0 and r["f1"] == pytest.approx(2 / 3)
    assert prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_reading_order_groups_lines():
    pytest.importorskip("torch")
    from scenetext.pipeline import TextRegion, reading_order

    def reg(x1, y1, text):
        return TextRegion([x1, y1, x1 + 40, y1 + 20], 0.9, text, 0.9)

    shuffled = [reg(200, 52, "c"), reg(0, 50, "b"), reg(100, 0, "a2"), reg(0, 2, "a1")]
    assert [r.text for r in reading_order(shuffled)] == ["a1", "a2", "b", "c"]
