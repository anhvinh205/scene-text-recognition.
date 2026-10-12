import json

from scenetext.data.crops import build_recognition_dataset, read_labels
from scenetext.data.icdar import WordBox, is_valid_label, parse_icdar_xml
from scenetext.data.splits import split_items
from scenetext.data.yolo_format import build_yolo_dataset, to_yolo_line


def test_label_filter():
    assert is_valid_label("Hello") and is_valid_label("a12")
    assert not is_valid_label("he's") and not is_valid_label("&") and not is_valid_label("")
    assert not is_valid_label("café") and not is_valid_label(None)


def test_parse_keeps_only_valid_words_and_drops_empty_scenes(icdar_dir):
    anns = parse_icdar_xml(icdar_dir)
    assert len(anns) == 2  # img2 only had '&'
    assert [b.text for b in anns[0].boxes] == ["hello", "a12"]
    assert anns[0].width == 200 and anns[0].height == 100


def test_uid_is_unique_across_folders(icdar_dir):
    uids = [a.uid for a in parse_icdar_xml(icdar_dir)]
    assert len(set(uids)) == len(uids) == 2


def test_yolo_line_is_center_normalised():
    line = to_yolo_line(WordBox(10, 20, 80, 30, "x"), 200, 100)
    cls, cx, cy, w, h = line.split()
    assert cls == "0"
    assert (float(cx), float(cy), float(w), float(h)) == (0.25, 0.35, 0.4, 0.3)


def test_split_is_deterministic_and_disjoint():
    items = list(range(100))
    a = split_items(items, 0.2, 0.1, seed=1)
    b = split_items(items, 0.2, 0.1, seed=1)
    assert a == b
    train, val, test = a
    assert (len(train), len(val), len(test)) == (70, 20, 10)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))


def test_build_yolo_dataset_structure(icdar_dir, tmp_path):
    anns = parse_icdar_xml(icdar_dir)
    out = tmp_path / "yolo"
    sizes = build_yolo_dataset(anns, out, val=0.0, test=0.5, seed=0)
    assert sum(sizes.values()) == 2
    assert (out / "data.yaml").exists()
    gt = json.loads((out / "annotations.json").read_text())
    for split in ("train", "val", "test"):
        for item in gt[split]:
            assert (out / item["image"]).exists()
            label_file = (
                out / split / "labels" / (item["image"].split("/")[-1].rsplit(".", 1)[0] + ".txt")
            )
            assert len(label_file.read_text().strip().splitlines()) == len(item["boxes"])


def test_build_recognition_dataset(icdar_dir, tmp_path):
    anns = parse_icdar_xml(icdar_dir)
    out = tmp_path / "ocr"
    n = build_recognition_dataset(anns, out, min_size=10, brightness_range=(35, 220))
    assert n == 3
    rows = read_labels(out)
    assert sorted(r[1] for r in rows) == ["a12", "hello", "world"]
    assert all(r[2] for r in rows)
