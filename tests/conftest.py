from pathlib import Path

import pytest
from PIL import Image

XML = """<?xml version="1.0" encoding="ISO-8859-1"?>
<tagset>
  <image>
    <imageName>folderA/img1.jpg</imageName>
    <resolution x="200" y="100" />
    <taggedRectangles>
      <taggedRectangle x="10.0" y="20.0" width="80.0" height="30.0" offset="0.0" rotation="0.0">
        <tag>Hello</tag><segmentation />
      </taggedRectangle>
      <taggedRectangle x="100.0" y="20.0" width="60.0" height="30.0" offset="0.0" rotation="0.0">
        <tag>he's</tag><segmentation />
      </taggedRectangle>
      <taggedRectangle x="100.0" y="60.0" width="60.0" height="30.0" offset="0.0" rotation="0.0">
        <tag>A12</tag><segmentation />
      </taggedRectangle>
    </taggedRectangles>
  </image>
  <image>
    <imageName>folderB/img1.jpg</imageName>
    <resolution x="200" y="100" />
    <taggedRectangles>
      <taggedRectangle x="0.0" y="0.0" width="50.0" height="30.0" offset="0.0" rotation="0.0">
        <tag>World</tag><segmentation />
      </taggedRectangle>
    </taggedRectangles>
  </image>
  <image>
    <imageName>folderB/img2.jpg</imageName>
    <resolution x="200" y="100" />
    <taggedRectangles>
      <taggedRectangle x="0.0" y="0.0" width="50.0" height="30.0" offset="0.0" rotation="0.0">
        <tag>&amp;</tag><segmentation />
      </taggedRectangle>
    </taggedRectangles>
  </image>
</tagset>
"""


@pytest.fixture()
def icdar_dir(tmp_path: Path) -> Path:
    """A tiny fake ICDAR2003 folder (2 valid scenes + 1 scene with only invalid labels)."""
    root = tmp_path / "SceneTrialTrain"
    for rel in ("folderA/img1.jpg", "folderB/img1.jpg", "folderB/img2.jpg"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (200, 100), (120, 140, 160)).save(p)
    (root / "words.xml").write_text(XML, encoding="iso-8859-1")
    return root
