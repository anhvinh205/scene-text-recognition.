import io
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
os.environ["SKIP_MODEL_LOAD"] = "1"

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from scenetext.pipeline import TextRegion  # noqa: E402
from scenetext.serving.app import app  # noqa: E402

AUTH_HEADERS = {"Authorization": "Bearer test-key"}


class FakePipeline:
    device = "cpu"

    def predict(self, image, sort=True, min_rec_conf=0.0):
        return [TextRegion([5, 5, 50, 30], 0.91, "hello", 0.88)]


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_health_without_model(monkeypatch):
    monkeypatch.setenv("SCENETEXT_API_KEY", "test-key")
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["model_loaded"] is False
        assert (
            c.post(
                "/predict",
                files={"file": ("a.png", _png(), "image/png")},
                headers=AUTH_HEADERS,
            ).status_code
            == 503
        )


def test_predict_and_visualise_with_fake_pipeline(monkeypatch):
    monkeypatch.setenv("SCENETEXT_API_KEY", "test-key")
    with TestClient(app) as c:
        app.state.pipeline = FakePipeline()
        r = c.post("/predict", files={"file": ("a.png", _png(), "image/png")}, headers=AUTH_HEADERS)
        body = r.json()
        assert r.status_code == 200
        assert body["full_text"] == "hello" and body["num_regions"] == 1
        assert (body["width"], body["height"]) == (64, 48)
        img = c.post(
            "/predict/image", files={"file": ("a.png", _png(), "image/png")}, headers=AUTH_HEADERS
        )
        assert img.headers["content-type"] == "image/png"
        bad = c.post(
            "/predict",
            files={"file": ("a.txt", b"not an image", "text/plain")},
            headers=AUTH_HEADERS,
        )
        assert bad.status_code == 400
