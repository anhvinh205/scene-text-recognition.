"""FastAPI inference service.

Run: ``scenetext serve`` or ``uvicorn scenetext.serving.app:app``.
Configure with env var ``PIPELINE_CONFIG`` (default: configs/pipeline.yaml).
"""

from __future__ import annotations

import hmac
import io
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, UnidentifiedImageError

from scenetext.serving.schemas import HealthResponse, PredictResponse, Region
from scenetext.utils import get_logger, load_yaml

log = get_logger()
MAX_UPLOAD_BYTES = max(1, int(os.getenv("MAX_UPLOAD_MB", "10"))) * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
bearer_scheme = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = None
    if os.getenv("SKIP_MODEL_LOAD") != "1":
        from scenetext.pipeline import OCRPipeline

        cfg = load_yaml(os.getenv("PIPELINE_CONFIG", "configs/pipeline.yaml"))
        app.state.pipeline = OCRPipeline.from_config(cfg)
        log.info("pipeline loaded on %s", app.state.pipeline.device)
    yield


app = FastAPI(
    title="Scene Text Recognition API",
    version="0.1.0",
    description="YOLO26 text detection + CRNN text recognition.",
    lifespan=lifespan,
)


def get_pipeline(request: Request):
    pipe = request.app.state.pipeline
    if pipe is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return pipe


def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    expected_key = os.getenv("SCENETEXT_API_KEY")
    if not expected_key:
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    if credentials is None or not hmac.compare_digest(credentials.credentials, expected_key):
        raise HTTPException(
            status_code=401,
            detail="invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def _read_image(file: UploadFile) -> Image.Image:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="image dimensions too large")
            return image.convert("RGB")
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="invalid image file") from exc


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    pipe = request.app.state.pipeline
    return HealthResponse(
        status="ok", model_loaded=pipe is not None, device=getattr(pipe, "device", None)
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    min_rec_conf: float = 0.0,
    _auth: None = Depends(verify_api_key),
    pipeline=Depends(get_pipeline),
) -> PredictResponse:
    img = await _read_image(file)
    start = time.perf_counter()
    regions = pipeline.predict(img, min_rec_conf=min_rec_conf)
    latency = (time.perf_counter() - start) * 1000
    return PredictResponse(
        width=img.width,
        height=img.height,
        num_regions=len(regions),
        full_text=" ".join(r.text for r in regions),
        regions=[Region(**r.to_dict()) for r in regions],
        latency_ms=round(latency, 1),
    )


@app.post("/predict/image", responses={200: {"content": {"image/png": {}}}})
async def predict_image(
    file: UploadFile = File(...),
    _auth: None = Depends(verify_api_key),
    pipeline=Depends(get_pipeline),
) -> Response:
    from scenetext.pipeline import draw_regions

    img = await _read_image(file)
    out = draw_regions(img, pipeline.predict(img))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
