from pydantic import BaseModel, Field


class Region(BaseModel):
    bbox: list[float] = Field(description="[x1, y1, x2, y2] in pixels")
    det_conf: float
    text: str
    rec_conf: float


class PredictResponse(BaseModel):
    width: int
    height: int
    num_regions: int
    full_text: str
    regions: list[Region]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str | None = None
