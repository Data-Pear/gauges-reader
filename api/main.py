from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from api.pipeline import GaugeReader, resolve_model_locations, settings_from_env

app = FastAPI(
    title="Gauge Reader API",
    version="0.1.0",
    description="Detect dial, keypoints, and needle mask, then return a normalized reading.",
)


@lru_cache(maxsize=1)
def get_reader() -> GaugeReader:
    return GaugeReader(settings_from_env())


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def models() -> Dict[str, Any]:
    try:
        return resolve_model_locations(settings_from_env())
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/read")
async def read_gauge(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await _predict_upload(file)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await _predict_upload(file)


async def _predict_upload(file: UploadFile) -> Dict[str, Any]:
    if file.content_type and not (
        file.content_type.startswith("image/")
        or file.content_type == "application/octet-stream"
    ):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {file.content_type}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        return get_reader().predict_bytes(image_bytes, image_name=file.filename)
    except (UnidentifiedImageError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
