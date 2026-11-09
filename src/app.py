"""FastAPI app serving the freight-rate model and its web UI.

Run from the project root with:
    python -m src.app
Then open http://localhost:8000 in a browser.

The first request trains the model (about a minute) and caches it in
models/freight_model.joblib; later requests are instant.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src import config, serving

app = FastAPI(title="Freight Rate Predictor", version="1.0.0")

_bundle: dict | None = None


def get_bundle() -> dict:
    global _bundle
    if _bundle is None:
        _bundle = serving.load_bundle()
    return _bundle


class LoadRequest(BaseModel):
    date: str = Field(..., examples=["2025-11-15"], description="Pickup date, YYYY-MM-DD")
    distance_miles: float = Field(..., gt=0, examples=[360])
    weight_lbs: float | None = Field(None, ge=0, examples=[32000])
    equipment: str = Field(..., examples=["Dry Van"])
    pickup_city: str = Field(..., examples=["Lexington"])
    delivery_city: str = Field(..., examples=["Fort Wayne"])


class PredictionResponse(BaseModel):
    predicted_rate: float
    inputs: dict


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.WEBAPP_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/meta")
def meta() -> dict:
    bundle = get_bundle()
    cities = sorted(bundle["city_table"].index.tolist())
    return {"cities": cities, "equipment_types": config.EQUIPMENT_TYPES}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(req: LoadRequest) -> PredictionResponse:
    bundle = get_bundle()
    try:
        pd.Timestamp(req.date)
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid date, expected YYYY-MM-DD")
    if req.equipment not in config.EQUIPMENT_TYPES:
        raise HTTPException(status_code=422, detail=f"equipment must be one of {config.EQUIPMENT_TYPES}")
    try:
        rate = serving.predict_load(
            bundle,
            req.date,
            req.distance_miles,
            req.weight_lbs,
            req.equipment,
            req.pickup_city,
            req.delivery_city,
        )
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    inputs = req.model_dump()
    return PredictionResponse(predicted_rate=round(rate, 2), inputs=inputs)


@app.get("/api/december")
def december() -> dict:
    bundle = get_bundle()
    df = serving.december_chart(bundle)
    records = df[["date", "equipment", "distance", "weight", "predicted_rate"]].copy()
    records["date"] = records["date"].dt.strftime("%Y-%m-%d")
    lane = df[["pickup", "delivery"]].iloc[0].to_dict()
    return {"lane": lane, "points": records.to_dict(orient="records")}


@app.get("/webapp/{file_path:path}")
def static_files(file_path: str) -> FileResponse:
    target = (config.WEBAPP_DIR / file_path).resolve()
    if not str(target).startswith(str(config.WEBAPP_DIR.resolve())) or not target.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(target)


def main() -> None:
    import uvicorn

    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
