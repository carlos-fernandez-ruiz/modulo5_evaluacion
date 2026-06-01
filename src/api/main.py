from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model_trainer import train as _train, MODELS, VARIANTS

from .model_store import save, load, list_models
from .predictor import predict as _predict


app = FastAPI(title="Api para entrenar modelos de ML y hacer predicciones")

SAMPLE_CSV_PATH = Path("data/dataset_practica_final.csv")
STATIC_DIR = Path(__file__).resolve().parent / "static"
NON_INPUT_COLUMNS = ["is_canceled", "reservation_status", "reservation_status_date"]

_sample_df: pd.DataFrame | None = None


class TrainRequest(BaseModel):
    csv_path: str
    model_name: str
    variant: str  # "full" | "without_deposit_type" | "without_deposit_type_and_parking" | "all"


class TrainedModel(BaseModel):
    model_id: str
    model_name: str
    variant: str
    metrics: dict[str, float]


class TrainResponse(BaseModel):
    trained: list[TrainedModel]


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]


@app.post("/train", response_model=TrainResponse)
def train_endpoint(req: TrainRequest):
    if req.model_name not in MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo no válido. Disponibles: {sorted(MODELS)}")
    if req.variant != "all" and req.variant not in VARIANTS:
        raise HTTPException(status_code=400, detail=f"Variante desconocida. Disponibles: {sorted(VARIANTS) + ['all']}")

    results = _train(req.csv_path, req.model_name, req.variant)
    if not isinstance(results, list):
        results = [results]

    trained = []
    for result in results:
        model_id = save(result)
        trained.append(TrainedModel(
            model_id=model_id,
            model_name=result.model_name,
            variant=result.variant,
            metrics={k: float(v) for k, v in result.metrics.items()},
        ))
    return TrainResponse(trained=trained)


@app.post("/predict/{model_id}", response_model=PredictResponse)
def predict_endpoint(model_id: str, req: PredictRequest):
    try:
        model_results = load(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    predictions, probabilities = _predict(model_results, req.rows)
    return PredictResponse(
        predictions=[int(p) for p in predictions],
        probabilities=[float(p) for p in probabilities],
    )


@app.get("/models")
def list_models_endpoint():
    return list_models()


@app.get("/sample")
def sample_endpoint():    
    global _sample_df
    if _sample_df is None:
        if not SAMPLE_CSV_PATH.exists():
            raise HTTPException(status_code=404, detail=f"CSV no encontrado en '{SAMPLE_CSV_PATH}'")
        _sample_df = pd.read_csv(SAMPLE_CSV_PATH)

    row = _sample_df.sample(n=1).iloc[0].to_dict()
    for column in NON_INPUT_COLUMNS:
        row.pop(column, None)
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}



app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
