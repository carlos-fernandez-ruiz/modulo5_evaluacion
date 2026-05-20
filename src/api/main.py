from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_trainer import train as _train, MODELS, VARIANTS

from .model_store import save, load, list_models
from .predictor import predict as _predict


app = FastAPI(title="Api para entrenar modelos de ML y hacer predicciones")


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
