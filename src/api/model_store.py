import json
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import joblib

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

ModelResults = namedtuple("ModelResults", "estimator scaler feature_columns scaled_columns metadata")


def save(result, store_dir=MODELS_DIR):
    """Guarda un modelo entrenado en el filesystem, con un ID único basado en su nombre, variante y timestamp."""
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    model_id = f"{result.model_name}__{result.variant}__{timestamp}"
    path = Path(store_dir) / model_id
    path.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.estimator, path / "estimator.joblib")
    if result.scaler is not None:
        joblib.dump(result.scaler, path / "scaler.joblib")

    metadata = {
        "model_id": model_id,
        "model_name": result.model_name,
        "variant": result.variant,
        "feature_columns": result.feature_columns,
        "scaled_columns": result.scaled_columns,
        "metrics": result.metrics,
        "created_at": timestamp,
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return model_id


def load(model_id, store_dir=MODELS_DIR):
    path = Path(store_dir) / model_id
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado '{model_id}' under {store_dir}")
    metadata = json.loads((path / "metadata.json").read_text())
    estimator = joblib.load(path / "estimator.joblib")
    scaler_path = path / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return ModelResults(
        estimator=estimator,
        scaler=scaler,
        feature_columns=metadata["feature_columns"],
        scaled_columns=metadata["scaled_columns"],
        metadata=metadata,
    )


def list_models(store_dir=MODELS_DIR):
    store_dir = Path(store_dir)
    if not store_dir.exists():
        return []
    return [
        json.loads((p / "metadata.json").read_text())
        for p in sorted(store_dir.iterdir())
        if (p / "metadata.json").exists()
    ]
