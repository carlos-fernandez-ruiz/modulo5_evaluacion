from collections import namedtuple
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from .data_loader_simple import TARGET_COLUMN, load_and_preprocess
except ImportError:
    from data_loader_simple import TARGET_COLUMN, load_and_preprocess

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
DEFAULT_SCORING = {
    "f1": "f1",
    "roc_auc": "roc_auc",
}

DEFAULT_REFIT_METRIC = "f1"

VARIANTS = {"full", "without_deposit_type", "without_deposit_type_and_parking"}

CONTINUOUS_COLUMNS = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "adr_per_person",
    "adr_per_night",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "total_guests",
    "total_nights",
]

Model = namedtuple("Model", "factory requires_scaling param_grid")

MODELS = {
    "logistic_regression": Model(
        factory=lambda: LogisticRegression(max_iter=500, C=10, solver="liblinear", random_state=RANDOM_STATE),
        requires_scaling=True,
        param_grid={"C": [0.01, 0.1, 1, 10], "solver": ["liblinear"]},
    ),
    "random_forest": Model(
        factory=lambda: RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, min_samples_split=5, min_samples_leaf=2, max_features=0.5, max_depth=None, class_weight="balanced"),
        requires_scaling=False,
        param_grid=None,
    ),
    "neural_network": Model(
        factory=lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_STATE),
        requires_scaling=True,
        param_grid=None,
    ),
}


def _get_model(model_name):
    try:
        return MODELS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Modelo desconocido '{model_name}'. Modelos disponibles: {sorted(MODELS)}"
        ) from exc


@dataclass
class PreparedDataset:
    variant: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: list[str]
    scaled_columns: list[str]
    scaler: StandardScaler | None


@dataclass
class TrainingResult:
    model_name: str
    variant: str
    estimator: BaseEstimator
    search: GridSearchCV | None
    best_params: dict[str, Any] | None
    cv_best_score: float | None
    metrics: dict[str, float]
    confusion_matrix: np.ndarray
    classification_report_text: str
    classification_report_dict: dict[str, Any]
    predictions: pd.Series
    prediction_probabilities: np.ndarray
    feature_ranking: pd.Series | None
    feature_columns: list[str]
    scaled_columns: list[str]
    scaler: StandardScaler | None


TRAIN_VARIANTS_ORDER = ("full", "without_deposit_type", "without_deposit_type_and_parking")


def train(csv_path, model_name, variant):
    """API entry point. Returns a TrainingResult, or a list of them when variant='all'."""
    df = load_and_preprocess(csv_path)
    if variant == "all":
        return [_train_one(df, model_name, v) for v in TRAIN_VARIANTS_ORDER]
    return _train_one(df, model_name, variant)


def _train_one(df, model_name, variant):
    prepared = prepare_dataset(df, variant=variant, model_name=model_name)
    return train_classifier(prepared, model_name=model_name)


def prepare_dataset(
    df,
    variant="full",
    model_name="logistic_regression",
    target_column=TARGET_COLUMN,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    scale=None,
):
    if variant not in VARIANTS:
        raise ValueError(f"Variante de entrenamiento desconocida '{variant}'. Variantes disponibles: {sorted(VARIANTS)}")
    model = _get_model(model_name)

    df = df.copy()
    if variant != "full":
        df = df.drop(columns=[c for c in df.columns if c.startswith("deposit_type_")])
    if variant == "without_deposit_type_and_parking":
        df = df.drop(columns="required_car_parking_spaces")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    should_scale = model.requires_scaling if scale is None else scale
    scaled_columns: list[str] = []
    scaler: StandardScaler | None = None

    if should_scale:
        scaled_columns = [c for c in CONTINUOUS_COLUMNS if c in X_train.columns]
        X_train, X_test, scaler = scale_continuous_features(X_train, X_test, scaled_columns)

    return PreparedDataset(
        variant=variant,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_columns=X_train.columns.tolist(),
        scaled_columns=scaled_columns,
        scaler=scaler,
    )


def scale_continuous_features(X_train, X_test, columns):
    X_train = X_train.copy()
    X_test = X_test.copy()
    scaler = StandardScaler()
    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns] = scaler.transform(X_test[columns])
    return X_train, X_test, scaler


def train_classifier(
    prepared,
    model_name="logistic_regression",
    param_grid=None,
    cv=CV_FOLDS,
    scoring=DEFAULT_SCORING,
    use_grid_search=None,
):
    model = _get_model(model_name)
    estimator = model.factory()
    grid = param_grid if param_grid is not None else model.param_grid
    should_search = use_grid_search if use_grid_search is not None else grid is not None
    search: GridSearchCV | None = None

    if should_search and grid is not None:
        search = GridSearchCV(estimator=estimator, param_grid=grid, cv=cv, scoring=scoring, refit=DEFAULT_REFIT_METRIC)
        search.fit(prepared.X_train, prepared.y_train)
        fitted = search.best_estimator_
    else:
        fitted = estimator
        fitted.fit(prepared.X_train, prepared.y_train)

    return evaluate_classifier(fitted, prepared, model_name=model_name, search=search)


def evaluate_classifier(estimator, prepared, model_name, search=None):
    pred_array = estimator.predict(prepared.X_test)
    predictions = pd.Series(pred_array, index=prepared.y_test.index, name="prediction")
    proba = estimator.predict_proba(prepared.X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(prepared.y_test, predictions),
        "precision": precision_score(prepared.y_test, predictions, zero_division=0),
        "recall":    recall_score(prepared.y_test, predictions, zero_division=0),
        "f1":        f1_score(prepared.y_test, predictions, zero_division=0),
        "roc_auc":   roc_auc_score(prepared.y_test, proba),
    }

    report_dict = classification_report(
        prepared.y_test, predictions, digits=4, output_dict=True, zero_division=0,
    )
    report_text = classification_report(
        prepared.y_test, predictions, digits=4, zero_division=0,
    )

    return TrainingResult(
        model_name=model_name,
        variant=prepared.variant,
        estimator=estimator,
        search=search,
        best_params=search.best_params_ if search is not None else None,
        cv_best_score=search.best_score_ if search is not None else None,
        metrics=metrics,
        confusion_matrix=confusion_matrix(prepared.y_test, predictions),
        classification_report_text=report_text,
        classification_report_dict=report_dict,
        predictions=predictions,
        prediction_probabilities=proba,
        feature_ranking=rank_features(estimator, prepared.feature_columns),
        feature_columns=prepared.feature_columns,
        scaled_columns=prepared.scaled_columns,
        scaler=prepared.scaler,
    )


def rank_features(estimator, feature_names):
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        if coef.ndim == 2:
            coef = coef[0]
        series = pd.Series(coef, index=feature_names)
        return series.reindex(series.abs().sort_values(ascending=False).index)

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_)
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)

    return None
