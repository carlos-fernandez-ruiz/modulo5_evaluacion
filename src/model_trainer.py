from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:
    from .data_loader_simple import TARGET_COLUMN, load_and_preprocess
    from .keras_neural_network import KerasNeuralNetwork
except ImportError:
    from data_loader_simple import TARGET_COLUMN, load_and_preprocess
    from keras_neural_network import KerasNeuralNetwork


RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

MODELS = ("logistic_regression", "decision_tree", "random_forest", "gradient_boosting", "support_vector_machine", "neural_network")
VARIANTS = ("full", "without_deposit_type", "without_deposit_type_and_parking")
TRAIN_VARIANTS_ORDER = ("full", "without_deposit_type", "without_deposit_type_and_parking")

GRID_SEARCH_SCORING = {
    "f1": "f1",
    "roc_auc": "roc_auc",
}
GRID_SEARCH_REFIT_METRIC = "f1"

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

# Columnas muy sesgadas a la derecha: log1p antes de escalar mejora LR y NN.
LOG_COLUMNS = [
    "lead_time",
    "days_in_waiting_list",
    "previous_cancellations",
    "previous_bookings_not_canceled",
]


@dataclass
class TrainingResult:

    model_name: str
    variant: str
    estimator: Any
    metrics: dict[str, float]
    confusion_matrix: np.ndarray
    classification_report_text: str
    feature_columns: list[str]
    scaled_columns: list[str]
    scaler: StandardScaler | None
    best_params: dict[str, Any] | None = None
    cv_best_score: float | None = None
    feature_ranking: pd.Series | None = None


def train(
    csv_path,
    model_name,
    variant="full",
    use_grid_search=False,
    param_grid=None,
):
   
    df = load_and_preprocess(csv_path)

    if variant == "all":
        return [
            train_from_dataframe(
                df,
                model_name=model_name,
                variant=current_variant,
                use_grid_search=use_grid_search,
                param_grid=param_grid,
            )
            for current_variant in TRAIN_VARIANTS_ORDER
        ]

    return train_from_dataframe(
        df,
        model_name=model_name,
        variant=variant,
        use_grid_search=use_grid_search,
        param_grid=param_grid,
    )


def train_from_dataframe(
    df,
    model_name,
    variant="full",
    use_grid_search=False,
    param_grid=None,
):
    validate_options(model_name, variant)

    df = apply_variant(df, variant)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = None
    scaled_columns: list[str] = []
    if requires_scaling(model_name):
        X_train, X_test, scaler, scaled_columns = scale_features(X_train, X_test)

    estimator, best_params, cv_best_score = fit_estimator(
        model_name,
        X_train,
        y_train,
        use_grid_search=use_grid_search,
        param_grid=param_grid,
    )

    return evaluate_model(
        estimator=estimator,
        model_name=model_name,
        variant=variant,
        X_test=X_test,
        y_test=y_test,
        feature_columns=X_train.columns.tolist(),
        scaled_columns=scaled_columns,
        scaler=scaler,
        best_params=best_params,
        cv_best_score=cv_best_score,
    )


def validate_options(model_name, variant):
    if model_name not in MODELS:
        raise ValueError(f"Modelo desconocido '{model_name}'. Available models: {list(MODELS)}")
    if variant not in VARIANTS:
        raise ValueError(f"Variante de entrenamiento desconocida '{variant}'. Variantes disponibles: {list(VARIANTS)}")


def apply_variant(df, variant):
    df = df.copy()

    if variant != "full":
        deposit_columns = [column for column in df.columns if column.startswith("deposit_type_")]
        df = df.drop(columns=deposit_columns)

    if variant == "without_deposit_type_and_parking":
        df = df.drop(columns="required_car_parking_spaces")

    return df


def build_estimator(model_name):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=500, C=10, solver="liblinear", random_state=RANDOM_STATE)

    if model_name == "decision_tree":
        return DecisionTreeClassifier(max_depth=14, min_samples_leaf=40, class_weight="balanced", random_state=RANDOM_STATE)

    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, min_samples_split=5, min_samples_leaf=2, max_features=0.5, max_depth=None, class_weight="balanced")

    if model_name == "gradient_boosting":
        return XGBClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    if model_name == "support_vector_machine":
        return CalibratedClassifierCV(estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000), method="sigmoid", cv=3)

    if model_name == "neural_network":
        return KerasNeuralNetwork()

    raise ValueError(f"Modelo desconocido '{model_name}'. Modelos disponibles: {list(MODELS)}")


def requires_scaling(model_name):
    return model_name in {"logistic_regression", "support_vector_machine", "neural_network"}


def default_param_grid(model_name):
    if model_name == "logistic_regression":
        return {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["liblinear"],
        }

    if model_name == "decision_tree":
        return {
            "criterion": ["gini", "entropy"],
            "max_depth": [6, 10, 14, None],
            "min_samples_leaf": [20, 40, 80],
            "class_weight": [None, "balanced"],
        }

    if model_name == "random_forest":
        return {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 2],
        }

    if model_name == "gradient_boosting":
        return {
            "n_estimators": [120, 180],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 4],
            "subsample": [0.8, 1.0],
        }

    if model_name == "support_vector_machine":
        return {
            "estimator__C": [0.1, 1, 10],
            "estimator__class_weight": [None, "balanced"],
        }

    if model_name == "neural_network":
        raise ValueError(
            "La red neuronal usa una configuración fija; "
            "explora hiperparámetros en el notebook, no con GridSearch."
        )

    raise ValueError(f"Modelo desconocido '{model_name}'. Modelos disponibles: {list(MODELS)}")


def scale_features(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    log_columns = [column for column in LOG_COLUMNS if column in X_train.columns]
    X_train[log_columns] = np.log1p(X_train[log_columns])
    X_test[log_columns] = np.log1p(X_test[log_columns])

    scaled_columns = [column for column in CONTINUOUS_COLUMNS if column in X_train.columns]

    scaler = StandardScaler()
    X_train[scaled_columns] = scaler.fit_transform(X_train[scaled_columns])
    X_test[scaled_columns] = scaler.transform(X_test[scaled_columns])

    return X_train, X_test, scaler, scaled_columns


def fit_estimator(
    model_name,
    X_train,
    y_train,
    use_grid_search=False,
    param_grid=None,
):
    estimator = build_estimator(model_name)

    if not use_grid_search:
        estimator.fit(X_train, y_train)
        return estimator, None, None

    grid = param_grid if param_grid is not None else default_param_grid(model_name)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        cv=CV_FOLDS,
        scoring=GRID_SEARCH_SCORING,
        refit=GRID_SEARCH_REFIT_METRIC,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate_model(
    estimator,
    model_name,
    variant,
    X_test,
    y_test,
    feature_columns,
    scaled_columns,
    scaler,
    best_params=None,
    cv_best_score=None,
):
    predictions = estimator.predict(X_test)
    probabilities = estimator.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    return TrainingResult(
        model_name=model_name,
        variant=variant,
        estimator=estimator,
        metrics=metrics,
        confusion_matrix=confusion_matrix(y_test, predictions),
        classification_report_text=classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        ),
        feature_columns=feature_columns,
        scaled_columns=scaled_columns,
        scaler=scaler,
        best_params=best_params,
        cv_best_score=cv_best_score,
        feature_ranking=rank_features(estimator, feature_columns),
    )


def rank_features(estimator, feature_names):
    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_)
        if coefficients.ndim == 2:
            coefficients = coefficients[0]
        ranking = pd.Series(coefficients, index=feature_names)
        return ranking.reindex(ranking.abs().sort_values(ascending=False).index)

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_)
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)

    return None
