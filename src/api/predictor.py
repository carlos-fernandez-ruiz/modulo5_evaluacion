import pandas as pd

from data_loader_simple import preprocess


def predict(bundle, raw_rows):
    """Predict on raw rows (list of dicts or DataFrame) using a saved Bundle."""
    if isinstance(raw_rows, list):
        raw_rows = pd.DataFrame(raw_rows)

    df = preprocess(raw_rows, drop_invalid_rows=False)
    df = df.reindex(columns=bundle.feature_columns, fill_value=0)

    if bundle.scaler is not None and bundle.scaled_columns:
        df[bundle.scaled_columns] = bundle.scaler.transform(df[bundle.scaled_columns])

    predictions = bundle.estimator.predict(df)
    probabilities = bundle.estimator.predict_proba(df)[:, 1]
    return predictions, probabilities
