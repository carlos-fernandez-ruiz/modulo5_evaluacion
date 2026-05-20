from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMN = "is_canceled"
MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def load_and_preprocess(csv_path):
    return preprocess(pd.read_csv(Path(csv_path)))


def preprocess(df_raw, drop_invalid_rows=True):
    df = df_raw.copy()

    df = df.drop(columns=["reservation_status", "reservation_status_date"])
    df["children"] = df["children"].fillna(0).astype("int64")
    df["has_agent"] = df["agent"].notna().astype(int)
    df["has_company"] = df["company"].notna().astype(int)
    df = df.drop(columns=["agent", "company"])

    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    if drop_invalid_rows:
        df = df[(df["total_guests"] > 0) & (df["total_nights"] > 0)].copy()
        df = df[(df["adr"] >= 0) & (df["adr"] < 5000)].copy()

    df = df.drop(columns=["assigned_room_type"])
    df["is_resort_hotel"] = (df["hotel"] == "Resort Hotel").astype(int)
    df = df.drop(columns="hotel")

    df["market_segment"] = df["market_segment"].replace("Undefined", "Online TA")
    df["distribution_channel"] = df["distribution_channel"].replace("Undefined", "TA/TO")
    df = df.drop(columns=["distribution_channel", "country"])

    df = _one_hot(df, "market_segment")
    df = _one_hot(df, "customer_type")
    df = _one_hot(df, "reserved_room_type")

    df["arrival_date_month"] = df["arrival_date_month"].map(MONTH_MAP)
    df["arrival_month_sin"] = np.sin(2 * np.pi * df["arrival_date_month"] / 12)
    df["arrival_month_cos"] = np.cos(2 * np.pi * df["arrival_date_month"] / 12)
    df = df.drop(columns="arrival_date_month")

    df["adr_per_person"] = (df["adr"] / df["total_guests"]).replace([np.inf, -np.inf], 0).fillna(0)
    df["adr_per_night"] = (df["adr"] / df["total_nights"]).replace([np.inf, -np.inf], 0).fillna(0)

    df = _one_hot(df, "meal")
    df = _one_hot(df, "deposit_type")

    assert df.isna().sum().sum() == 0, "preprocessed frame has NaNs"
    return df


def _one_hot(df, column):
    dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
    return pd.concat([df.drop(columns=column), dummies], axis=1)
