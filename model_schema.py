"""Общая схема признаков, подготовка ML-входа и версия metadata модели."""

import numpy as np
import pandas as pd


FEATURE_COLUMNS = (
    "temperature_filled",
    "rolling_mean",
    "rolling_std",
    "temp_diff",
    "abs_temp_diff",
    "abs_z_score",
    "is_missing",
    "is_stuck",
    "abs_diff_from_group_mean",
    "rolling_temp_diff_mean_20",
)

ML_STATUS_APPLIED = "applied"
ML_STATUS_SKIPPED_MISSING_TEMPERATURE = "skipped_missing_temperature"
ML_STATUS_NOT_APPLIED_RULES_ONLY = "not_applied_rules_only"

SCORE_CALIBRATION_METHOD = "training_baseline_min_max"
RISK_MEDIUM_THRESHOLD = 0.60
RISK_HIGH_THRESHOLD = 0.85

METADATA_VERSION = 2


def prepare_ml_features(df):
    """Единообразно готовит признаки для обучения и инференса.

    Строки без реального измерения temperature исключаются из ML: потерю
    сигнала объясняет инженерное правило. Структурные значения первой точки
    (например, temp_diff == 0) формируются в preprocessing.py. Остаточные
    NaN/inf в измеренных строках не маскируются нулём, а вызывают явную ошибку.
    """
    required_columns = {
        "timestamp",
        "sensor_id",
        "temperature",
        "temp_diff",
        *FEATURE_COLUMNS[:-1],
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "Невозможно подготовить ML-признаки: отсутствуют колонки "
            f"{', '.join(missing_columns)}."
        )

    prepared = df.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])
    prepared = prepared.sort_values(
        by=["sensor_id", "timestamp"]
    ).reset_index(drop=True)
    prepared["rolling_temp_diff_mean_20"] = (
        prepared.groupby("sensor_id")["temp_diff"]
        .transform(lambda values: values.rolling(window=20, min_periods=1).mean())
    )

    temperature = pd.to_numeric(prepared["temperature"], errors="coerce")
    invalid_temperature = prepared["temperature"].notna() & temperature.isna()
    if invalid_temperature.any():
        raise ValueError(
            "Невозможно подготовить ML-признаки: temperature содержит "
            "нечисловые значения."
        )

    ml_eligible = temperature.notna()
    features = prepared[list(FEATURE_COLUMNS)].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    invalid_features = features.loc[ml_eligible].isna()
    if invalid_features.any().any():
        columns = invalid_features.columns[invalid_features.any()].tolist()
        raise ValueError(
            "Измеренные строки содержат пустые или бесконечные ML-признаки: "
            f"{', '.join(columns)}. Исправьте предобработку вместо fillna(0)."
        )

    return prepared, features.loc[ml_eligible].astype(float), ml_eligible