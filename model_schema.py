"""Общая схема признаков и версия metadata сохранённой модели."""

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

SCORE_CALIBRATION_METHOD = "training_baseline_min_max"
RISK_MEDIUM_THRESHOLD = 0.60
RISK_HIGH_THRESHOLD = 0.85

METADATA_VERSION = 2
