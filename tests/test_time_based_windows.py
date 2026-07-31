import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
import pytest

from anomaly_detection import detect_anomalies
from model_schema import prepare_ml_features
from preprocessing import preprocess_data
from rule_config import RULE_PARAMS
from time_windows import causal_time_rolling


def _linear_growth(freq_seconds, duration_seconds=20 * 60, rate_c_per_min=0.2):
    offsets = np.arange(0, duration_seconds + 1, freq_seconds, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(offsets, unit="s"),
            "sensor_id": "T-LINEAR",
            "temperature": 70.0 + rate_c_per_min * offsets / 60.0,
        }
    )


def test_same_physical_growth_has_same_meaning_at_1_10_and_60_seconds():
    observations = []
    for frequency in (1, 10, 60):
        prepared = preprocess_data(_linear_growth(frequency))
        detected, _ = detect_anomalies(prepared, use_ml=False)
        last = detected.iloc[-1]
        observations.append(
            (last["rolling_mean"], last["temp_slope_overheat"])
        )
        assert last["rule_anomaly"] == 1
        assert "sustained_overheat" in last["triggered_rules"]

    means, slopes = zip(*observations)
    np.testing.assert_allclose(means, means[0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(slopes, slopes[0], rtol=0, atol=1e-3)


def test_irregular_timestamps_keep_physical_rate_and_overheat_slope():
    offsets = [0]
    steps = (30, 60, 90, 45)
    step_index = 0
    while offsets[-1] < 20 * 60:
        step = min(steps[step_index % len(steps)], 20 * 60 - offsets[-1])
        offsets.append(offsets[-1] + step)
        step_index += 1

    raw = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2026-01-01")
            + pd.to_timedelta(offsets, unit="s"),
            "sensor_id": "T-IRREGULAR",
            "temperature": 70.0 + 0.2 * np.asarray(offsets) / 60.0,
        }
    )
    prepared = preprocess_data(raw)
    detected, _ = detect_anomalies(prepared, use_ml=False)

    np.testing.assert_allclose(
        prepared["temp_rate_c_per_min"].iloc[1:],
        0.2,
        rtol=0,
        atol=1e-10,
    )
    assert detected["temp_slope_overheat"].iloc[-1] > RULE_PARAMS[
        "overheat_slope_c_per_min"
    ]
    assert "sustained_overheat" in detected["triggered_rules"].iloc[-1]


def test_large_time_gap_resets_stuck_and_rolling_history():
    before = pd.date_range("2026-01-01", periods=11, freq="1min")
    after = pd.DatetimeIndex([pd.Timestamp("2026-01-01 01:00:00")])
    raw = pd.DataFrame(
        {
            "timestamp": before.append(after),
            "sensor_id": "T-GAP",
            "temperature": [70.0] * len(before) + [90.0],
        }
    )

    prepared = preprocess_data(raw)
    assert prepared["is_stuck"].iloc[-2] == 1
    assert prepared["is_stuck"].iloc[-1] == 0
    assert prepared["rolling_mean"].iloc[-1] == 90.0
    assert prepared["rolling_std"].iloc[-1] == 1e-6
    assert np.isnan(prepared["temp_rate_c_per_min"].iloc[-1])

    with_ml_feature, _features, _eligible = prepare_ml_features(prepared)
    assert with_ml_feature["rolling_temp_diff_mean_20"].iloc[-1] == 0.0


def test_future_rows_do_not_change_past_features_or_rules():
    raw = _linear_growth(60, duration_seconds=29 * 60, rate_c_per_min=0.05)
    raw.loc[25:, "temperature"] += 50.0
    prefix = raw.iloc[:25].copy()

    prepared_prefix = preprocess_data(prefix)
    prepared_full = preprocess_data(raw)
    columns = ["rolling_mean", "rolling_std", "is_stuck", "stuck_score"]
    for column in columns:
        assert_series_equal(
            prepared_prefix[column].reset_index(drop=True),
            prepared_full[column].iloc[: len(prefix)].reset_index(drop=True),
            check_names=False,
        )

    detected_prefix, _ = detect_anomalies(prepared_prefix, use_ml=False)
    detected_full, _ = detect_anomalies(prepared_full, use_ml=False)
    for column in ("temp_slope_overheat", "rule_anomaly"):
        assert_series_equal(
            detected_prefix[column].reset_index(drop=True),
            detected_full[column].iloc[: len(prefix)].reset_index(drop=True),
            check_names=False,
        )


@pytest.mark.parametrize(
    "timestamps",
    [
        ["2026-01-01 00:00:00", "2026-01-01 00:00:00"],
        ["2026-01-01 00:01:00", "2026-01-01 00:00:00"],
    ],
)
def test_time_window_rejects_zero_and_negative_intervals(timestamps):
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "sensor_id": ["T-BAD", "T-BAD"],
            "value": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="нулевой или отрицательный интервал"):
        causal_time_rolling(frame, "value", 60, "mean", 120)


def test_one_minute_windows_match_legacy_feature_values():
    raw = _linear_growth(60, duration_seconds=29 * 60, rate_c_per_min=0.2)
    prepared = preprocess_data(raw)

    legacy_mean = prepared["temperature_filled"].rolling(
        window=10, min_periods=1
    ).mean()
    legacy_std = prepared["temperature_filled"].rolling(
        window=10, min_periods=1
    ).std().fillna(0).replace(0, 1e-6)
    np.testing.assert_allclose(prepared["rolling_mean"], legacy_mean)
    np.testing.assert_allclose(prepared["rolling_std"], legacy_std)

    with_ml_feature, _features, _eligible = prepare_ml_features(prepared)
    legacy_rate_mean = prepared["temp_diff"].rolling(
        window=20, min_periods=1
    ).mean()
    np.testing.assert_allclose(
        with_ml_feature["rolling_temp_diff_mean_20"],
        legacy_rate_mean,
    )

    detected, _ = detect_anomalies(prepared, use_ml=False)
    legacy_slope = prepared["rolling_mean"].rolling(
        window=20, min_periods=20
    ).apply(
        lambda values: np.polyfit(np.arange(len(values)), values, 1)[0],
        raw=True,
    )
    np.testing.assert_allclose(
        detected["temp_slope_overheat"],
        legacy_slope,
        equal_nan=True,
    )
