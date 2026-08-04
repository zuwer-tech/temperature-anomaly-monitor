import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from preprocessing import preprocess_data
from sensor_alignment import causal_group_mean


def test_synchronous_measurements_match_exact_timestamp_mean():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:01:00",
                    "2026-01-01 00:01:00",
                ]
            ),
            "sensor_id": ["T-01", "T-02", "T-01", "T-02"],
            "value": [10.0, 14.0, 20.0, 24.0],
        }
    )

    group_mean, peer_count = causal_group_mean(frame, "value", 5)

    assert_series_equal(
        group_mean,
        pd.Series([12.0, 12.0, 22.0, 22.0]),
        check_names=False,
    )
    assert_series_equal(
        peer_count,
        pd.Series([1, 1, 1, 1]),
        check_names=False,
    )


def test_offsets_inside_tolerance_use_only_past_peers():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:01",
                    "2026-01-01 00:00:03",
                ]
            ),
            "sensor_id": ["T-01", "T-02", "T-03"],
            "value": [10.0, 12.0, 14.0],
        }
    )

    group_mean, peer_count = causal_group_mean(frame, "value", 5)

    np.testing.assert_allclose(group_mean, [10.0, 11.0, 12.0])
    np.testing.assert_array_equal(peer_count, [0, 1, 2])
    # Первая точка не получает будущие T-02 и T-03.
    assert group_mean.iloc[0] == frame["value"].iloc[0]


def test_measurement_outside_tolerance_does_not_participate():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 00:00:00", "2026-01-01 00:00:06"]
            ),
            "sensor_id": ["T-01", "T-02"],
            "value": [10.0, 14.0],
        }
    )

    group_mean, peer_count = causal_group_mean(frame, "value", 5)

    np.testing.assert_allclose(group_mean, [10.0, 14.0])
    np.testing.assert_array_equal(peer_count, [0, 0])


def test_missing_sensor_does_not_create_artificial_group_value():
    raw = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:01",
                "2026-01-01 00:00:01",
            ],
            "sensor_id": ["T-01", "T-02", "T-01", "T-02"],
            "temperature": [10.0, np.nan, 11.0, 20.0],
        }
    )

    prepared = preprocess_data(raw)
    early_measured = prepared[
        (prepared["sensor_id"] == "T-01")
        & (prepared["timestamp"] == pd.Timestamp("2026-01-01 00:00:00"))
    ].iloc[0]
    early_missing = prepared[
        (prepared["sensor_id"] == "T-02")
        & (prepared["timestamp"] == pd.Timestamp("2026-01-01 00:00:00"))
    ].iloc[0]

    # bfill используется для некоторых временных признаков, но групповое правило
    # не должно принять будущие 20 °C за реально измеренную раннюю пробу.
    assert early_missing["temperature_filled"] == 20.0
    assert np.isnan(early_missing["mean_temp_all_sensors"])
    assert early_missing["group_peer_count"] == 0
    assert early_measured["mean_temp_all_sensors"] == 10.0
    assert early_measured["group_peer_count"] == 0


def test_input_row_order_does_not_change_alignment():
    raw = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:01",
                "2026-01-01 00:00:03",
                "2026-01-01 00:01:00",
                "2026-01-01 00:01:02",
                "2026-01-01 00:01:04",
            ],
            "sensor_id": ["T-01", "T-02", "T-03", "T-01", "T-02", "T-03"],
            "temperature": [10.0, 12.0, 14.0, 20.0, 22.0, 24.0],
        }
    )

    expected = preprocess_data(raw)
    actual = preprocess_data(raw.sample(frac=1, random_state=42))
    columns = [
        "sensor_id",
        "timestamp",
        "mean_temp_all_sensors",
        "diff_from_group_mean",
        "abs_diff_from_group_mean",
        "group_peer_count",
    ]
    expected = expected[columns].sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    actual = actual[columns].sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    assert_frame_equal(actual, expected)


def test_single_sensor_keeps_rolling_mean_fallback():
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="1min"),
            "sensor_id": "T-ONLY",
            "temperature": [10.0, 11.0, 12.0],
        }
    )

    prepared = preprocess_data(raw)

    assert_series_equal(
        prepared["mean_temp_all_sensors"],
        prepared["rolling_mean"],
        check_names=False,
    )
    np.testing.assert_array_equal(prepared["group_peer_count"], [0, 0, 0])