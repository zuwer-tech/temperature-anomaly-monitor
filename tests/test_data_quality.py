import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from data_quality import build_data_quality_report


def test_clean_data_has_good_quality_status():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="1min"),
        "sensor_id": ["T-01", "T-01", "T-02", "T-02"],
        "temperature": [70.0, 71.0, 80.0, 81.0],
    })
    report = build_data_quality_report(df)
    assert report["status"] == "good"
    assert report["row_count"] == 4
    assert report["sensor_count"] == 2
    assert report["warnings"] == []


def test_report_counts_allowed_quality_limitations():
    df = pd.DataFrame({
        "timestamp": [
            "2026-01-01 00:01:00", "2026-01-01 00:00:00",
            "2026-01-01 00:00:00", "2026-01-01 00:00:00",
            "2026-01-01 00:01:00",
        ],
        "sensor_id": ["T-01", "T-01", "T-01", "T-EMPTY", "T-EMPTY"],
        "temperature": [70.0, np.nan, 72.0, np.nan, np.nan],
    })
    report = build_data_quality_report(df)
    assert report["status"] == "error"
    assert report["row_count"] == 5
    assert report["sensor_count"] == 2
    assert report["missing_temperature_count"] == 3
    assert report["missing_temperature_percent"] == 60.0
    assert report["duplicate_measurement_count"] == 1
    assert report["exact_duplicate_measurement_count"] == 0
    assert report["conflicting_duplicate_key_count"] == 1
    assert "останавливают анализ" in report["duplicate_measurement_action"]
    assert report["out_of_order_count"] == 1
    assert report["fully_missing_sensor_count"] == 1
    assert len(report["warnings"]) == 4


def test_invalid_values_have_error_status():
    df = pd.DataFrame({
        "timestamp": ["bad-time", "2026-01-01"],
        "sensor_id": ["T-01", "  "],
        "temperature": ["bad-temperature", 71.0],
    })
    report = build_data_quality_report(df)
    assert report["status"] == "error"
    assert report["invalid_timestamp_count"] == 1
    assert report["missing_sensor_id_count"] == 1
    assert report["non_numeric_temperature_count"] == 1


@pytest.mark.parametrize("column", ["timestamp", "sensor_id", "temperature"])
def test_missing_required_column_is_rejected(column):
    df = pd.DataFrame({
        "timestamp": ["2026-01-01"], "sensor_id": ["T-01"], "temperature": [70.0],
    }).drop(columns=column)
    with pytest.raises(ValueError, match=column):
        build_data_quality_report(df)


def test_report_does_not_modify_input_dataframe():
    df = pd.DataFrame({
        "timestamp": ["2026-01-01 00:01:00", "2026-01-01 00:00:00"],
        "sensor_id": ["T-01", "T-01"], "temperature": [70.0, np.nan],
    })
    original = df.copy(deep=True)
    build_data_quality_report(df)
    assert_frame_equal(df, original)
