import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from data_quality import build_data_quality_report
from preprocessing import preprocess_data


def _measurements():
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:00",
                "2026-01-01 00:01:00",
            ],
            "sensor_id": ["T-01", "T-01", "T-01"],
            "temperature": [70.0, 70.0, 71.0],
            "scenario": ["normal", "normal", "normal"],
        }
    )


def test_identical_full_rows_are_collapsed_before_features():
    source = _measurements()
    original = source.copy(deep=True)

    result = preprocess_data(source)

    assert len(result) == 2
    assert result["temperature"].tolist() == [70.0, 71.0]
    assert result["time_diff_seconds"].isna().sum() == 1
    assert result["time_diff_seconds"].iloc[1] == 60.0
    assert_frame_equal(source, original)


def test_same_normalized_key_with_different_temperature_is_rejected():
    source = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-01 00:00:00"],
            "sensor_id": [" T-01 ", "T-01"],
            "temperature": [70.0, 72.0],
        }
    )

    with pytest.raises(ValueError, match="конфликтующие повторные измерения"):
        preprocess_data(source)


def test_duplicate_policy_is_independent_between_sensors():
    source = pd.DataFrame(
        {
            "timestamp": ["2026-01-01"] * 4,
            "sensor_id": ["T-01", "T-01", "T-02", "T-02"],
            "temperature": [70.0, 70.0, 80.0, 80.0],
        }
    )

    result = preprocess_data(source)

    assert result[["sensor_id", "temperature"]].values.tolist() == [
        ["T-01", 70.0],
        ["T-02", 80.0],
    ]


def test_exact_duplicate_result_is_stable_when_rows_are_reordered():
    source = _measurements()

    expected = preprocess_data(source)
    reordered = preprocess_data(source.sample(frac=1, random_state=7))

    assert_frame_equal(expected, reordered)


def test_conflict_is_rejected_for_both_row_orders():
    source = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-01"],
            "sensor_id": ["T-01", "T-01"],
            "temperature": [70.0, 72.0],
        }
    )

    for candidate in (source, source.iloc[::-1]):
        with pytest.raises(ValueError, match="Исправьте CSV"):
            preprocess_data(candidate)


def test_quality_report_explains_duplicate_action():
    report = build_data_quality_report(_measurements())

    assert report["status"] == "warning"
    assert report["duplicate_measurement_count"] == 1
    assert report["exact_duplicate_measurement_count"] == 1
    assert report["conflicting_duplicate_key_count"] == 0
    assert report["duplicate_measurement_policy"] == (
        "drop_exact_error_on_conflict"
    )
    assert "одинаковые строки" in report["duplicate_measurement_action"]
    assert any(
        "будет оставлена одна строка" in warning
        for warning in report["warnings"]
    )
