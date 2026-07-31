import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
import pytest

from data_quality import build_data_quality_report
from preprocessing import preprocess_data, validate_input_data


def _base_frame():
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:01:00",
            ],
            "sensor_id": ["T-01", "T-01"],
            "temperature": [70.0, 71.0],
        }
    )


def test_naive_timestamps_remain_local_and_unzoned():
    result = preprocess_data(_base_frame())

    assert result["timestamp"].dt.tz is None
    assert result["time_diff_seconds"].iloc[1] == 60.0
    report = build_data_quality_report(_base_frame())
    assert report["timestamp_mode"] == "naive"
    assert report["timestamp_normalized_timezone"] is None
    assert "не указан" in report["timestamp_policy"]


def test_aware_offsets_are_normalized_to_utc_across_date_boundary():
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-02T00:30:00+03:00",
                "2026-01-01T22:00:00+00:00",
            ],
            "sensor_id": ["T-01", "T-01"],
            "temperature": [70.0, 71.0],
        }
    )

    result = preprocess_data(df)

    expected = pd.Series(
        pd.to_datetime(
            ["2026-01-01T21:30:00Z", "2026-01-01T22:00:00Z"],
            utc=True,
        ),
        name="timestamp",
    )
    assert_series_equal(result["timestamp"], expected)
    np.testing.assert_allclose(
        result["time_diff_seconds"], [np.nan, 1800.0], equal_nan=True
    )


def test_mixing_aware_and_naive_timestamps_is_rejected():
    df = _base_frame()
    df["timestamp"] = [
        "2026-01-01T00:00:00+03:00",
        "2026-01-01 00:01:00",
    ]

    with pytest.raises(ValueError, match="timezone-aware"):
        validate_input_data(df)

    report = build_data_quality_report(df)
    assert report["status"] == "error"
    assert report["timestamp_mode"] == "mixed"


def test_same_instant_with_different_offsets_is_one_exact_measurement():
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T03:00:00+03:00",
                "2026-01-01T00:00:00Z",
            ],
            "sensor_id": ["T-01", "T-01"],
            "temperature": [70.0, 70.0],
        }
    )

    report = build_data_quality_report(df)
    result = preprocess_data(df)

    assert report["duplicate_measurement_count"] == 1
    assert report["exact_duplicate_measurement_count"] == 1
    assert len(result) == 1
    assert str(result["timestamp"].dt.tz) == "UTC"


def test_base_csv_without_optional_metadata_remains_valid():
    assert validate_input_data(_base_frame()) is None


@pytest.mark.parametrize("unit", ["C", "°C", "degC", "celsius"])
def test_celsius_unit_aliases_are_accepted(unit):
    df = _base_frame().assign(temperature_unit=unit)

    assert validate_input_data(df) is None
    result = preprocess_data(df)
    assert result["temperature_unit"].eq(unit).all()


@pytest.mark.parametrize("unit", ["°F", "F", "K", "kelvin"])
def test_fahrenheit_and_kelvin_are_rejected_without_conversion(unit):
    df = _base_frame().assign(temperature_unit=unit)

    with pytest.raises(ValueError, match="Автоматический перевод"):
        validate_input_data(df)

    report = build_data_quality_report(df)
    assert report["status"] == "error"
    assert report["unsupported_temperature_units"] == [unit]


@pytest.mark.parametrize("accuracy", [-0.1, "unknown"])
def test_invalid_sensor_accuracy_is_rejected(accuracy):
    df = _base_frame().assign(sensor_accuracy=[0.5, accuracy])

    with pytest.raises(ValueError, match="sensor_accuracy"):
        validate_input_data(df)



def test_metadata_is_reported_but_does_not_become_an_ml_decision():
    df = _base_frame().assign(
        temperature_unit="°C",
        sensor_accuracy=[0.5, 0.5],
        quality_flag=["good", "suspect"],
    )

    result = preprocess_data(df)
    report = build_data_quality_report(df)

    assert result["quality_flag"].tolist() == ["good", "suspect"]
    assert report["temperature_units"] == ["°C"]
    assert report["sensor_accuracy_declared_count"] == 2
    assert report["quality_flags"] == ["good", "suspect"]
    assert report["status"] == "warning"
    assert any("калибровку" in warning for warning in report["warnings"])


def test_report_displays_source_offsets_and_utc_normalization():
    df = _base_frame()
    df["timestamp"] = [
        "2026-01-01T03:00:00+03:00",
        "2026-01-01T01:01:00+00:00",
    ]

    report = build_data_quality_report(df)

    assert report["timestamp_mode"] == "aware_utc"
    assert report["timestamp_normalized_timezone"] == "UTC"
    assert len(report["timestamp_source_timezones"]) == 2
    assert "UTC" in report["timestamp_policy"]
