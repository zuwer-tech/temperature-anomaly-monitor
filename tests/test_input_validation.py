import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from preprocessing import preprocess_data, validate_input_data


def _valid_dataframe():
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:00:00", "2026-01-01 00:01:00"],
            "sensor_id": ["T-01", "T-01"],
            "temperature": [70.0, 71.0],
        }
    )


def test_valid_dataframe_passes():
    assert validate_input_data(_valid_dataframe()) is None


def test_preprocess_data_accepts_valid_dataframe():
    result = preprocess_data(_valid_dataframe())

    assert len(result) == 2
    assert "rolling_mean" in result.columns


def test_non_dataframe_raises_type_error():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        validate_input_data([])


def test_one_required_column_missing():
    df = _valid_dataframe().drop(columns="sensor_id")

    with pytest.raises(ValueError, match="Missing required columns: sensor_id"):
        validate_input_data(df)


def test_multiple_required_columns_missing_in_stable_order():
    df = pd.DataFrame({"temperature": [70.0]})

    with pytest.raises(ValueError) as error:
        validate_input_data(df)

    message = str(error.value)
    assert "Missing required columns" in message
    assert message.index("timestamp") < message.index("sensor_id")


def test_empty_dataframe_raises_value_error():
    df = _valid_dataframe().iloc[0:0]

    with pytest.raises(ValueError, match="at least one row"):
        validate_input_data(df)


def test_invalid_timestamp_raises_value_error():
    df = _valid_dataframe()
    df.loc[1, "timestamp"] = "not-a-date"

    with pytest.raises(ValueError, match=r"timestamp contains 1"):
        validate_input_data(df)


def test_missing_timestamp_raises_value_error():
    df = _valid_dataframe()
    df.loc[0, "timestamp"] = None

    with pytest.raises(ValueError, match=r"timestamp contains 1"):
        validate_input_data(df)


def test_missing_sensor_id_raises_value_error():
    df = _valid_dataframe()
    df.loc[0, "sensor_id"] = None

    with pytest.raises(ValueError, match=r"sensor_id contains 1"):
        validate_input_data(df)


def test_empty_sensor_id_raises_value_error():
    df = _valid_dataframe()
    df.loc[0, "sensor_id"] = ""

    with pytest.raises(ValueError, match="sensor_id"):
        validate_input_data(df)


def test_whitespace_sensor_id_raises_value_error():
    df = _valid_dataframe()
    df.loc[0, "sensor_id"] = "   "

    with pytest.raises(ValueError, match="sensor_id"):
        validate_input_data(df)


def test_text_temperature_raises_value_error():
    df = _valid_dataframe()
    df["temperature"] = pd.Series(["error", 71.0], dtype=object)

    with pytest.raises(ValueError, match=r"temperature contains 1"):
        validate_input_data(df)


def test_numeric_string_temperature_is_allowed():
    df = _valid_dataframe()
    df["temperature"] = pd.Series(["-12.5", 71.0], dtype=object)

    assert validate_input_data(df) is None


def test_missing_temperature_is_allowed():
    df = _valid_dataframe()
    df.loc[0, "temperature"] = np.nan

    assert validate_input_data(df) is None


def test_preprocess_marks_missing_temperature():
    df = _valid_dataframe()
    df.loc[0, "temperature"] = np.nan

    result = preprocess_data(df)

    assert result.loc[result["timestamp"] == pd.Timestamp("2026-01-01"), "is_missing"].item() == 1


def test_extra_columns_are_allowed():
    df = _valid_dataframe().assign(comment=["ok", "ok"])

    assert validate_input_data(df) is None


def test_scenario_column_is_optional():
    df = _valid_dataframe()

    assert "scenario" not in df.columns
    assert validate_input_data(df) is None


def test_validation_does_not_modify_dataframe():
    df = _valid_dataframe()
    original = df.copy(deep=True)

    validate_input_data(df)

    assert_frame_equal(df, original)
