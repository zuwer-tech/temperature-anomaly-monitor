import numpy as np
import pandas as pd
import pytest

from events import EVENT_COLUMNS, group_anomaly_events


def _detected_rows():
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:02:00",
                "2026-01-01 00:00:00",
                "2026-01-01 00:01:00",
                "2026-01-01 00:03:00",
                "2026-01-01 00:05:00",
                "2026-01-01 00:04:00",
                "2026-01-01 00:00:30",
            ],
            "sensor_id": [
                "T-01",
                "T-01",
                "T-01",
                "T-01",
                "T-01",
                "T-01",
                "T-02",
            ],
            "final_anomaly": [1, 1, 1, 0, 1, 1, 1],
            "temperature": [74.0, 71.0, 73.0, 72.0, np.nan, 78.0, 65.0],
            "temperature_filled": [74.0, 71.0, 73.0, 72.0, 80.0, 78.0, 65.0],
            "triggered_rules": [
                ["sustained_overheat"],
                ["z_score"],
                ["z_score", "group_deviation"],
                [],
                ["signal_loss"],
                "['sharp_jump']",
                [],
            ],
            "primary_reason": [
                "sustained_overheat",
                "z_score",
                "group_deviation",
                None,
                "signal_loss",
                "sharp_jump",
                "iforest_anomaly",
            ],
            "rule_event_type": [
                "Устойчивый перегрев",
                "Сильное отклонение от нормы",
                "Отклонение от группы датчиков",
                "normal",
                "Потеря сигнала",
                "Резкий скачок температуры",
                "Нетипичное поведение по ИИ-модели",
            ],
            "rule_risk_level": [
                "Warning",
                "Warning",
                "Medium",
                "Normal",
                "High",
                "Medium",
                "Warning",
            ],
            "rule_recommendation": [
                "Проверить рост",
                "Проверить тренд",
                "Сравнить датчики",
                "Наблюдение",
                "Проверить канал",
                "Проверить скачок",
                "Проверить график",
            ],
        }
    )


def test_long_anomaly_becomes_one_event_with_aggregated_fields():
    detected = _detected_rows()
    original = detected.copy(deep=True)

    events = group_anomaly_events(detected)

    pd.testing.assert_frame_equal(detected, original)
    assert list(events.columns) == list(EVENT_COLUMNS)
    assert len(events) == 3

    first = events.iloc[0]
    assert first["event_id"] == "T-01-0001"
    assert first["sensor_id"] == "T-01"
    assert first["event_start"] == pd.Timestamp("2026-01-01 00:00:00")
    assert first["event_end"] == pd.Timestamp("2026-01-01 00:02:00")
    assert first["duration_seconds"] == 120.0
    assert first["point_count"] == 3
    assert first["max_temperature"] == 74.0
    assert first["reasons"] == [
        "z_score",
        "group_deviation",
        "sustained_overheat",
    ]
    assert first["max_risk"] == "Medium"
    assert first["recommendation"] == "Сравнить датчики"

    second = events.iloc[1]
    assert second["event_id"] == "T-01-0002"
    assert second["event_start"] == pd.Timestamp("2026-01-01 00:04:00")
    assert second["event_end"] == pd.Timestamp("2026-01-01 00:05:00")
    assert second["duration_seconds"] == 60.0
    assert second["max_temperature"] == 78.0
    assert second["reasons"] == ["sharp_jump", "signal_loss"]
    assert second["max_risk"] == "High"
    assert second["recommendation"] == "Проверить канал"

    third = events.iloc[2]
    assert third["event_id"] == "T-02-0001"
    assert third["duration_seconds"] == 0.0
    assert third["reasons"] == ["iforest_anomaly"]


def test_no_anomalies_returns_empty_event_schema():
    detected = _detected_rows()
    detected["final_anomaly"] = 0

    events = group_anomaly_events(detected)

    assert events.empty
    assert list(events.columns) == list(EVENT_COLUMNS)


@pytest.mark.parametrize(
    "column",
    ["timestamp", "sensor_id", "final_anomaly"],
)
def test_required_columns_are_validated(column):
    with pytest.raises(ValueError, match=column):
        group_anomaly_events(_detected_rows().drop(columns=[column]))


def test_final_anomaly_must_be_binary():
    detected = _detected_rows()
    detected.loc[0, "final_anomaly"] = 2

    with pytest.raises(ValueError, match="0 и 1"):
        group_anomaly_events(detected)


def test_unknown_risk_is_rejected():
    detected = _detected_rows()
    detected.loc[0, "rule_risk_level"] = "Critical"

    with pytest.raises(ValueError, match="Critical"):
        group_anomaly_events(detected)


def test_missing_optional_explanation_fields_still_groups_events():
    detected = _detected_rows().drop(
        columns=[
            "temperature",
            "temperature_filled",
            "triggered_rules",
            "primary_reason",
            "rule_event_type",
            "rule_risk_level",
            "rule_recommendation",
        ]
    )

    events = group_anomaly_events(detected)

    assert len(events) == 3
    assert events["max_temperature"].isna().all()
    assert events["reasons"].map(len).eq(0).all()
    assert set(events["max_risk"]) == {"Warning"}
    assert events["recommendation"].eq("").all()
