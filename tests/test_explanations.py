import math

from explanations import build_alert_explanation


def _alarm(**overrides):
    alarm = {
        "Время": "2026-01-01 10:00:00",
        "Датчик": "T-01",
        "Температура": 78.5,
        "Температура_заполненная": 78.5,
        "Сработавшие_правила": ["sharp_jump", "group_deviation"],
        "primary_reason": "sharp_jump",
        "Тип_события": "Резкий скачок температуры",
        "Уровень": "High",
        "Anomaly_score": 0.91,
        "Рекомендация": "Немедленно проверить процесс.",
        "Режим_анализа": "rules+ML",
    }
    alarm.update(overrides)
    return alarm


def test_rule_alarm_contains_reasons_features_and_recommendation():
    detected = {"temp_rate_c_per_min": 6.2, "abs_diff_from_group_mean": 9.4, "iforest_anomaly": 0}
    explanation = build_alert_explanation(_alarm(), detected)
    assert explanation["event_type"] == "Резкий скачок температуры"
    assert explanation["risk"] == "High"
    assert explanation["reasons"] == ["Резкий скачок температуры", "Расхождение с соседними датчиками"]
    assert {feature["Признак"] for feature in explanation["features"]} >= {
        "Измеренная температура", "Скорость изменения температуры",
        "Отклонение от группы датчиков", "Anomaly score",
    }
    assert explanation["recommendation"] == "Немедленно проверить процесс."


def test_csv_encoded_reasons_are_decoded():
    explanation = build_alert_explanation(_alarm(Сработавшие_правила="['signal_loss', 'stuck_sensor']"))
    assert explanation["reasons"] == ["Потеря сигнала датчика", "Показание датчика не меняется"]


def test_ml_only_alarm_is_described_without_probability_claim():
    explanation = build_alert_explanation(
        _alarm(Сработавшие_правила=[], primary_reason="iforest_anomaly"),
        {"iforest_anomaly": 1},
    )
    assert explanation["reasons"] == ["Нетипичное сочетание признаков по Isolation Forest"]
    assert "нетипичное" in explanation["ml_summary"]
    assert "не вероятность аварии" in explanation["ml_summary"]


def test_rules_only_mode_says_that_ml_was_not_applied():
    explanation = build_alert_explanation(_alarm(Режим_анализа="rules-only", Anomaly_score=math.nan))
    assert explanation["ml_summary"].startswith("ML-модель не применялась")
    assert all(feature["Признак"] != "Anomaly score" for feature in explanation["features"])


def test_missing_optional_fields_have_safe_fallbacks():
    explanation = build_alert_explanation({"Датчик": "T-02"})
    assert explanation["sensor"] == "T-02"
    assert explanation["reasons"] == ["Причина не записана в журнале"]
    assert explanation["features"] == []
    assert explanation["recommendation"]