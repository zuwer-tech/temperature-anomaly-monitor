"""Подготовка проверяемого объяснения одной тревоги для интерфейса."""
import ast
import math


REASON_LABELS = {
    "signal_loss": "Потеря сигнала датчика",
    "sharp_jump": "Резкий скачок температуры",
    "z_score": "Сильное отклонение от обычного режима",
    "stuck_sensor": "Показание датчика не меняется",
    "group_deviation": "Расхождение с соседними датчиками",
    "sustained_overheat": "Устойчивый рост температуры",
    "iforest_anomaly": "Нетипичное сочетание признаков по Isolation Forest",
}

FEATURE_FIELDS = (
    ("Измеренная температура", ("Температура", "temperature"), "°C"),
    ("Температура после заполнения пропуска", ("Температура_заполненная", "temperature_filled"), "°C"),
    ("Скорость изменения температуры", ("temp_rate_c_per_min",), "°C/мин"),
    ("Отклонение от обычного режима", ("abs_z_score",), "σ"),
    ("Отклонение от группы датчиков", ("abs_diff_from_group_mean",), "°C"),
    ("Наклон устойчивого перегрева", ("temp_slope_overheat",), "°C/точку"),
    ("Anomaly score", ("Anomaly_score", "anomaly_score_norm"), "0–1"),
)


def _as_mapping(row):
    if row is None:
        return {}
    if hasattr(row, "to_dict"):
        return row.to_dict()
    return dict(row)


def _first_value(primary, secondary, keys):
    for source in (primary, secondary):
        for key in keys:
            if key not in source:
                continue
            value = source[key]
            if value is None:
                continue
            try:
                if math.isnan(float(value)):
                    continue
            except (TypeError, ValueError):
                pass
            return value
    return None


def _reason_values(value):
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "nan", "normal"}:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = stripped
        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return [str(value)]


def _format_feature_value(value):
    if isinstance(value, bool):
        return "да" if value else "нет"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.3f}".rstrip("0").rstrip(".")


def build_alert_explanation(alarm_row, detected_row=None):
    """Собирает факты для карточки, не изменяя решение детектора."""
    alarm = _as_mapping(alarm_row)
    detected = _as_mapping(detected_row)

    rules = _reason_values(alarm.get("Сработавшие_правила"))
    primary_reason = _first_value(alarm, detected, ("primary_reason",))
    if not rules and primary_reason:
        rules = [str(primary_reason)]
    reason_labels = [REASON_LABELS.get(reason, reason) for reason in rules]

    features = []
    for name, keys, unit in FEATURE_FIELDS:
        value = _first_value(alarm, detected, keys)
        if value is not None:
            features.append({
                "Признак": name,
                "Значение": _format_feature_value(value),
                "Единица": unit,
            })

    analysis_mode = _first_value(alarm, detected, ("Режим_анализа", "analysis_mode"))
    if analysis_mode == "rules-only":
        ml_summary = "ML-модель не применялась: тревога получена только правилами."
    else:
        iforest_anomaly = _first_value(alarm, detected, ("iforest_anomaly",))
        score = _first_value(alarm, detected, ("Anomaly_score", "anomaly_score_norm"))
        if bool(iforest_anomaly) or primary_reason == "iforest_anomaly":
            ml_summary = "Isolation Forest отметил сочетание признаков как нетипичное."
        else:
            ml_summary = "Isolation Forest не был самостоятельной причиной тревоги."
        if score is not None:
            ml_summary += (
                f" Anomaly score: {_format_feature_value(score)}; "
                "это мера необычности, а не вероятность аварии."
            )

    return {
        "sensor": _first_value(alarm, detected, ("Датчик", "sensor_id")) or "—",
        "timestamp": _first_value(alarm, detected, ("Время", "timestamp")) or "—",
        "event_type": _first_value(alarm, detected, ("Тип_события", "rule_event_type")) or "Не указан",
        "risk": _first_value(alarm, detected, ("Уровень", "rule_risk_level")) or "Не указан",
        "reasons": reason_labels or ["Причина не записана в журнале"],
        "features": features,
        "ml_summary": ml_summary,
        "recommendation": _first_value(alarm, detected, ("Рекомендация", "rule_recommendation")) or "Проверить исходные данные и журнал анализа.",
    }