"""Единая прозрачная матрица уровней риска и действий оператора."""
import math


RISK_MATRIX = {
    "Normal": {
        "rank": 0,
        "criteria": "Нет тревоги ни по инженерным правилам, ни по ML-модели.",
        "recommendation": "Продолжать штатное наблюдение.",
    },
    "Warning": {
        "rank": 1,
        "criteria": (
            "Есть тревога ML либо правило z_score, group_deviation или "
            "sustained_overheat; более сильных признаков нет."
        ),
        "recommendation": (
            "Проверить тренд и соседние датчики при ближайшем обходе."
        ),
    },
    "Medium": {
        "rank": 2,
        "criteria": (
            "Сработало правило signal_loss или stuck_sensor либо ML-score "
            "выше среднего порога обученной модели."
        ),
        "recommendation": (
            "Оперативно проверить датчик и процесс по резервному измерению."
        ),
    },
    "High": {
        "rank": 3,
        "criteria": (
            "Сработало правило sharp_jump либо ML-score выше высокого "
            "порога обученной модели."
        ),
        "recommendation": (
            "Немедленно проверить процесс; решение об остановке принимает "
            "оператор по технологическому регламенту."
        ),
    },
}

RISK_RANK = {
    level: definition["rank"]
    for level, definition in RISK_MATRIX.items()
}

RULE_RISK_LEVEL = {
    "signal_loss": "Medium",
    "sharp_jump": "High",
    "z_score": "Warning",
    "stuck_sensor": "Medium",
    "group_deviation": "Warning",
    "sustained_overheat": "Warning",
}


def _finite_number(value):
    """Возвращает число только для конечных числовых значений."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def assess_risk(
    triggered_rules,
    iforest_anomaly=False,
    anomaly_score_norm=None,
    medium_threshold=None,
    high_threshold=None,
):
    """Возвращает максимальный прозрачный уровень риска и рекомендацию.

    ML-score только повышает уровень уже обнаруженной тревоги и сам по себе не
    создаёт событие. Это сохраняет разделение между решением модели и тяжестью.
    """
    rules = list(triggered_rules or [])
    unknown_rules = sorted(set(rules) - set(RULE_RISK_LEVEL))
    if unknown_rules:
        raise ValueError(
            "Для правил не задан уровень риска: "
            f"{', '.join(unknown_rules)}."
        )

    evidence_levels = [RULE_RISK_LEVEL[rule] for rule in rules]
    if bool(iforest_anomaly):
        evidence_levels.append("Warning")

    if not evidence_levels:
        level = "Normal"
        return level, RISK_MATRIX[level]["recommendation"]

    score = _finite_number(anomaly_score_norm)
    medium = _finite_number(medium_threshold)
    high = _finite_number(high_threshold)
    if score is not None and high is not None and score > high:
        evidence_levels.append("High")
    elif score is not None and medium is not None and score > medium:
        evidence_levels.append("Medium")

    level = max(evidence_levels, key=RISK_RANK.__getitem__)
    return level, RISK_MATRIX[level]["recommendation"]