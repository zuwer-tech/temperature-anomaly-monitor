"""Группировка последовательных аномальных точек в операторские события."""
import ast

import numpy as np
import pandas as pd

from risk_config import RISK_RANK


EVENT_COLUMNS = (
    "event_id",
    "sensor_id",
    "event_start",
    "event_end",
    "duration_seconds",
    "point_count",
    "max_temperature",
    "reasons",
    "max_risk",
    "recommendation",
)



def _reason_values(value):
    """Возвращает список причин из Python-списка или его CSV-представления."""
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "nan", "normal"}:
            return []
        if stripped.startswith(("[", "(", "{")):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                parsed = stripped
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
        return [stripped]
    if pd.isna(value):
        return []
    return [value]


def _event_reasons(event):
    reasons = []
    for _, row in event.iterrows():
        candidates = _reason_values(row.get("triggered_rules"))
        candidates.extend(_reason_values(row.get("primary_reason")))
        if not candidates:
            candidates.extend(_reason_values(row.get("rule_event_type")))
        for reason in candidates:
            reason = str(reason)
            if reason not in reasons:
                reasons.append(reason)
    return reasons


def _event_risk_and_recommendation(event):
    if "rule_risk_level" in event:
        risks = event["rule_risk_level"].fillna("Warning").astype(str)
    else:
        risks = pd.Series("Warning", index=event.index)

    unknown_risks = sorted(set(risks) - set(RISK_RANK))
    if unknown_risks:
        raise ValueError(
            "Неизвестные уровни риска: "
            f"{', '.join(unknown_risks)}."
        )

    max_risk_index = risks.map(RISK_RANK).idxmax()
    max_risk = risks.loc[max_risk_index]

    if "rule_recommendation" not in event:
        return max_risk, ""

    recommendations = event["rule_recommendation"].fillna("").astype(str)
    recommendation = recommendations.loc[max_risk_index].strip()
    if not recommendation:
        non_empty = recommendations[recommendations.str.strip().ne("")]
        recommendation = non_empty.iloc[0].strip() if not non_empty.empty else ""
    return max_risk, recommendation


def group_anomaly_events(detected):
    """Объединяет непрерывные аномальные точки отдельно для каждого датчика.

    Новое событие начинается после нормальной точки или в начале ряда датчика.
    Временные ряды сортируются стабильно, исходный DataFrame не изменяется.
    """
    required_columns = {"timestamp", "sensor_id", "final_anomaly"}
    missing_columns = sorted(required_columns - set(detected.columns))
    if missing_columns:
        raise ValueError(
            "Для группировки событий отсутствуют колонки: "
            f"{', '.join(missing_columns)}."
        )

    if detected.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    work = detected.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="raise")
    if work["timestamp"].isna().any():
        raise ValueError("timestamp не должен содержать пустые значения.")
    if work["sensor_id"].isna().any():
        raise ValueError("sensor_id не должен содержать пустые значения.")

    final_anomaly = pd.to_numeric(work["final_anomaly"], errors="raise")
    if not final_anomaly.isin([0, 1]).all():
        raise ValueError("final_anomaly должен содержать только 0 и 1.")
    work["final_anomaly"] = final_anomaly.astype(int)

    work = work.sort_values(
        ["sensor_id", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)

    previous_anomaly = work.groupby(
        "sensor_id",
        sort=False,
    )["final_anomaly"].shift(fill_value=0)
    starts_event = work["final_anomaly"].eq(1) & previous_anomaly.ne(1)
    work["_event_number"] = starts_event.groupby(work["sensor_id"]).cumsum()

    anomaly_rows = work[work["final_anomaly"].eq(1)]
    if anomaly_rows.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    event_rows = []
    for (sensor_id, event_number), event in anomaly_rows.groupby(
        ["sensor_id", "_event_number"],
        sort=False,
    ):
        event_start = event["timestamp"].iloc[0]
        event_end = event["timestamp"].iloc[-1]

        temperature_column = (
            "temperature"
            if "temperature" in event
            else "temperature_filled"
            if "temperature_filled" in event
            else None
        )
        if temperature_column is None:
            max_temperature = np.nan
        else:
            temperatures = pd.to_numeric(
                event[temperature_column],
                errors="coerce",
            )
            max_temperature = (
                float(temperatures.max())
                if temperatures.notna().any()
                else np.nan
            )

        max_risk, recommendation = _event_risk_and_recommendation(event)
        event_rows.append(
            {
                "event_id": f"{sensor_id}-{int(event_number):04d}",
                "sensor_id": str(sensor_id),
                "event_start": event_start,
                "event_end": event_end,
                "duration_seconds": (
                    event_end - event_start
                ).total_seconds(),
                "point_count": int(len(event)),
                "max_temperature": max_temperature,
                "reasons": _event_reasons(event),
                "max_risk": max_risk,
                "recommendation": recommendation,
            }
        )

    return pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
