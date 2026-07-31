"""Build a non-mutating quality summary for raw measurements."""

import pandas as pd

from input_contract import (
    inspect_measurement_metadata,
    inspect_timestamp_values,
    timestamp_policy_description,
)


REQUIRED_COLUMNS = ("timestamp", "sensor_id", "temperature")
DUPLICATE_MEASUREMENT_POLICY = "drop_exact_error_on_conflict"
DUPLICATE_MEASUREMENT_ACTION = (
    "полностью одинаковые строки сворачиваются в одну; различающиеся строки "
    "с одинаковыми sensor_id и timestamp останавливают анализ"
)


def _normalized_duplicate_frame(df, valid_keys=None):
    """Return a copy with normalized fields used by the duplicate key."""
    normalized = df.copy(deep=True)
    normalized["timestamp"] = inspect_timestamp_values(
        normalized["timestamp"]
    ).normalized
    normalized["sensor_id"] = (
        normalized["sensor_id"].astype("string").str.strip()
    )
    normalized["temperature"] = pd.to_numeric(
        normalized["temperature"], errors="coerce"
    )
    if valid_keys is not None:
        normalized = normalized.loc[valid_keys].copy()
    return normalized


def _duplicate_summary(df, valid_keys):
    normalized = _normalized_duplicate_frame(df, valid_keys=valid_keys)
    key_columns = ["sensor_id", "timestamp"]
    duplicate_count = int(
        normalized.duplicated(key_columns, keep="first").sum()
    )
    exact_duplicate_count = int(normalized.duplicated(keep="first").sum())

    without_exact = normalized.drop_duplicates(keep="first")
    conflict_mask = without_exact.duplicated(key_columns, keep=False)
    conflicting_key_count = int(
        without_exact.loc[conflict_mask, key_columns].drop_duplicates().shape[0]
    )
    return duplicate_count, exact_duplicate_count, conflicting_key_count


def apply_duplicate_measurement_policy(df):
    """Drop exact repeats and reject ambiguous measurements.

    The input is copied. sensor_id is stripped and timestamp is normalized
    according to the shared time policy before the duplicate key is evaluated.
    Run this after ordinary input validation and before feature calculations.
    """
    normalized = _normalized_duplicate_frame(df)
    without_exact = normalized.drop_duplicates(keep="first")
    key_columns = ["sensor_id", "timestamp"]
    conflict_mask = without_exact.duplicated(key_columns, keep=False)

    if conflict_mask.any():
        examples = without_exact.loc[conflict_mask, key_columns].drop_duplicates()
        sample = ", ".join(
            f"{row.sensor_id} @ {row.timestamp}"
            for row in examples.head(3).itertuples(index=False)
        )
        raise ValueError(
            "Обнаружены конфликтующие повторные измерения: для одинаковых "
            "sensor_id + timestamp строки различаются. Исправьте CSV вместо "
            f"автоматического выбора или усреднения. Примеры: {sample}."
        )

    return without_exact.reset_index(drop=True)


def build_data_quality_report(df):
    """Return quality metrics, time policy and metadata for the dashboard."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    missing = [name for name in REQUIRED_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    rows = len(df)
    time_inspection = inspect_timestamp_values(df["timestamp"])
    timestamps = time_inspection.normalized
    metadata = inspect_measurement_metadata(df)
    ids = df["sensor_id"]
    keys = ids.astype("string").str.strip()
    valid_ids = ids.notna() & keys.ne("")
    temperatures = df["temperature"]
    numeric = pd.to_numeric(temperatures, errors="coerce")

    invalid_times = time_inspection.invalid_count
    mixed_timezones = time_inspection.mode == "mixed"
    missing_ids = int((~valid_ids).sum())
    non_numeric = int((temperatures.notna() & numeric.isna()).sum())
    missing_temperatures = int(temperatures.isna().sum())
    missing_percent = round(missing_temperatures / rows * 100, 1) if rows else 0.0

    valid_keys = valid_ids & timestamps.notna()
    measurements = pd.DataFrame({
        "sensor_id": keys[valid_keys],
        "timestamp": timestamps[valid_keys],
    })
    duplicates, exact_duplicates, conflicting_keys = _duplicate_summary(
        df, valid_keys
    )
    if mixed_timezones:
        out_of_order = 0
    else:
        out_of_order = int(
            measurements.groupby("sensor_id", sort=False)["timestamp"]
            .diff().lt(pd.Timedelta(0)).sum()
        )
    channels = pd.DataFrame({
        "sensor_id": keys[valid_ids],
        "temperature": numeric[valid_ids],
    })
    empty_sensors = int(
        channels.groupby("sensor_id")["temperature"]
        .apply(lambda values: values.notna().sum() == 0).sum()
    )

    warnings = []
    if rows == 0:
        warnings.append("Файл не содержит строк измерений.")
    if invalid_times:
        warnings.append(f"Некорректных или пустых меток времени: {invalid_times}.")
    if mixed_timezones:
        warnings.append(
            "Смешаны метки времени с часовым поясом и без него; "
            "анализ остановлен."
        )
    if missing_ids:
        warnings.append(f"Строк без идентификатора датчика: {missing_ids}.")
    if non_numeric:
        warnings.append(f"Нечисловых значений температуры: {non_numeric}.")
    if missing_temperatures:
        warnings.append(
            f"Пропусков температуры: {missing_temperatures} ({missing_percent:.1f}%)."
        )
    if empty_sensors:
        warnings.append(f"Датчиков без единого измеренного значения: {empty_sensors}.")
    if exact_duplicates:
        warnings.append(
            f"Полностью совпадающих повторных строк: {exact_duplicates}; "
            "перед расчётом признаков будет оставлена одна строка."
        )
    if conflicting_keys:
        warnings.append(
            f"Конфликтующих ключей sensor_id + timestamp: {conflicting_keys}; "
            "анализ будет остановлен, поскольку строки различаются."
        )
    if out_of_order:
        warnings.append(
            f"Переходов назад по времени: {out_of_order}; "
            "перед анализом строки были отсортированы."
        )
    if metadata["unsupported_temperature_units"]:
        warnings.append(
            "Неподдерживаемые единицы температуры: "
            + ", ".join(metadata["unsupported_temperature_units"])
            + "; автоматический перевод не выполняется."
        )
    if metadata["invalid_sensor_accuracy_count"]:
        warnings.append(
            "Некорректных значений sensor_accuracy: "
            f'{metadata["invalid_sensor_accuracy_count"]}.'
        )
    if metadata["sensor_accuracy_declared_count"]:
        warnings.append(
            "sensor_accuracy — заявленная характеристика источника; "
            "она не подтверждает калибровку или достоверность измерений."
        )

    blocking = (
        rows == 0
        or invalid_times > 0
        or mixed_timezones
        or missing_ids > 0
        or non_numeric > 0
        or conflicting_keys > 0
        or bool(metadata["unsupported_temperature_units"])
        or metadata["invalid_sensor_accuracy_count"] > 0
    )
    return {
        "status": "error" if blocking else ("warning" if warnings else "good"),
        "row_count": rows,
        "sensor_count": int(keys[valid_ids].nunique()),
        "missing_temperature_count": missing_temperatures,
        "missing_temperature_percent": missing_percent,
        "duplicate_measurement_count": duplicates,
        "exact_duplicate_measurement_count": exact_duplicates,
        "conflicting_duplicate_key_count": conflicting_keys,
        "duplicate_measurement_policy": DUPLICATE_MEASUREMENT_POLICY,
        "duplicate_measurement_action": DUPLICATE_MEASUREMENT_ACTION,
        "out_of_order_count": out_of_order,
        "fully_missing_sensor_count": empty_sensors,
        "invalid_timestamp_count": invalid_times,
        "timestamp_mode": time_inspection.mode,
        "timestamp_source_timezones": list(time_inspection.source_timezones),
        "timestamp_normalized_timezone": time_inspection.normalized_timezone,
        "timestamp_policy": timestamp_policy_description(time_inspection),
        "missing_sensor_id_count": missing_ids,
        "non_numeric_temperature_count": non_numeric,
        **metadata,
        "warnings": warnings,
    }
