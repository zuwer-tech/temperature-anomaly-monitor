"""Build a non-mutating quality summary for raw measurements."""

import pandas as pd


REQUIRED_COLUMNS = ("timestamp", "sensor_id", "temperature")


def build_data_quality_report(df):
    """Return quality metrics and warnings for the dashboard."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    missing = [name for name in REQUIRED_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    rows = len(df)
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    ids = df["sensor_id"]
    keys = ids.astype("string").str.strip()
    valid_ids = ids.notna() & keys.ne("")
    temperatures = df["temperature"]
    numeric = pd.to_numeric(temperatures, errors="coerce")

    invalid_times = int(timestamps.isna().sum())
    missing_ids = int((~valid_ids).sum())
    non_numeric = int((temperatures.notna() & numeric.isna()).sum())
    missing_temperatures = int(temperatures.isna().sum())
    missing_percent = round(missing_temperatures / rows * 100, 1) if rows else 0.0

    valid_keys = valid_ids & timestamps.notna()
    measurements = pd.DataFrame({
        "sensor_id": keys[valid_keys],
        "timestamp": timestamps[valid_keys],
    })
    duplicates = int(
        measurements.duplicated(["sensor_id", "timestamp"], keep="first").sum()
    )
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
    if duplicates:
        warnings.append(f"Повторных строк для одного датчика и времени: {duplicates}.")
    if out_of_order:
        warnings.append(
            f"Переходов назад по времени: {out_of_order}; "
            "перед анализом строки были отсортированы."
        )

    blocking = rows == 0 or invalid_times > 0 or missing_ids > 0 or non_numeric > 0
    return {
        "status": "error" if blocking else ("warning" if warnings else "good"),
        "row_count": rows,
        "sensor_count": int(keys[valid_ids].nunique()),
        "missing_temperature_count": missing_temperatures,
        "missing_temperature_percent": missing_percent,
        "duplicate_measurement_count": duplicates,
        "out_of_order_count": out_of_order,
        "fully_missing_sensor_count": empty_sensors,
        "invalid_timestamp_count": invalid_times,
        "missing_sensor_id_count": missing_ids,
        "non_numeric_temperature_count": non_numeric,
        "warnings": warnings,
    }
