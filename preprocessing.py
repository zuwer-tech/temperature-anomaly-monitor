import pandas as pd
import numpy as np

from data_quality import apply_duplicate_measurement_policy
from input_contract import (
    inspect_timestamp_values,
    validate_measurement_metadata,
)
from rule_config import RULE_PARAMS
from sensor_alignment import causal_group_mean
from time_windows import causal_stuck_flags, causal_time_rolling


def validate_input_data(df):
    """Validate raw temperature data before preprocessing.

    Checks the DataFrame type, required columns, row presence, timestamps,
    sensor identifiers, and numeric-compatible temperature values. The
    function does not modify the supplied DataFrame.

    Raises
    ------
    TypeError
        If ``df`` is not a pandas DataFrame.
    ValueError
        If required data is missing or a timestamp, sensor identifier, or
        non-missing temperature value is invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    required_columns = ["timestamp", "sensor_id", "temperature"]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    if df.empty:
        raise ValueError("Input DataFrame must contain at least one row.")

    timestamp_inspection = inspect_timestamp_values(df["timestamp"])
    if timestamp_inspection.invalid_count:
        raise ValueError(
            "timestamp contains "
            f"{timestamp_inspection.invalid_count} invalid or missing value(s)."
        )
    if timestamp_inspection.mode == "mixed":
        raise ValueError(
            "timestamp mixes timezone-aware and timezone-naive values. "
            "Use one local timezone for every unzoned value or include a "
            "timezone/UTC offset in every value."
        )

    sensor_ids = df["sensor_id"]
    invalid_sensor_ids = sensor_ids.isna() | sensor_ids.map(
        lambda value: isinstance(value, str) and not value.strip()
    )
    invalid_sensor_id_count = int(invalid_sensor_ids.sum())
    if invalid_sensor_id_count:
        raise ValueError(
            "sensor_id contains "
            f"{invalid_sensor_id_count} missing or blank value(s)."
        )

    temperatures = df["temperature"]
    numeric_temperatures = pd.to_numeric(temperatures, errors="coerce")
    invalid_temperatures = temperatures.notna() & numeric_temperatures.isna()
    invalid_temperature_count = int(invalid_temperatures.sum())
    if invalid_temperature_count:
        raise ValueError(
            "temperature contains "
            f"{invalid_temperature_count} invalid non-numeric value(s)."
        )

    validate_measurement_metadata(df)


def preprocess_data(df):
    """
    Предобработка температурных данных.

    На вход получает DataFrame с колонками:
    - timestamp
    - sensor_id
    - temperature

    На выход возвращает DataFrame с дополнительными признаками:
    - is_missing
    - temperature_filled
    - rolling_mean
    - rolling_std
    - temp_diff
    - time_diff_seconds
    - temp_rate_c_per_min
    - abs_temp_diff
    - z_score
    - abs_z_score
    - is_stuck
    - abs_diff_from_group_mean
    - group_peer_count
    - preliminary_warning
    """

    validate_input_data(df)

    # Resolve duplicate keys before diff() and rolling feature calculations.
    df = apply_duplicate_measurement_policy(df)

    # Если нет колонки scenario, добавляем ее для пользовательских данных
    if "scenario" not in df.columns:
        df["scenario"] = "user_data"

    # Время уже нормализовано политикой дублей: aware -> UTC, naive остаётся naive.

    # Температуру принудительно переводим в число
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")

    # Сортируем данные по датчику и времени
    df = df.sort_values(by=["sensor_id", "timestamp"]).reset_index(drop=True)

    # Фактический интервал между соседними измерениями каждого датчика.
    # Для первой точки датчика интервал неизвестен и остаётся NaN.
    df["time_diff_seconds"] = (
        df.groupby("sensor_id")["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    # Признак пропуска сигнала
    df["is_missing"] = df["temperature"].isna().astype(int)

    # Заполнение пропусков внутри каждого датчика
    df["temperature_filled"] = (
        df.groupby("sensor_id")["temperature"]
        .transform(lambda values: values.ffill().bfill())
    )

    # Полностью пустой канал остаётся NaN: отсутствие измерения не является 0 °C.
    # Частичные пропуски по-прежнему заполняются только значениями того же датчика.

    # Скользящие признаки в физическом причинном окне [t-duration, t].
    # Большой разрыв начинает новое окно: мы не считаем неизвестный участок
    # между пробами частью непрерывного процесса.
    df["rolling_mean"] = causal_time_rolling(
        df,
        "temperature_filled",
        RULE_PARAMS["rolling_duration_seconds"],
        "mean",
        RULE_PARAMS["continuity_gap_seconds"],
    )
    df["rolling_std"] = causal_time_rolling(
        df,
        "temperature_filled",
        RULE_PARAMS["rolling_duration_seconds"],
        "std",
        RULE_PARAMS["continuity_gap_seconds"],
    )

    # 0/std на ранних точках (окно не накопилось) -> маленькая константа, чтобы
    # z-score не взрывался в inf.
    df["rolling_std"] = df["rolling_std"].fillna(0).replace(0, 1e-6)

    # Скорость изменения температуры
    df["temp_diff"] = (
        df.groupby("sensor_id")["temperature_filled"]
        .diff()
        .fillna(0)
    )

    df["abs_temp_diff"] = df["temp_diff"].abs()

    # Физическая скорость изменения температуры, °C/мин.
    # Нулевой/отрицательный интервал и отсутствующие измерения не дают
    # осмысленной скорости, поэтому результат для них остаётся NaN.
    previous_temperature = (
        df.groupby("sensor_id")["temperature"]
        .shift()
    )
    valid_rate_interval = (
        (df["time_diff_seconds"] > 0)
        & df["temperature"].notna()
        & previous_temperature.notna()
    )
    df["temp_rate_c_per_min"] = np.nan
    df.loc[valid_rate_interval, "temp_rate_c_per_min"] = (
        df.loc[valid_rate_interval, "temp_diff"]
        * 60
        / df.loc[valid_rate_interval, "time_diff_seconds"]
    )

    # Z-score: скользящий (локальный) — отклонение от собственного rolling_mean.
    # Раньше считался по глобальному среднему всего датчика: на растущем реальном
    # сигнале (Т2.csv: 49->109 °C) весь верхний полубас уходил в аномалию. Скользящий
    # z-score адаптируется к тренду и ловит локальные выбросы, а не сам тренд.
    df["z_score"] = (df["temperature_filled"] - df["rolling_mean"]) / df["rolling_std"]
    df["z_score"] = df["z_score"].replace([np.inf, -np.inf], 0).fillna(0)
    df["abs_z_score"] = df["z_score"].abs()

    # Зависание измеряется реальным временем непрерывного равенства сигнала.
    # Разрыв длиннее continuity_gap_seconds сбрасывает серию: по отсутствующим
    # пробам нельзя заключать, что датчик всё это время был неподвижен.
    df["is_stuck"] = causal_stuck_flags(
        df,
        "temperature_filled",
        RULE_PARAMS["stuck_duration_seconds"],
        RULE_PARAMS["continuity_gap_seconds"],
        RULE_PARAMS["stuck_abs_tolerance"],
    )

    # Диагностический stuck_score сохранён, но теперь это доля малых изменений
    # в физическом окне (0..1), а не зависимый от частоты опроса счётчик строк.
    df["small_change"] = (df["abs_temp_diff"] < 0.05).astype(int)
    df["stuck_score"] = causal_time_rolling(
        df,
        "small_change",
        RULE_PARAMS["stuck_score_duration_seconds"],
        "mean",
        RULE_PARAMS["continuity_gap_seconds"],
    )

    # Отклонение от группы. Для каждой точки берём только одновременные или
    # последние прошлые измерения соседних датчиков в пределах явного допуска.
    # Будущие точки не используются. group_peer_count показывает, сколько
    # именно соседних датчиков подтвердили групповое среднее.
    if df["sensor_id"].nunique() > 1:
        (
            df["mean_temp_all_sensors"],
            df["group_peer_count"],
        ) = causal_group_mean(
            df,
            "temperature",
            RULE_PARAMS["group_alignment_tolerance_seconds"],
        )
        df["diff_from_group_mean"] = (
            df["temperature_filled"] - df["mean_temp_all_sensors"]
        )
    else:
        df["mean_temp_all_sensors"] = df["rolling_mean"]
        df["diff_from_group_mean"] = df["temperature_filled"] - df["rolling_mean"]
        df["group_peer_count"] = 0

    df["abs_diff_from_group_mean"] = df["diff_from_group_mean"].abs()

    # Предварительная техническая метка подозрительности
    df["preliminary_warning"] = 0

    df.loc[df["is_missing"] == 1, "preliminary_warning"] = 1
    df.loc[
        df["temp_rate_c_per_min"].abs()
        > RULE_PARAMS["sharp_jump_rate_c_per_min"],
        "preliminary_warning",
    ] = 1
    df.loc[
        df["abs_z_score"] > RULE_PARAMS["z_score"],
        "preliminary_warning",
    ] = 1
    df.loc[df["is_stuck"] == 1, "preliminary_warning"] = 1
    df.loc[
        df["abs_diff_from_group_mean"] > RULE_PARAMS["group_deviation"],
        "preliminary_warning",
    ] = 1

    return df


if __name__ == "__main__":
    input_file = "synthetic_temperature_data.csv"
    output_file = "preprocessed_temperature_data.csv"

    df = pd.read_csv(input_file)
    processed_df = preprocess_data(df)

    processed_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\nПредобработка завершена.")
    print(f"Файл сохранён: {output_file}")
    print("\nПервые строки обработанных данных:")
    print(processed_df.head())
    print("\nКоличество предварительно подозрительных точек:")
    print(processed_df["preliminary_warning"].sum())
