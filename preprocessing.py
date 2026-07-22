import pandas as pd
import numpy as np


# Параметры предобработки (вынесены из тела функции — единая точка настройки).
# Подобраны так, чтобы работать и на синтетике (несколько датчиков), и на реальных
# одноканальных трендовых данных (Т2.csv). См. tests/ и issue по объединению mod_AI_2.
ROLLING_WINDOW = 10        # окно скользящих признаков, точек
STUCK_MIN_RUN = 10        # сколько подряд одинаковых значений считать зависанием
STUCK_ABS_TOL = 1e-6       # точное равенство для зависания (ступени квантования не ловим)


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

    parsed_timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    invalid_timestamp_count = int(parsed_timestamps.isna().sum())
    if invalid_timestamp_count:
        raise ValueError(
            "timestamp contains "
            f"{invalid_timestamp_count} invalid or missing value(s)."
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


def preprocess_data(df, rolling_window=ROLLING_WINDOW):
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
    - preliminary_warning
    """

    validate_input_data(df)

    df = df.copy()

    # Если нет колонки scenario, добавляем ее для пользовательских данных
    if "scenario" not in df.columns:
        df["scenario"] = "user_data"

    # Переводим timestamp в формат даты и времени
    df["timestamp"] = pd.to_datetime(df["timestamp"])

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
        .ffill()
        .bfill()
    )

    # Полностью пустой канал остаётся NaN: отсутствие измерения не является 0 °C.
    # Частичные пропуски по-прежнему заполняются только значениями того же датчика.

    # Скользящие признаки
    window_size = rolling_window

    df["rolling_mean"] = (
        df.groupby("sensor_id")["temperature_filled"]
        .transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    )

    df["rolling_std"] = (
        df.groupby("sensor_id")["temperature_filled"]
        .transform(lambda x: x.rolling(window=window_size, min_periods=1).std())
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

    # Признак зависания датчика: точное равенство подряд STUCK_MIN_RUN значений.
    # Пороговый критерий (|Δt|<0.05 за окно 15) ловил ступени квантования АЦП
    # реальных данных как зависание — тысячи ложных тревог. Точное равенство
    # срабатывает только на по-настоящему застывшем сигнале.
    is_stuck = np.zeros(len(df), dtype=int)
    for _sid, idx in df.groupby("sensor_id").groups.items():
        vals = df.loc[idx, "temperature_filled"].to_numpy()
        run = 0
        prev = None
        for j, i in enumerate(idx):
            v = vals[j]
            if (
                prev is not None
                and not (np.isnan(v) or np.isnan(prev))
                and abs(v - prev) < STUCK_ABS_TOL
            ):
                run += 1
            else:
                run = 0
            if run >= STUCK_MIN_RUN:
                is_stuck[i] = 1
            prev = v
    df["is_stuck"] = is_stuck

    # Совместимость со старыми выходами: small_change/stuck_score сохранены
    # (используются в отчётах), но is_stuck теперь считается точным равенством.
    df["small_change"] = (df["abs_temp_diff"] < 0.05).astype(int)
    df["stuck_score"] = (
        df.groupby("sensor_id")["small_change"]
        .transform(lambda x: x.rolling(window=15, min_periods=1).sum())
    )

    # Отклонение от группы. Для одного датчика кросс-сенсорное среднее равно самому
    # значению (отклонение 0, правило мёртвое). Поэтому при одном датчике берём
    # отклонение от собственного rolling_mean (подход mod_AI_2) — оно осмысленно.
    if df["sensor_id"].nunique() > 1:
        df["mean_temp_all_sensors"] = (
            df.groupby("timestamp")["temperature_filled"]
            .transform("mean")
        )
        df["diff_from_group_mean"] = (
            df["temperature_filled"] - df["mean_temp_all_sensors"]
        )
    else:
        df["mean_temp_all_sensors"] = df["rolling_mean"]
        df["diff_from_group_mean"] = df["temperature_filled"] - df["rolling_mean"]

    df["abs_diff_from_group_mean"] = df["diff_from_group_mean"].abs()

    # Предварительная техническая метка подозрительности
    df["preliminary_warning"] = 0

    df.loc[df["is_missing"] == 1, "preliminary_warning"] = 1
    df.loc[df["abs_temp_diff"] > 5, "preliminary_warning"] = 1
    df.loc[df["abs_z_score"] > 3, "preliminary_warning"] = 1
    df.loc[df["is_stuck"] == 1, "preliminary_warning"] = 1
    df.loc[df["abs_diff_from_group_mean"] > 8, "preliminary_warning"] = 1

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
