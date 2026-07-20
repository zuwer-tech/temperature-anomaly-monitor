import os
import sys

import pandas as pd
import numpy as np

from joblib import load


class ModelNotTrainedError(RuntimeError):
    """Обученная модель отсутствует или сохранена не полностью."""


# Единое место настройки порогов правил. Меняйте значения здесь, а не в теле
# функций. Пороги подобраны так, чтобы минимум ложных тревог на штатном режиме
# при сохранении высокого полноты по реальным сценариям (см. tests/test_rule_accuracy.py).
RULE_PARAMS = {
    "sharp_jump_diff": 5.0,      # |Δt| между соседними точками, °C
    "z_score": 3.0,             # отклонение от среднего датчика
    "group_deviation": 8.0,      # отклонение от среднего по группе датчиков, °C
    "overheat_window": 20,      # окно для наклона (устойчивый перегрев), точки
    "overheat_slope": 0.12,     # наклон rolling_mean, °C/точку
}


def _rolling_slope(series, window):
    """Скользящий наклон линии тренда (линейная регрессия по окну).

    Считается по сглаженному rolling_mean (а не по сырой температуре), чтобы
    шум не порождал ложных срабатываний. Возвращает NaN, пока окно не накопилось.
    """
    def _slope(y):
        if len(y) < window:
            return np.nan
        x = np.arange(len(y), dtype=float)
        return np.polyfit(x, y, 1)[0]
    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def _load_or_fit_iforest(X, model_dir="models"):
    """Загружает модель и возвращает результаты инференса без переобучения."""
    path_scaler = os.path.join(model_dir, "scaler.joblib")
    path_model = os.path.join(model_dir, "iforest.joblib")
    artifacts = {
        "scaler.joblib": path_scaler,
        "iforest.joblib": path_model,
    }
    missing = [name for name, path in artifacts.items() if not os.path.isfile(path)]

    if missing:
        raise ModelNotTrainedError(
            "Обученная модель Isolation Forest не найдена или комплект "
            "артефактов неполный. Отсутствуют файлы: "
            f"{', '.join(missing)}. Сначала выполните:\n"
            "python preprocessing.py\n"
            "python train_model.py"
        )

    scaler = load(path_scaler)
    model = load(path_model)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    score_raw = model.decision_function(X_scaled)
    return X_scaled, predictions, score_raw, True


def detect_anomalies(df, model_dir="models"):
    """
    Обнаружение температурных аномалий.

    На вход получает предобработанный DataFrame.
    На выход возвращает:
    - df с результатами анализа
    - alarm_log с журналом тревог
    """

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["sensor_id", "timestamp"]).reset_index(drop=True)

    # ============================================================
    # 1. ОБНАРУЖЕНИЕ АНОМАЛИЙ ПО ПРАВИЛАМ
    # ============================================================

    df["rule_anomaly"] = 0
    df["rule_event_type"] = "normal"
    df["rule_risk_level"] = "Normal"
    df["rule_recommendation"] = "Наблюдение в штатном режиме"

    # Потеря сигнала
    mask_missing = df["is_missing"] == 1

    df.loc[mask_missing, "rule_anomaly"] = 1
    df.loc[mask_missing, "rule_event_type"] = "Потеря сигнала"
    df.loc[mask_missing, "rule_risk_level"] = "Medium"
    df.loc[mask_missing, "rule_recommendation"] = (
        "Проверить канал измерения и наличие связи с датчиком"
    )

    # Резкий скачок температуры
    mask_sharp_jump = df["abs_temp_diff"] > RULE_PARAMS["sharp_jump_diff"]

    df.loc[mask_sharp_jump, "rule_anomaly"] = 1
    df.loc[mask_sharp_jump, "rule_event_type"] = "Резкий скачок температуры"
    df.loc[mask_sharp_jump, "rule_risk_level"] = "High"
    df.loc[mask_sharp_jump, "rule_recommendation"] = (
        "Проверить показания датчика и сравнить с соседними каналами"
    )

    # Сильное отклонение от обычного режима датчика
    mask_z_score = df["abs_z_score"] > RULE_PARAMS["z_score"]

    df.loc[mask_z_score, "rule_anomaly"] = 1
    df.loc[mask_z_score, "rule_event_type"] = "Сильное отклонение от нормы"
    df.loc[mask_z_score, "rule_risk_level"] = "Warning"
    df.loc[mask_z_score, "rule_recommendation"] = (
        "Проверить тренд температуры и устойчивость отклонения"
    )

    # Зависший датчик
    mask_stuck = df["is_stuck"] == 1

    df.loc[mask_stuck, "rule_anomaly"] = 1
    df.loc[mask_stuck, "rule_event_type"] = "Зависание датчика"
    df.loc[mask_stuck, "rule_risk_level"] = "Medium"
    df.loc[mask_stuck, "rule_recommendation"] = (
        "Проверить исправность датчика и цепь передачи данных"
    )

    # Отклонение от группы датчиков
    mask_group_deviation = df["abs_diff_from_group_mean"] > RULE_PARAMS["group_deviation"]

    df.loc[mask_group_deviation, "rule_anomaly"] = 1
    df.loc[mask_group_deviation, "rule_event_type"] = "Отклонение от группы датчиков"
    df.loc[mask_group_deviation, "rule_risk_level"] = "Warning"
    df.loc[mask_group_deviation, "rule_recommendation"] = (
        "Сравнить показания с соседними температурными каналами"
    )

    # Медленный перегрев / устойчивый рост.
    # rolling_temp_diff_mean_20 остаётся как признак для Isolation Forest.
    df["rolling_temp_diff_mean_20"] = (
        df.groupby("sensor_id")["temp_diff"]
        .transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    )

    # Наклон сглаженной температуры ловит устойчивый перегрев, который простые
    # пороги пропускают (slow_overheating, correlated_growth). Считается по
    # сглаженному rolling_mean, чтобы шум не порождал ложных срабатываний.
    # Медленный дрейф датчика (sensor_drift) этим правилом намеренно не ловится:
    # его наклон ниже пика нормальной синусоиды, поэтому правилом от нормы не
    # отделить — его берёт на себя Isolation Forest (см. train_model.py).
    df["temp_slope_overheat"] = (
        df.groupby("sensor_id")["rolling_mean"]
        .transform(lambda x: _rolling_slope(x, window=RULE_PARAMS["overheat_window"]))
    )

    # Устойчивый перегрев — крутой наклон на коротком окне (slow_overheating,
    # correlated_growth).
    mask_overheat = df["temp_slope_overheat"] > RULE_PARAMS["overheat_slope"]
    df.loc[mask_overheat, "rule_anomaly"] = 1
    df.loc[mask_overheat, "rule_event_type"] = "Устойчивый перегрев"
    df.loc[mask_overheat, "rule_risk_level"] = "Warning"
    df.loc[mask_overheat, "rule_recommendation"] = (
        "Проверить устойчивость роста температуры и сравнить с соседними датчиками"
    )

    # ============================================================
    # 2. ISOLATION FOREST
    # ============================================================

    feature_columns = [
        "temperature_filled",
        "rolling_mean",
        "rolling_std",
        "temp_diff",
        "abs_temp_diff",
        "abs_z_score",
        "is_missing",
        "is_stuck",
        "abs_diff_from_group_mean",
        "rolling_temp_diff_mean_20"
    ]

    X = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

    X_scaled, iforest_prediction, iforest_score_raw, _used_saved = (
        _load_or_fit_iforest(X, model_dir=model_dir)
    )

    df["iforest_prediction"] = iforest_prediction
    df["iforest_anomaly"] = (df["iforest_prediction"] == -1).astype(int)

    df["iforest_score_raw"] = iforest_score_raw
    df["anomaly_score"] = -df["iforest_score_raw"]

    min_score = df["anomaly_score"].min()
    max_score = df["anomaly_score"].max()

    if max_score != min_score:
        df["anomaly_score_norm"] = (
            (df["anomaly_score"] - min_score) / (max_score - min_score)
        )
    else:
        df["anomaly_score_norm"] = 0

    df["anomaly_score_norm"] = df["anomaly_score_norm"].fillna(0)

    # ============================================================
    # 3. ОБЪЕДИНЕНИЕ ПРАВИЛ И ИИ
    # ============================================================

    df["final_anomaly"] = (
        (df["rule_anomaly"] == 1) |
        (df["iforest_anomaly"] == 1)
    ).astype(int)

    mask_ai_only = (
        (df["iforest_anomaly"] == 1) &
        (df["rule_anomaly"] == 0)
    )

    df.loc[mask_ai_only, "rule_event_type"] = "Нетипичное поведение по ИИ-модели"
    df.loc[mask_ai_only, "rule_risk_level"] = "Warning"
    df.loc[mask_ai_only, "rule_recommendation"] = (
        "Проверить участок графика и сравнить с другими температурными каналами"
    )

    # Уточнение уровня риска по anomaly score
    mask_high_score = df["anomaly_score_norm"] > 0.85

    df.loc[
        (df["final_anomaly"] == 1) & mask_high_score,
        "rule_risk_level"
    ] = "High"

    mask_medium_score = (
        (df["anomaly_score_norm"] > 0.60) &
        (df["anomaly_score_norm"] <= 0.85)
    )

    df.loc[
        (df["final_anomaly"] == 1) &
        mask_medium_score &
        (df["rule_risk_level"] == "Normal"),
        "rule_risk_level"
    ] = "Medium"

    # ============================================================
    # 4. ЖУРНАЛ ТРЕВОГ
    # ============================================================

    alarm_log = df[df["final_anomaly"] == 1].copy()

    alarm_log = alarm_log[
        [
            "timestamp",
            "sensor_id",
            "temperature",
            "temperature_filled",
            "rule_event_type",
            "rule_risk_level",
            "anomaly_score_norm",
            "rule_recommendation",
            "scenario"
        ]
    ]

    alarm_log = alarm_log.rename(columns={
        "timestamp": "Время",
        "sensor_id": "Датчик",
        "temperature": "Температура",
        "temperature_filled": "Температура_заполненная",
        "rule_event_type": "Тип_события",
        "rule_risk_level": "Уровень",
        "anomaly_score_norm": "Anomaly_score",
        "rule_recommendation": "Рекомендация",
        "scenario": "Истинный_сценарий"
    })

    alarm_log["Температура"] = alarm_log["Температура"].round(2)
    alarm_log["Температура_заполненная"] = alarm_log["Температура_заполненная"].round(2)
    alarm_log["Anomaly_score"] = alarm_log["Anomaly_score"].round(3)

    return df, alarm_log


if __name__ == "__main__":
    input_file = "preprocessed_temperature_data.csv"
    results_file = "temperature_anomaly_results.csv"
    alarm_log_file = "alarm_log.csv"

    df = pd.read_csv(input_file)

    try:
        results_df, alarm_log = detect_anomalies(df)
    except ModelNotTrainedError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    results_df.to_csv(results_file, index=False, encoding="utf-8-sig")
    alarm_log.to_csv(alarm_log_file, index=False, encoding="utf-8-sig")

    print("\nФинальных аномалий найдено:")
    print(results_df["final_anomaly"].sum())

    print(f"\nФайл с полными результатами сохранён: {results_file}")
    print(f"Журнал тревог сохранён: {alarm_log_file}")

    print("\nПервые строки журнала тревог:")
    print(alarm_log.head(20))
