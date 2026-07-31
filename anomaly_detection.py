import json
import os
import sys

from model_schema import (
    FEATURE_COLUMNS,
    METADATA_VERSION,
    SCORE_CALIBRATION_METHOD,
    ML_STATUS_APPLIED,
    ML_STATUS_NOT_APPLIED_RULES_ONLY,
    ML_STATUS_SKIPPED_MISSING_TEMPERATURE,
    prepare_ml_features,
)
from rule_config import RULE_PARAMS
from time_windows import causal_time_slope
from risk_config import assess_risk
from events import group_anomaly_events

import pandas as pd
import numpy as np

from joblib import load


class ModelNotTrainedError(RuntimeError):
    """Обученная модель отсутствует или сохранена не полностью."""


class ModelCompatibilityError(RuntimeError):
    """Сохранённая модель повреждена или несовместима с текущим кодом."""

ANALYSIS_MODE_RULES_ONLY = "rules-only"
ANALYSIS_MODE_RULES_ML = "rules+ML"

PRIMARY_REASON_PRIORITY = (
    "signal_loss",
    "sharp_jump",
    "stuck_sensor",
    "sustained_overheat",
    "group_deviation",
    "z_score",
)


REQUIRED_METADATA_FIELDS = (
    "feature_columns",
    "contamination",
    "random_state",
    "n_estimators",
    "train_rows",
    "trained_on_normal",
    "split_strategy",
    "test_start",
    "evaluation_rows",
    "evaluation_normal_rows",
    "evaluation_anomaly_rows",
    "score_calibration",
)
RETRAIN_COMMANDS = "python preprocessing.py\npython train_model.py"



def _record_triggered_rule(df, mask, rule_id):
    """Добавляет идентификатор правила, не стирая уже найденные причины."""
    for index in df.index[mask.fillna(False)]:
        df.at[index, "triggered_rules"] = [
            *df.at[index, "triggered_rules"],
            rule_id,
        ]

def _select_primary_reason(triggered_rules, iforest_anomaly):
    """Выбирает главную причину по явному инженерному приоритету."""
    for reason in PRIMARY_REASON_PRIORITY:
        if reason in triggered_rules:
            return reason
    if iforest_anomaly:
        return "iforest_anomaly"
    return None

def _compatibility_error(message):
    return ModelCompatibilityError(
        f"{message} Переобучите модель:\n{RETRAIN_COMMANDS}"
    )


def _validate_score_calibration(calibration):
    """Проверяет сохранённую шкалу score и пороги риска."""
    if not isinstance(calibration, dict):
        raise _compatibility_error(
            "Поле score_calibration должно содержать JSON-объект."
        )

    required_fields = (
        "method",
        "score_min",
        "score_max",
        "medium_threshold",
        "high_threshold",
    )
    missing_fields = [
        field for field in required_fields if field not in calibration
    ]
    if missing_fields:
        raise _compatibility_error(
            "В score_calibration отсутствуют обязательные поля: "
            f"{', '.join(missing_fields)}."
        )

    if calibration["method"] != SCORE_CALIBRATION_METHOD:
        raise _compatibility_error(
            "Метод калибровки anomaly score несовместим с текущим кодом."
        )

    numeric_fields = (
        "score_min",
        "score_max",
        "medium_threshold",
        "high_threshold",
    )
    for field in numeric_fields:
        value = calibration[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise _compatibility_error(
                f"Поле score_calibration.{field} должно быть конечным числом."
            )

    if calibration["score_max"] <= calibration["score_min"]:
        raise _compatibility_error(
            "Границы калибровки score должны удовлетворять score_max > score_min."
        )
    if not (
        0 <= calibration["medium_threshold"]
        < calibration["high_threshold"]
        <= 1
    ):
        raise _compatibility_error(
            "Пороги риска должны удовлетворять "
            "0 <= medium_threshold < high_threshold <= 1."
        )


def _load_and_validate_model_metadata(model_dir):
    """Читает metadata и проверяет совместимость модели."""
    path = os.path.join(model_dir, "model_meta.json")
    try:
        with open(path, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _compatibility_error(
            f"Не удалось прочитать model_meta.json: {error}."
        ) from None

    if not isinstance(metadata, dict):
        raise _compatibility_error(
            "model_meta.json должен содержать JSON-объект верхнего уровня."
        )

    if metadata.get("metadata_version") != METADATA_VERSION:
        raise _compatibility_error(
            "Версия metadata модели не поддерживается: "
            f"ожидалась {METADATA_VERSION}, получена "
            f"{metadata.get('metadata_version')!r}."
        )

    missing_fields = [
        field for field in REQUIRED_METADATA_FIELDS if field not in metadata
    ]
    if missing_fields:
        raise _compatibility_error(
            "В model_meta.json отсутствуют обязательные поля: "
            f"{', '.join(missing_fields)}."
        )

    actual_features = metadata["feature_columns"]
    expected_features = list(FEATURE_COLUMNS)
    if (
        not isinstance(actual_features, list)
        or not all(isinstance(column, str) for column in actual_features)
        or actual_features != expected_features
    ):
        raise _compatibility_error(
            "Набор признаков модели несовместим с текущим кодом. "
            f"Ожидался список: {expected_features}. "
            f"Фактический список: {actual_features!r}."
        )

    _validate_score_calibration(metadata["score_calibration"])
    return metadata


def _load_joblib_artifact(path, filename):
    try:
        return load(path)
    except Exception as error:
        raise _compatibility_error(
            f"Не удалось загрузить {filename}: {error}."
        ) from None


def _validate_model_objects(scaler, model):
    """Проверяет интерфейсы и число входных признаков объектов joblib."""
    if not callable(getattr(scaler, "transform", None)):
        raise _compatibility_error(
            "scaler.joblib не содержит объект с методом transform()."
        )

    for method_name in ("predict", "decision_function"):
        if not callable(getattr(model, method_name, None)):
            raise _compatibility_error(
                "iforest.joblib не содержит объект с методом "
                f"{method_name}()."
            )

    expected_features = len(FEATURE_COLUMNS)
    for filename, artifact in (
        ("scaler.joblib", scaler),
        ("iforest.joblib", model),
    ):
        if hasattr(artifact, "n_features_in_"):
            actual_features = artifact.n_features_in_
            if actual_features != expected_features:
                raise _compatibility_error(
                    f"{filename} ожидает несовместимое число признаков: "
                    f"ожидалось {expected_features}, получено {actual_features}."
                )


def _load_or_fit_iforest(X, model_dir="models"):
    """Загружает проверенную модель и выполняет инференс без переобучения."""
    artifacts = {
        "scaler.joblib": os.path.join(model_dir, "scaler.joblib"),
        "iforest.joblib": os.path.join(model_dir, "iforest.joblib"),
        "model_meta.json": os.path.join(model_dir, "model_meta.json"),
    }
    missing = [name for name, path in artifacts.items() if not os.path.isfile(path)]

    if missing:
        raise ModelNotTrainedError(
            "Обученная модель Isolation Forest не найдена или комплект "
            "артефактов неполный. Отсутствуют файлы: "
            f"{', '.join(missing)}. Сначала выполните:\n"
            f"{RETRAIN_COMMANDS}"
        )

    metadata = _load_and_validate_model_metadata(model_dir)
    scaler = _load_joblib_artifact(artifacts["scaler.joblib"], "scaler.joblib")
    model = _load_joblib_artifact(artifacts["iforest.joblib"], "iforest.joblib")
    _validate_model_objects(scaler, model)
    if X.empty:
        empty_scaled = np.empty((0, len(FEATURE_COLUMNS)))
        return empty_scaled, np.array([], dtype=int), np.array([]), metadata
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    score_raw = model.decision_function(X_scaled)
    return X_scaled, predictions, score_raw, metadata


def _normalize_anomaly_score(scores, calibration):
    """Переводит raw score на фиксированную шкалу обучающего baseline."""
    normalized = (
        (scores - calibration["score_min"])
        / (calibration["score_max"] - calibration["score_min"])
    )
    return np.clip(normalized, 0, 1)


def detect_anomalies(df, model_dir="models", use_ml=True):
    """
    Обнаружение температурных аномалий.

    На вход получает предобработанный DataFrame.
    На выход возвращает:
    - df с результатами анализа
    - alarm_log с журналом тревог

    use_ml=False явно включает режим только инженерных правил. Автоматического
    перехода в этот режим при ошибке модели нет.
    """

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["sensor_id", "timestamp"]).reset_index(drop=True)

    # ============================================================
    # 1. ОБНАРУЖЕНИЕ АНОМАЛИЙ ПО ПРАВИЛАМ
    # ============================================================

    df["rule_anomaly"] = 0
    df["triggered_rules"] = [[] for _ in range(len(df))]
    df["rule_event_type"] = "normal"

    # Потеря сигнала
    mask_missing = df["is_missing"] == 1
    _record_triggered_rule(df, mask_missing, "signal_loss")

    df.loc[mask_missing, "rule_anomaly"] = 1
    df.loc[mask_missing, "rule_event_type"] = "Потеря сигнала"

    # Резкий скачок температуры
    mask_sharp_jump = (
        df["temp_rate_c_per_min"].abs()
        > RULE_PARAMS["sharp_jump_rate_c_per_min"]
    )
    _record_triggered_rule(df, mask_sharp_jump, "sharp_jump")

    df.loc[mask_sharp_jump, "rule_anomaly"] = 1
    df.loc[mask_sharp_jump, "rule_event_type"] = "Резкий скачок температуры"

    # Сильное отклонение от обычного режима датчика
    mask_z_score = df["abs_z_score"] > RULE_PARAMS["z_score"]
    _record_triggered_rule(df, mask_z_score, "z_score")

    df.loc[mask_z_score, "rule_anomaly"] = 1
    df.loc[mask_z_score, "rule_event_type"] = "Сильное отклонение от нормы"

    # Зависший датчик
    mask_stuck = df["is_stuck"] == 1
    _record_triggered_rule(df, mask_stuck, "stuck_sensor")

    df.loc[mask_stuck, "rule_anomaly"] = 1
    df.loc[mask_stuck, "rule_event_type"] = "Зависание датчика"

    # Отклонение от группы датчиков
    mask_group_deviation = df["abs_diff_from_group_mean"] > RULE_PARAMS["group_deviation"]
    _record_triggered_rule(df, mask_group_deviation, "group_deviation")

    df.loc[mask_group_deviation, "rule_anomaly"] = 1
    df.loc[mask_group_deviation, "rule_event_type"] = "Отклонение от группы датчиков"

    # Наклон сглаженной температуры считается по фактическому времени, °C/мин.
    # Окно причинное: текущая и прошлые точки. Разрыв длиннее допустимого
    # начинает новый участок, поэтому неизвестный промежуток не выдаётся за
    # устойчивый физический рост.
    df["temp_slope_overheat"] = causal_time_slope(
        df,
        "rolling_mean",
        RULE_PARAMS["overheat_duration_seconds"],
        RULE_PARAMS["continuity_gap_seconds"],
    )

    mask_overheat = (
        df["temp_slope_overheat"]
        > RULE_PARAMS["overheat_slope_c_per_min"]
    )
    _record_triggered_rule(df, mask_overheat, "sustained_overheat")
    df.loc[mask_overheat, "rule_anomaly"] = 1
    df.loc[mask_overheat, "rule_event_type"] = "Устойчивый перегрев"

    # ============================================================
    # 2. ISOLATION FOREST
    # ============================================================

    if use_ml:
        df, X, _ml_eligible = prepare_ml_features(df)
        (
            _X_scaled,
            iforest_prediction,
            iforest_score_raw,
            metadata,
        ) = _load_or_fit_iforest(X, model_dir=model_dir)

        df["iforest_prediction"] = np.nan
        df["iforest_anomaly"] = 0
        df["iforest_score_raw"] = np.nan
        df["anomaly_score"] = np.nan
        df["anomaly_score_norm"] = np.nan
        df["ml_inference_status"] = (
            ML_STATUS_SKIPPED_MISSING_TEMPERATURE
        )

        if not X.empty:
            df.loc[X.index, "iforest_prediction"] = iforest_prediction
            df.loc[X.index, "iforest_anomaly"] = (
                iforest_prediction == -1
            ).astype(int)
            df.loc[X.index, "iforest_score_raw"] = iforest_score_raw
            anomaly_score = -iforest_score_raw
            df.loc[X.index, "anomaly_score"] = anomaly_score
            df.loc[X.index, "anomaly_score_norm"] = (
                _normalize_anomaly_score(
                    anomaly_score,
                    metadata["score_calibration"],
                )
            )
            df.loc[X.index, "ml_inference_status"] = ML_STATUS_APPLIED

        calibration = metadata["score_calibration"]
        risk_medium_threshold = calibration["medium_threshold"]
        risk_high_threshold = calibration["high_threshold"]
        analysis_mode = ANALYSIS_MODE_RULES_ML
    else:
        df["iforest_prediction"] = np.nan
        df["iforest_anomaly"] = 0
        df["iforest_score_raw"] = np.nan
        df["anomaly_score"] = np.nan
        df["anomaly_score_norm"] = np.nan
        df["ml_inference_status"] = ML_STATUS_NOT_APPLIED_RULES_ONLY
        risk_medium_threshold = np.nan
        risk_high_threshold = np.nan
        analysis_mode = ANALYSIS_MODE_RULES_ONLY

    df["analysis_mode"] = analysis_mode
    df["rule_count"] = df["triggered_rules"].map(len)
    df["primary_reason"] = [
        _select_primary_reason(rules, iforest_anomaly)
        for rules, iforest_anomaly in zip(
            df["triggered_rules"],
            df["iforest_anomaly"],
        )
    ]

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

    # Единая матрица выбирает самый высокий риск из всех независимых сигналов.
    risk_assessments = [
        assess_risk(
            rules,
            iforest_anomaly=iforest_anomaly,
            anomaly_score_norm=score,
            medium_threshold=risk_medium_threshold,
            high_threshold=risk_high_threshold,
        )
        for rules, iforest_anomaly, score in zip(
            df["triggered_rules"],
            df["iforest_anomaly"],
            df["anomaly_score_norm"],
        )
    ]
    df["rule_risk_level"] = [level for level, _ in risk_assessments]
    df["rule_recommendation"] = [
        recommendation for _, recommendation in risk_assessments
    ]
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
            "triggered_rules",
            "primary_reason",
            "rule_count",
            "rule_event_type",
            "rule_risk_level",
            "anomaly_score_norm",
            "rule_recommendation",
            "scenario",
            "ml_inference_status",
            "analysis_mode",
        ]
    ]

    alarm_log = alarm_log.rename(columns={
        "timestamp": "Время",
        "sensor_id": "Датчик",
        "temperature": "Температура",
        "temperature_filled": "Температура_заполненная",
        "triggered_rules": "Сработавшие_правила",
        "rule_event_type": "Тип_события",
        "rule_risk_level": "Уровень",
        "anomaly_score_norm": "Anomaly_score",
        "rule_recommendation": "Рекомендация",
        "scenario": "Истинный_сценарий",
        "ml_inference_status": "Статус_ML",
        "analysis_mode": "Режим_анализа",
    })

    alarm_log["Температура"] = alarm_log["Температура"].round(2)
    alarm_log["Температура_заполненная"] = alarm_log["Температура_заполненная"].round(2)
    alarm_log["Anomaly_score"] = alarm_log["Anomaly_score"].round(3)

    return df, alarm_log


if __name__ == "__main__":
    input_file = "preprocessed_temperature_data.csv"
    results_file = "temperature_anomaly_results.csv"
    alarm_log_file = "alarm_log.csv"
    event_log_file = "event_log.csv"

    df = pd.read_csv(input_file)

    try:
        results_df, alarm_log = detect_anomalies(df)
        event_log = group_anomaly_events(results_df)
    except (ModelNotTrainedError, ModelCompatibilityError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    results_df.to_csv(results_file, index=False, encoding="utf-8-sig")
    alarm_log.to_csv(alarm_log_file, index=False, encoding="utf-8-sig")
    event_log.to_csv(event_log_file, index=False, encoding="utf-8-sig")

    print("\nФинальных аномалий найдено:")
    print(results_df["final_anomaly"].sum())

    print(f"\nФайл с полными результатами сохранён: {results_file}")
    print(f"Журнал тревог сохранён: {alarm_log_file}")
    print(f"Журнал событий сохранён: {event_log_file}")

    print("\nПервые строки журнала тревог:")
    print(alarm_log.head(20))
