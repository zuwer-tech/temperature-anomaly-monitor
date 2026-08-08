"""Обучение модели Isolation Forest на нормальных данных и её сохранение.

Раньше Isolation Forest обучался прямо на тех данных, которые оценивал
(`anomaly_detection.detect_anomalies` делал `fit_predict` на всём DataFrame).
Это data leakage: модель «видела» аномалии при обучении и не училась отличать
штатный режим от настоящих отклонений. Кроме того, модель нигде не
сохранялась — каждый запуск обучал её заново.

Этот скрипт исправляет оба недостатка:
- обучает StandardScaler + Isolation Forest ТОЛЬКО на строках со сценарием
  `normal` (модель учит штатный режим, а аномалии становятся для неё
  «нетипичными»);
- сохраняет артефакты (`models/scaler.joblib`, `models/iforest.joblib`,
  `models/model_meta.json`), которые потом загружает `detect_anomalies`;
- печатает отчёт precision/recall/F1 против истинных сценариев.

Запуск::

    python preprocessing.py     # если ещё нет preprocessed_temperature_data.csv
    python train_model.py        # обучить и сохранить модель
"""
import argparse
import json
import os
from datetime import datetime

from model_schema import (
    FEATURE_COLUMNS,
    METADATA_VERSION,
    RISK_HIGH_THRESHOLD,
    RISK_MEDIUM_THRESHOLD,
    SCORE_CALIBRATION_METHOD,
    prepare_ml_features,
)

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MODEL_DIR = "models"
DEFAULT_INPUT = "preprocessed_temperature_data.csv"
DEFAULT_CONTAMINATION = 0.04
RANDOM_STATE = 42
N_ESTIMATORS = 200
VALIDATION_START = "2026-06-06 14:00:00"
TEST_START = "2026-06-06 17:00:00"
SPLIT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
SPLIT_STRATEGY = "time_train_validation_test"


def parse_split_boundary(value, name):
    """Проверяет явную временную границу и возвращает pandas Timestamp."""
    if not isinstance(value, str):
        raise ValueError(
            f"Граница {name} должна быть строкой в формате "
            "YYYY-MM-DD HH:MM:SS."
        )
    try:
        parsed = datetime.strptime(value, SPLIT_TIME_FORMAT)
    except ValueError as exc:
        raise ValueError(
            f"Некорректная граница {name}. Ожидается формат "
            f"YYYY-MM-DD HH:MM:SS, получено: {value!r}."
        ) from exc
    if parsed.strftime(SPLIT_TIME_FORMAT) != value:
        raise ValueError(
            f"Некорректная граница {name}. Ожидается точный формат "
            f"YYYY-MM-DD HH:MM:SS, получено: {value!r}."
        )
    return pd.Timestamp(parsed)

def prepare_features(df):
    """Готовит тот же безопасный ML-вход, что и detect_anomalies."""
    prepared, X, _ml_eligible = prepare_ml_features(df)
    return prepared, X


def _normal_mask(df):
    """Возвращает только подтверждённые normal-строки или останавливает обучение."""
    if "scenario" not in df.columns:
        raise ValueError(
            "Обучение разрешено только на подтверждённых normal-данных: "
            "отсутствует обязательная колонка scenario."
        )

    mask = df["scenario"] == "normal"
    if not mask.any():
        raise ValueError(
            "Обучение разрешено только на подтверждённых normal-данных: "
            "нет строк scenario == 'normal'."
        )
    return mask, True


def _partition_counts(frame, prefix):
    """Возвращает размеры normal/anomaly для одной временной части."""
    normal_rows = int((frame["scenario"] == "normal").sum())
    return {
        f"{prefix}_rows": int(len(frame)),
        f"{prefix}_normal_rows": normal_rows,
        f"{prefix}_anomaly_rows": int(len(frame) - normal_rows),
    }


def split_train_validation_test(
    df,
    validation_start=VALIDATION_START,
    test_start=TEST_START,
):
    """Причинно готовит признаки и делит ряд на train, validation и test.

    Признаки рассчитываются на полном прошлом ряду до временных масок. Train
    содержит только подтверждённые normal-строки. Validation предназначен для
    выбора параметров, а самый поздний test — только для финального отчёта.

    Возвращает train/validation/test DataFrame, три матрицы признаков и info.
    """
    prepared_df, X = prepare_features(df)
    validation_boundary = parse_split_boundary(
        validation_start,
        "validation_start",
    )
    test_boundary = parse_split_boundary(test_start, "test_start")
    if validation_boundary >= test_boundary:
        raise ValueError(
            "Граница validation_start должна быть раньше test_start."
        )

    normal_mask, used_normal = _normal_mask(prepared_df)
    ml_eligible = prepared_df.index.isin(X.index)
    train_mask = (
        (prepared_df["timestamp"] < validation_boundary)
        & normal_mask
        & ml_eligible
    )
    validation_mask = (
        (prepared_df["timestamp"] >= validation_boundary)
        & (prepared_df["timestamp"] < test_boundary)
    )
    test_mask = prepared_df["timestamp"] >= test_boundary

    if not train_mask.any():
        raise ValueError(
            "В train нет подтверждённых normal-строк до границы "
            f"{validation_boundary.isoformat()}."
        )
    if not validation_mask.any():
        raise ValueError(
            "В validation нет строк между границами "
            f"{validation_boundary.isoformat()} и {test_boundary.isoformat()}."
        )
    if not test_mask.any():
        raise ValueError(
            "В test нет строк начиная с границы "
            f"{test_boundary.isoformat()}."
        )

    train_df = prepared_df.loc[train_mask].copy()
    validation_df = prepared_df.loc[validation_mask].copy()
    test_df = prepared_df.loc[test_mask].copy()
    X_train = X.loc[train_df.index].copy()
    X_validation = X.loc[
        X.index.intersection(validation_df.index, sort=False)
    ].copy()
    X_test = X.loc[
        X.index.intersection(test_df.index, sort=False)
    ].copy()

    info = {
        "train_rows": int(len(train_df)),
        "trained_on_normal": used_normal,
        "split_strategy": SPLIT_STRATEGY,
        "validation_start": validation_boundary.isoformat(),
        **_partition_counts(validation_df, "validation"),
        "test_start": test_boundary.isoformat(),
        **_partition_counts(test_df, "test"),
    }
    return (
        train_df,
        validation_df,
        test_df,
        X_train,
        X_validation,
        X_test,
        info,
    )

def train(
    df,
    contamination=DEFAULT_CONTAMINATION,
    random_state=RANDOM_STATE,
    validation_start=VALIDATION_START,
    test_start=TEST_START,
):
    """Обучает scaler+IsolationForest на раннем штатном режиме.

    Возвращает (scaler, model, info) где info — словарь с числом train-строк и
    сведениями о временном разбиении.
    """
    (
        _train_df,
        _validation_df,
        _test_df,
        X_train,
        _X_validation,
        _X_test,
        info,
    ) = split_train_validation_test(
        df,
        validation_start=validation_start,
        test_start=test_start,
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=contamination,
        random_state=random_state,
    ).fit(X_train_scaled)

    train_scores = -model.decision_function(X_train_scaled)
    score_min = float(np.min(train_scores))
    score_max = float(np.max(train_scores))
    if not np.isfinite(score_min) or not np.isfinite(score_max) or score_max <= score_min:
        raise ValueError(
            "Не удалось построить шкалу anomaly score по обучающему baseline."
        )
    info["score_calibration"] = {
        "method": SCORE_CALIBRATION_METHOD,
        "score_min": score_min,
        "score_max": score_max,
        "medium_threshold": RISK_MEDIUM_THRESHOLD,
        "high_threshold": RISK_HIGH_THRESHOLD,
    }

    return scaler, model, info


def _iforest_partition_report(frame, features, scaler, model):
    """Считает Isolation Forest метрики для одной неизменяемой части."""
    pred = pd.Series(0, index=frame.index, dtype=int)
    if not features.empty:
        pred.loc[features.index] = (
            model.predict(scaler.transform(features)) == -1
        ).astype(int)

    report = {
        "rows": int(len(frame)),
        "normal_rows": int((frame["scenario"] == "normal").sum()),
        "anomaly_rows": int((frame["scenario"] != "normal").sum()),
        "iforest_anomalies": int(pred.sum()),
    }
    gt = (frame["scenario"] != "normal").astype(int)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    report.update({
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    })
    per_scenario = {}
    for scenario, sub in frame.groupby("scenario"):
        scenario_pred = pred.loc[sub.index]
        total = int(len(sub))
        detected = int(scenario_pred.sum())
        per_scenario[scenario] = {
            "total": total,
            "detected": detected,
            "recall": round(detected / total, 3) if total else 0.0,
        }
    report["per_scenario"] = per_scenario
    return report


def evaluate(
    df,
    scaler,
    model,
    validation_start=VALIDATION_START,
    test_start=TEST_START,
):
    """Считает рабочие validation и финальные test метрики отдельно."""
    (
        _train_df,
        validation_df,
        test_df,
        _X_train,
        X_validation,
        X_test,
        split_info,
    ) = split_train_validation_test(
        df,
        validation_start=validation_start,
        test_start=test_start,
    )
    return {
        "split": split_info,
        "validation": _iforest_partition_report(
            validation_df,
            X_validation,
            scaler,
            model,
        ),
        "test": _iforest_partition_report(
            test_df,
            X_test,
            scaler,
            model,
        ),
        "final_report_dataset": "test",
    }

def save_model(scaler, model, info, model_dir=MODEL_DIR, contamination=DEFAULT_CONTAMINATION):
    os.makedirs(model_dir, exist_ok=True)
    dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    dump(model, os.path.join(model_dir, "iforest.joblib"))
    meta = {
        "metadata_version": METADATA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "contamination": contamination,
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "train_rows": info["train_rows"],
        "trained_on_normal": info["trained_on_normal"],
        "split_strategy": info["split_strategy"],
        "validation_start": info["validation_start"],
        "validation_rows": info["validation_rows"],
        "validation_normal_rows": info["validation_normal_rows"],
        "validation_anomaly_rows": info["validation_anomaly_rows"],
        "test_start": info["test_start"],
        "test_rows": info["test_rows"],
        "test_normal_rows": info["test_normal_rows"],
        "test_anomaly_rows": info["test_anomaly_rows"],
        "score_calibration": info["score_calibration"],
    }
    with open(os.path.join(model_dir, "model_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)


def main(
    input_file=DEFAULT_INPUT,
    contamination=DEFAULT_CONTAMINATION,
    validation_start=VALIDATION_START,
    test_start=TEST_START,
):
    df = pd.read_csv(input_file)
    scaler, model, info = train(
        df,
        contamination=contamination,
        validation_start=validation_start,
        test_start=test_start,
    )
    save_model(scaler, model, info, contamination=contamination)
    report = evaluate(
        df,
        scaler,
        model,
        validation_start=validation_start,
        test_start=test_start,
    )
    print("Модель обучена.")
    for key in (
        "train_rows",
        "validation_rows",
        "validation_normal_rows",
        "validation_anomaly_rows",
        "test_rows",
        "test_normal_rows",
        "test_anomaly_rows",
        "split_strategy",
        "validation_start",
        "test_start",
        "trained_on_normal",
    ):
        print(f"{key} = {info[key]}")
    print("Артефакты сохранены в:", MODEL_DIR)
    print("Отчёт по качеству (Isolation Forest):")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Обучение Isolation Forest на раннем normal-baseline."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--contamination",
        type=float,
        default=DEFAULT_CONTAMINATION,
    )
    parser.add_argument(
        "--validation-start",
        default=VALIDATION_START,
        help=(
            "Начало validation в формате YYYY-MM-DD HH:MM:SS. "
            "По умолчанию 2026-06-06 14:00:00."
        ),
    )
    parser.add_argument(
        "--test-start",
        default=TEST_START,
        help=(
            "Начало закрытого test в формате YYYY-MM-DD HH:MM:SS. "
            "По умолчанию 2026-06-06 17:00:00."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        input_file=args.input,
        contamination=args.contamination,
        validation_start=args.validation_start,
        test_start=args.test_start,
    )
