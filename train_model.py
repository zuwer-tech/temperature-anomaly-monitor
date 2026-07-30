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
import json
import os

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
TEST_START = "2026-06-06 14:00:00"
SPLIT_STRATEGY = "time"


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


def split_train_evaluation(df, test_start=TEST_START):
    """Готовит признаки и делит данные на ранний train и поздний evaluation.

    Признаки рассчитываются до применения масок, поэтому первые строки
    evaluation используют доступную прошлую rolling-историю train. В train
    попадают только normal-строки до фиксированной временной границы.

    Возвращает ``(train_df, evaluation_df, X_train, X_evaluation, info)``.
    """
    prepared_df, X = prepare_features(df)
    boundary = pd.Timestamp(test_start)
    normal_mask, used_normal = _normal_mask(prepared_df)
    ml_eligible = prepared_df.index.isin(X.index)
    train_mask = (
        (prepared_df["timestamp"] < boundary)
        & normal_mask
        & ml_eligible
    )
    evaluation_mask = prepared_df["timestamp"] >= boundary

    if not train_mask.any():
        raise ValueError(
            "В train нет строк до временной границы "
            f"{boundary.isoformat()}."
        )
    if not evaluation_mask.any():
        raise ValueError(
            "В evaluation нет строк начиная с временной границы "
            f"{boundary.isoformat()}."
        )

    train_df = prepared_df.loc[train_mask].copy()
    evaluation_df = prepared_df.loc[evaluation_mask].copy()
    X_train = X.loc[train_df.index].copy()
    evaluation_ml_index = X.index.intersection(
        evaluation_df.index,
        sort=False,
    )
    X_evaluation = X.loc[evaluation_ml_index].copy()

    if "scenario" in evaluation_df.columns:
        evaluation_normal_rows = int(
            (evaluation_df["scenario"] == "normal").sum()
        )
        evaluation_anomaly_rows = int(
            (evaluation_df["scenario"] != "normal").sum()
        )
    else:
        evaluation_normal_rows = 0
        evaluation_anomaly_rows = 0

    info = {
        "train_rows": int(len(train_df)),
        "trained_on_normal": used_normal,
        "split_strategy": SPLIT_STRATEGY,
        "test_start": boundary.isoformat(),
        "evaluation_rows": int(len(evaluation_df)),
        "evaluation_normal_rows": evaluation_normal_rows,
        "evaluation_anomaly_rows": evaluation_anomaly_rows,
    }
    return train_df, evaluation_df, X_train, X_evaluation, info


def train(df, contamination=DEFAULT_CONTAMINATION, random_state=RANDOM_STATE):
    """Обучает scaler+IsolationForest на раннем штатном режиме.

    Возвращает (scaler, model, info) где info — словарь с числом train-строк и
    сведениями о временном разбиении.
    """
    _train_df, _evaluation_df, X_train, _X_evaluation, info = (
        split_train_evaluation(df)
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


def evaluate(df, scaler, model):
    """Считает метрики только на более поздней evaluation-части."""
    _train_df, evaluation_df, _X_train, X_evaluation, split_info = (
        split_train_evaluation(df)
    )
    pred = pd.Series(0, index=evaluation_df.index, dtype=int)
    if not X_evaluation.empty:
        pred.loc[X_evaluation.index] = (
            model.predict(scaler.transform(X_evaluation)) == -1
        ).astype(int)
    report = {
        "iforest_anomalies": int(pred.sum()),
        "train_rows": split_info["train_rows"],
        "evaluation_rows": split_info["evaluation_rows"],
        "evaluation_normal_rows": split_info["evaluation_normal_rows"],
        "evaluation_anomaly_rows": split_info["evaluation_anomaly_rows"],
        "split_strategy": split_info["split_strategy"],
        "test_start": split_info["test_start"],
    }
    if "scenario" not in evaluation_df.columns:
        return report
    gt = (evaluation_df["scenario"] != "normal").astype(int)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    report.update({
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    })
    per_scenario = {}
    for scenario, sub in evaluation_df.groupby("scenario"):
        sp = pred.loc[sub.index]
        total = int(len(sub))
        detected = int(sp.sum())
        per_scenario[scenario] = {
            "total": total,
            "detected": detected,
            "recall": round(detected / total, 3) if total else 0.0,
        }
    report["per_scenario"] = per_scenario
    return report


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
        "test_start": info["test_start"],
        "evaluation_rows": info["evaluation_rows"],
        "evaluation_normal_rows": info["evaluation_normal_rows"],
        "evaluation_anomaly_rows": info["evaluation_anomaly_rows"],
        "score_calibration": info["score_calibration"],
    }
    with open(os.path.join(model_dir, "model_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)


def main(input_file=DEFAULT_INPUT, contamination=DEFAULT_CONTAMINATION):
    df = pd.read_csv(input_file)
    scaler, model, info = train(df, contamination=contamination)
    save_model(scaler, model, info, contamination=contamination)
    report = evaluate(df, scaler, model)
    print("Модель обучена.")
    for key in (
        "train_rows",
        "evaluation_rows",
        "evaluation_normal_rows",
        "evaluation_anomaly_rows",
        "split_strategy",
        "test_start",
        "trained_on_normal",
    ):
        print(f"{key} = {info[key]}")
    print("Артефакты сохранены в:", MODEL_DIR)
    print("Отчёт по качеству (Isolation Forest):")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
