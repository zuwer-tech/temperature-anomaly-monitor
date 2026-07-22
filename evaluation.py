"""Единый воспроизводимый отчёт качества правил, ML и их объединения.

Скрипт не обучает модель: он загружает заранее сохранённые артефакты,
выполняет обычный инференс и считает метрики только на поздней
evaluation-части временного ряда.
"""
import argparse
import json

import numpy as np
import pandas as pd

from anomaly_detection import detect_anomalies
from train_model import DEFAULT_INPUT, MODEL_DIR, TEST_START, split_train_evaluation


LAYER_COLUMNS = {
    "rules": "rule_anomaly",
    "ml": "iforest_anomaly",
    "combined": "final_anomaly",
}


def binary_classification_metrics(y_true, y_pred):
    """Возвращает confusion matrix и precision/recall/F1 для двух классов."""
    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)

    if truth.ndim != 1 or predicted.ndim != 1:
        raise ValueError("y_true и y_pred должны быть одномерными.")
    if truth.shape != predicted.shape:
        raise ValueError("y_true и y_pred должны иметь одинаковую длину.")
    if not set(np.unique(truth)).issubset({0, 1}):
        raise ValueError("y_true должен содержать только 0 и 1.")
    if not set(np.unique(predicted)).issubset({0, 1}):
        raise ValueError("y_pred должен содержать только 0 и 1.")

    tp = int(((predicted == 1) & (truth == 1)).sum())
    fp = int(((predicted == 1) & (truth == 0)).sum())
    fn = int(((predicted == 0) & (truth == 1)).sum())
    tn = int(((predicted == 0) & (truth == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_detection_layers(df, model_dir=MODEL_DIR, test_start=TEST_START):
    """Считает единый отчёт на одной независимой evaluation-выборке."""
    if "scenario" not in df.columns:
        raise ValueError(
            "Для оценки нужна колонка scenario с эталонной разметкой."
        )

    detected, _alarm_log = detect_anomalies(df, model_dir=model_dir)
    _train, evaluation, _x_train, _x_evaluation, split_info = (
        split_train_evaluation(detected, test_start=test_start)
    )


    truth = (evaluation["scenario"] != "normal").astype(int)
    layers = {
        layer: binary_classification_metrics(
            truth,
            evaluation[column].astype(int),
        )
        for layer, column in LAYER_COLUMNS.items()
    }

    return {
        "positive_class": "scenario != normal",
        "split": split_info,
        "layers": layers,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Отчёт качества правил, ML и объединённой системы."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument(
        "--output",
        help="Необязательный путь для сохранения JSON-отчёта.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    report = evaluate_detection_layers(df, model_dir=args.model_dir)
    report_json = json.dumps(report, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(report_json + "\n")

    print(report_json)


if __name__ == "__main__":
    main()
