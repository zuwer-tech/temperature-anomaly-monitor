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
from train_model import (
    DEFAULT_INPUT,
    MODEL_DIR,
    TEST_START,
    split_train_evaluation,
)


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


def detection_delay_metrics(evaluation, prediction_column):
    """Измеряет задержку до первой тревоги для каждого размеченного события."""
    required_columns = {
        "timestamp",
        "sensor_id",
        "scenario",
        prediction_column,
    }
    missing_columns = sorted(required_columns - set(evaluation.columns))
    if missing_columns:
        raise ValueError(
            "Для расчёта задержки отсутствуют колонки: "
            f"{', '.join(missing_columns)}."
        )

    events_df = evaluation[list(required_columns)].copy()
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
    events_df[prediction_column] = events_df[prediction_column].astype(int)
    if not set(events_df[prediction_column].unique()).issubset({0, 1}):
        raise ValueError(
            f"{prediction_column} должен содержать только 0 и 1."
        )

    events_df = events_df.sort_values(
        ["sensor_id", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)
    previous_scenario = events_df.groupby("sensor_id")["scenario"].shift()
    starts_new_event = events_df["scenario"].ne(previous_scenario)
    events_df["_event_id"] = starts_new_event.groupby(
        events_df["sensor_id"]
    ).cumsum()

    event_reports = []
    anomaly_rows = events_df[events_df["scenario"] != "normal"]
    for (_sensor_id, _event_id), event in anomaly_rows.groupby(
        ["sensor_id", "_event_id"],
        sort=False,
    ):
        event_start = event["timestamp"].iloc[0]
        event_end = event["timestamp"].iloc[-1]
        alarms = event[event[prediction_column] == 1]
        detected = not alarms.empty
        first_alarm = alarms["timestamp"].iloc[0] if detected else None
        delay_seconds = (
            round((first_alarm - event_start).total_seconds(), 3)
            if detected
            else None
        )
        event_reports.append(
            {
                "sensor_id": str(event["sensor_id"].iloc[0]),
                "scenario": str(event["scenario"].iloc[0]),
                "event_start": event_start.isoformat(),
                "event_end": event_end.isoformat(),
                "detected": detected,
                "first_alarm": (
                    first_alarm.isoformat() if detected else None
                ),
                "delay_seconds": delay_seconds,
            }
        )

    delays = [
        event["delay_seconds"]
        for event in event_reports
        if event["delay_seconds"] is not None
    ]
    detected_events = len(delays)
    return {
        "events_total": len(event_reports),
        "events_detected": detected_events,
        "events_missed": len(event_reports) - detected_events,
        "mean_delay_seconds": (
            round(float(np.mean(delays)), 3) if delays else None
        ),
        "median_delay_seconds": (
            round(float(np.median(delays)), 3) if delays else None
        ),
        "max_delay_seconds": max(delays) if delays else None,
        "events": event_reports,
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
    layers = {}
    for layer, column in LAYER_COLUMNS.items():
        layer_report = binary_classification_metrics(
            truth,
            evaluation[column].astype(int),
        )
        layer_report["detection_delay"] = detection_delay_metrics(
            evaluation,
            column,
        )
        layers[layer] = layer_report

    return {
        "positive_class": "scenario != normal",
        "split": split_info,
        "layers": layers,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Отчёт качества правил, ML и объединённой системы."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument(
        "--test-start",
        default=TEST_START,
        help=(
            "Начало evaluation в формате YYYY-MM-DD HH:MM:SS. "
            "Должно совпадать с границей обучения."
        ),
    )
    parser.add_argument(
        "--output",
        help="Необязательный путь для сохранения JSON-отчёта.",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    df = pd.read_csv(args.input)
    report = evaluate_detection_layers(
        df,
        model_dir=args.model_dir,
        test_start=args.test_start,
    )
    report_json = json.dumps(report, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(report_json + "\n")

    print(report_json)


if __name__ == "__main__":
    main()
