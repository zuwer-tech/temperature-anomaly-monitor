import hashlib

import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from anomaly_detection import detect_anomalies
from evaluation import (
    LAYER_COLUMNS,
    binary_classification_metrics,
    detection_delay_metrics,
    evaluate_detection_layers,
)
import evaluation
import train_model


def _artifact_hashes(model_dir):
    return {
        name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
        for name in ("scaler.joblib", "iforest.joblib", "model_meta.json")
    }


def test_binary_metrics_include_all_confusion_matrix_cells():
    report = binary_classification_metrics(
        y_true=[0, 0, 0, 1, 1],
        y_pred=[0, 1, 0, 0, 1],
    )

    assert report == {
        "confusion_matrix": {
            "tn": 2,
            "fp": 1,
            "fn": 1,
            "tp": 1,
        },
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_detection_delay_uses_event_start_and_marks_missed_events():
    evaluation = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:03:00",
                "2026-01-01 00:00:00",
                "2026-01-01 00:05:00",
                "2026-01-01 00:01:00",
                "2026-01-01 00:04:00",
                "2026-01-01 00:00:00",
            ],
            "sensor_id": ["T-01", "T-01", "T-01", "T-01", "T-01", "T-02"],
            "scenario": [
                "sensor_drift",
                "normal",
                "sensor_drift",
                "sensor_drift",
                "normal",
                "sharp_jump",
            ],
            "alarm": [1, 0, 0, 0, 0, 1],
        }
    )

    report = detection_delay_metrics(evaluation, "alarm")

    assert report["events_total"] == 3
    assert report["events_detected"] == 2
    assert report["events_missed"] == 1
    assert report["mean_delay_seconds"] == 60.0
    assert report["median_delay_seconds"] == 60.0
    assert report["max_delay_seconds"] == 120.0
    assert report["events"] == [
        {
            "sensor_id": "T-01",
            "scenario": "sensor_drift",
            "event_start": "2026-01-01T00:01:00",
            "event_end": "2026-01-01T00:03:00",
            "detected": True,
            "first_alarm": "2026-01-01T00:03:00",
            "delay_seconds": 120.0,
        },
        {
            "sensor_id": "T-01",
            "scenario": "sensor_drift",
            "event_start": "2026-01-01T00:05:00",
            "event_end": "2026-01-01T00:05:00",
            "detected": False,
            "first_alarm": None,
            "delay_seconds": None,
        },
        {
            "sensor_id": "T-02",
            "scenario": "sharp_jump",
            "event_start": "2026-01-01T00:00:00",
            "event_end": "2026-01-01T00:00:00",
            "detected": True,
            "first_alarm": "2026-01-01T00:00:00",
            "delay_seconds": 0.0,
        },
    ]


def test_unified_report_matches_each_pipeline_layer(
    preprocessed_synth,
    tmp_path,
):
    scaler, model, info = train_model.train(preprocessed_synth)
    train_model.save_model(scaler, model, info, model_dir=str(tmp_path))

    report = evaluate_detection_layers(
        preprocessed_synth,
        model_dir=str(tmp_path),
    )
    detected, _alarm_log = detect_anomalies(
        preprocessed_synth,
        model_dir=str(tmp_path),
    )
    _train, evaluation, _x_train, _x_evaluation, split_info = (
        train_model.split_train_evaluation(detected)
    )
    truth = (evaluation["scenario"] != "normal").astype(int)

    assert report["positive_class"] == "scenario != normal"
    assert report["split"] == split_info
    assert set(report["layers"]) == set(LAYER_COLUMNS)

    for layer, column in LAYER_COLUMNS.items():
        expected_metrics = binary_classification_metrics(
            truth,
            evaluation[column].astype(int),
        )
        expected_delay = detection_delay_metrics(evaluation, column)
        assert report["layers"][layer] == {
            **expected_metrics,
            "detection_delay": expected_delay,
        }
        assert sum(
            report["layers"][layer]["confusion_matrix"].values()
        ) == split_info["evaluation_rows"]
        assert (
            expected_delay["events_detected"]
            + expected_delay["events_missed"]
            == expected_delay["events_total"]
        )

    assert split_info["evaluation_rows"] < len(preprocessed_synth)


def test_report_is_reproducible_without_training_or_artifact_changes(
    preprocessed_synth,
    tmp_path,
    monkeypatch,
):
    scaler, model, info = train_model.train(preprocessed_synth)
    train_model.save_model(scaler, model, info, model_dir=str(tmp_path))
    hashes_before = _artifact_hashes(tmp_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Оценка не должна обучать модель.")

    monkeypatch.setattr(StandardScaler, "fit", fail_if_called)
    monkeypatch.setattr(StandardScaler, "fit_transform", fail_if_called)
    monkeypatch.setattr(IsolationForest, "fit", fail_if_called)

    first = evaluate_detection_layers(
        preprocessed_synth,
        model_dir=str(tmp_path),
    )
    second = evaluate_detection_layers(
        preprocessed_synth,
        model_dir=str(tmp_path),
    )

    assert first == second
    assert _artifact_hashes(tmp_path) == hashes_before


def test_report_requires_reference_scenario(preprocessed_synth, tmp_path):
    scaler, model, info = train_model.train(preprocessed_synth)
    train_model.save_model(scaler, model, info, model_dir=str(tmp_path))

    with pytest.raises(ValueError, match="scenario"):
        evaluate_detection_layers(
            preprocessed_synth.drop(columns=["scenario"]),
            model_dir=str(tmp_path),
        )


def test_report_uses_configured_time_split(
    preprocessed_full_synth,
    tmp_path,
):
    custom_start = "2026-06-06 13:00:00"
    scaler, model, info = train_model.train(
        preprocessed_full_synth,
        test_start=custom_start,
    )
    train_model.save_model(
        scaler,
        model,
        info,
        model_dir=str(tmp_path),
    )

    report = evaluate_detection_layers(
        preprocessed_full_synth,
        model_dir=str(tmp_path),
        test_start=custom_start,
    )

    assert report["split"]["test_start"] == "2026-06-06T13:00:00"
    assert report["split"]["train_rows"] == 360
    assert report["split"]["evaluation_rows"] == 2520


def test_evaluation_cli_accepts_time_split():
    args = evaluation.parse_args(
        ["--test-start", "2026-06-06 13:00:00"]
    )

    assert args.test_start == "2026-06-06 13:00:00"