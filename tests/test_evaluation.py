import hashlib

import pytest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from anomaly_detection import detect_anomalies
from evaluation import (
    LAYER_COLUMNS,
    binary_classification_metrics,
    evaluate_detection_layers,
)
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
        assert report["layers"][layer] == binary_classification_metrics(
            truth,
            evaluation[column].astype(int),
        )
        assert sum(
            report["layers"][layer]["confusion_matrix"].values()
        ) == split_info["evaluation_rows"]

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