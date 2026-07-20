"""Тесты обучения и персистентности модели (PR1).

Проверяем главное:
- модель обучается только на штатном режиме (scenario=='normal') — нет data leakage;
- сохранение/загрузка детерминированы (та же random_state → те же предсказания);
- detect_anomalies подхватывает сохранённую модель вместо fit на лету;
- качество модели не хуже базового порога.
"""
import json
import hashlib

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from preprocessing import preprocess_data
from anomaly_detection import ModelNotTrainedError, detect_anomalies
import train_model


def test_train_uses_normal_only(preprocessed_synth):
    """Модель должна обучаться только на строках scenario=='normal'."""
    scaler, model, info = train_model.train(preprocessed_synth)
    assert info["trained_on_normal"] is True
    normal_rows = int((preprocessed_synth["scenario"] == "normal").sum())
    assert info["train_rows"] == normal_rows


def test_save_load_is_deterministic(preprocessed_synth, tmp_path):
    """Два обучения с одним random_state дают идентичные предсказания."""
    scaler_a, model_a, _ = train_model.train(preprocessed_synth)
    scaler_b, model_b, _ = train_model.train(preprocessed_synth)

    _, X = train_model.prepare_features(preprocessed_synth)
    pred_a = (model_a.predict(scaler_a.transform(X)) == -1).astype(int)
    pred_b = (model_b.predict(scaler_b.transform(X)) == -1).astype(int)
    np.testing.assert_array_equal(pred_a, pred_b)


def test_detect_anomalies_uses_saved_model(preprocessed_synth, tmp_path):
    """После train_model.save_model detect_anomalies загружает модель, а не fit-ит."""
    scaler, model, info = train_model.train(preprocessed_synth)
    train_model.save_model(scaler, model, info, model_dir=str(tmp_path))

    results, _ = detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))

    # Прямое предсказание сохранённой моделью
    _, X = train_model.prepare_features(preprocessed_synth)
    expected = (model.predict(scaler.transform(X)) == -1).astype(int)
    np.testing.assert_array_equal(
        results["iforest_anomaly"].to_numpy(), expected
    )
    assert {
        "iforest_prediction",
        "iforest_anomaly",
        "iforest_score_raw",
        "anomaly_score",
        "anomaly_score_norm",
        "final_anomaly",
    }.issubset(results.columns)


def test_missing_model_artifacts_raise_without_creating_files(
    preprocessed_synth, tmp_path
):
    with pytest.raises(ModelNotTrainedError) as exc_info:
        detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))

    message = str(exc_info.value)
    assert "scaler.joblib" in message
    assert "iforest.joblib" in message
    assert "python preprocessing.py" in message
    assert "python train_model.py" in message
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("existing_name", "missing_name"),
    [
        ("scaler.joblib", "iforest.joblib"),
        ("iforest.joblib", "scaler.joblib"),
    ],
)
def test_incomplete_model_artifacts_report_missing_file(
    preprocessed_synth, tmp_path, existing_name, missing_name
):
    existing_path = tmp_path / existing_name
    existing_path.write_bytes(b"existing artifact")

    with pytest.raises(ModelNotTrainedError) as exc_info:
        detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))

    assert missing_name in str(exc_info.value)
    assert existing_path.read_bytes() == b"existing artifact"
    assert sorted(path.name for path in tmp_path.iterdir()) == [existing_name]


def test_inference_never_calls_fit(preprocessed_synth, tmp_path, monkeypatch):
    scaler, model, info = train_model.train(preprocessed_synth)
    train_model.save_model(scaler, model, info, model_dir=str(tmp_path))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("fit methods must not run during inference")

    monkeypatch.setattr(StandardScaler, "fit", fail_if_called)
    monkeypatch.setattr(StandardScaler, "fit_transform", fail_if_called)
    monkeypatch.setattr(IsolationForest, "fit", fail_if_called)

    detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))


def test_inference_does_not_modify_model_artifacts(preprocessed_synth, tmp_path):
    scaler, model, info = train_model.train(preprocessed_synth)
    train_model.save_model(scaler, model, info, model_dir=str(tmp_path))
    artifact_paths = [tmp_path / "scaler.joblib", tmp_path / "iforest.joblib"]
    hashes_before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in artifact_paths
    }

    detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))

    hashes_after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in artifact_paths
    }
    assert hashes_after == hashes_before


def test_model_quality_floor(preprocessed_synth):
    """Обученная на normal модель: precision >= 0.7, F1 >= 0.45 (бейзлайн-порог)."""
    scaler, model, _ = train_model.train(preprocessed_synth)
    report = train_model.evaluate(preprocessed_synth, scaler, model)
    assert report["precision"] >= 0.70, report
    assert report["f1"] >= 0.45, report


def test_meta_json_written(preprocessed_synth, tmp_path):
    """model_meta.json содержит список признаков и флаг trained_on_normal."""
    scaler, model, info = train_model.train(preprocessed_synth)
    train_model.save_model(scaler, model, info, model_dir=str(tmp_path))
    with open(tmp_path / "model_meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["feature_columns"] == train_model.FEATURE_COLUMNS
    assert meta["trained_on_normal"] is True
