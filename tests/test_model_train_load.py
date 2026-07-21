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
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from preprocessing import preprocess_data
from anomaly_detection import ModelNotTrainedError, detect_anomalies
import train_model


ANOMALY_SCENARIOS = {
    "sharp_jump",
    "slow_overheating",
    "sensor_drift",
    "stuck_sensor",
    "high_noise",
    "signal_loss",
    "correlated_growth",
}


def test_train_uses_normal_only(preprocessed_synth):
    """Модель должна обучаться на 720 ранних normal-строках."""
    _scaler, _model, info = train_model.train(preprocessed_synth)
    assert info["trained_on_normal"] is True
    assert info["train_rows"] == 720

    train_df, _evaluation_df, _X_train, _X_evaluation, _ = (
        train_model.split_train_evaluation(preprocessed_synth)
    )
    assert (train_df["scenario"] == "normal").all()
    assert (train_df["timestamp"] < pd.Timestamp(train_model.TEST_START)).all()


def test_current_benchmark_split_is_complete_and_disjoint(
    preprocessed_full_synth,
):
    train_df, evaluation_df, _X_train, _X_evaluation, info = (
        train_model.split_train_evaluation(preprocessed_full_synth)
    )

    assert info == {
        "train_rows": 720,
        "trained_on_normal": True,
        "split_strategy": "time",
        "test_start": "2026-06-06T14:00:00",
        "evaluation_rows": 2160,
        "evaluation_normal_rows": 1580,
        "evaluation_anomaly_rows": 580,
    }
    assert (train_df["scenario"] == "normal").all()
    assert evaluation_df["sensor_id"].nunique() == 6
    assert set(evaluation_df.loc[
        evaluation_df["scenario"] != "normal", "scenario"
    ]) == ANOMALY_SCENARIOS

    train_keys = set(zip(train_df["sensor_id"], train_df["timestamp"]))
    evaluation_keys = set(zip(
        evaluation_df["sensor_id"], evaluation_df["timestamp"]
    ))
    assert train_keys.isdisjoint(evaluation_keys)


def test_time_split_is_deterministic(preprocessed_full_synth):
    first = train_model.split_train_evaluation(preprocessed_full_synth)
    second = train_model.split_train_evaluation(preprocessed_full_synth)

    for first_frame, second_frame in zip(first[:4], second[:4]):
        pd.testing.assert_frame_equal(first_frame, second_frame)
    assert first[4] == second[4]


def test_future_values_do_not_change_train_features(preprocessed_synth):
    prepared_before, X_before = train_model.prepare_features(preprocessed_synth)
    changed = preprocessed_synth.copy()
    future_mask = pd.to_datetime(changed["timestamp"]) >= pd.Timestamp(
        train_model.TEST_START
    )
    changed.loc[future_mask, "temp_diff"] += 1000
    changed.loc[future_mask, "temperature_filled"] += 1000
    changed.loc[future_mask, "rolling_mean"] += 1000

    prepared_after, X_after = train_model.prepare_features(changed)
    train_mask = prepared_before["timestamp"] < pd.Timestamp(
        train_model.TEST_START
    )
    pd.testing.assert_frame_equal(
        X_before.loc[train_mask],
        X_after.loc[train_mask],
    )
    pd.testing.assert_series_equal(
        prepared_before.loc[train_mask, "timestamp"],
        prepared_after.loc[train_mask, "timestamp"],
    )


def test_evaluation_keeps_train_rolling_history(preprocessed_synth):
    _train_df, evaluation_df, _X_train, X_evaluation, _info = (
        train_model.split_train_evaluation(preprocessed_synth)
    )
    first_evaluation_index = evaluation_df[
        evaluation_df["sensor_id"] == "T-01"
    ].index[0]

    _isolated_df, isolated_X = train_model.prepare_features(evaluation_df)
    full_history_value = X_evaluation.loc[
        first_evaluation_index, "rolling_temp_diff_mean_20"
    ]
    reset_history_value = isolated_X.loc[
        isolated_X["rolling_temp_diff_mean_20"].index[0],
        "rolling_temp_diff_mean_20",
    ]

    assert not np.isclose(full_history_value, reset_history_value)


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


def test_model_quality_floor(preprocessed_full_synth):
    """Независимые precision >= 0.66 и F1 >= 0.45 на полном benchmark."""
    scaler, model, _ = train_model.train(preprocessed_full_synth)
    report = train_model.evaluate(preprocessed_full_synth, scaler, model)
    assert report["precision"] >= 0.66, report
    assert report["f1"] >= 0.45, report


def test_evaluate_uses_only_evaluation_rows(preprocessed_synth):
    scaler, model, _ = train_model.train(preprocessed_synth)
    _train_df, evaluation_df, _X_train, X_evaluation, info = (
        train_model.split_train_evaluation(preprocessed_synth)
    )
    report = train_model.evaluate(preprocessed_synth, scaler, model)

    expected_pred = (
        model.predict(scaler.transform(X_evaluation)) == -1
    ).astype(int)
    expected_gt = (evaluation_df["scenario"] != "normal").astype(int).to_numpy()
    assert report["iforest_anomalies"] == int(expected_pred.sum())
    assert report["tp"] == int(((expected_pred == 1) & (expected_gt == 1)).sum())
    assert report["fp"] == int(((expected_pred == 1) & (expected_gt == 0)).sum())
    assert report["fn"] == int(((expected_pred == 0) & (expected_gt == 1)).sum())
    for key in (
        "train_rows",
        "evaluation_rows",
        "evaluation_normal_rows",
        "evaluation_anomaly_rows",
        "split_strategy",
        "test_start",
    ):
        assert report[key] == info[key]


def test_meta_json_written(preprocessed_synth, tmp_path):
    """model_meta.json содержит признаки и сведения о разбиении."""
    scaler, model, info = train_model.train(preprocessed_synth)
    train_model.save_model(scaler, model, info, model_dir=str(tmp_path))
    with open(tmp_path / "model_meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["feature_columns"] == train_model.FEATURE_COLUMNS
    assert meta["trained_on_normal"] is True
    for key in (
        "split_strategy",
        "test_start",
        "train_rows",
        "evaluation_rows",
        "evaluation_normal_rows",
        "evaluation_anomaly_rows",
    ):
        assert meta[key] == info[key]
