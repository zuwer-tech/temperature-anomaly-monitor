"""Тесты обучения и персистентности модели (PR1).

Проверяем главное:
- модель обучается только на штатном режиме (scenario=='normal') — нет data leakage;
- сохранение/загрузка детерминированы (та же random_state → те же предсказания);
- detect_anomalies подхватывает сохранённую модель вместо fit на лету;
- качество модели не хуже базового порога.
"""
import json
import hashlib
import warnings

import numpy as np
import pandas as pd
import pytest
from joblib import dump
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import anomaly_detection
import model_schema
from preprocessing import preprocess_data
from anomaly_detection import (
    ANALYSIS_MODE_RULES_ML,
    ANALYSIS_MODE_RULES_ONLY,
    FEATURE_COLUMNS,
    ModelCompatibilityError,
    ModelNotTrainedError,
    detect_anomalies,
)
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


class StubScaler:
    n_features_in_ = len(FEATURE_COLUMNS)
    transform_calls = 0

    def transform(self, X):
        type(self).transform_calls += 1
        return np.asarray(X)


class StubModel:
    n_features_in_ = len(FEATURE_COLUMNS)
    predict_calls = 0
    decision_function_calls = 0

    def predict(self, X):
        type(self).predict_calls += 1
        return np.ones(len(X), dtype=int)

    def decision_function(self, X):
        type(self).decision_function_calls += 1
        return np.zeros(len(X), dtype=float)


class ModelWithoutPredict:
    n_features_in_ = len(FEATURE_COLUMNS)

    def decision_function(self, X):
        return np.zeros(len(X), dtype=float)


class ModelWithoutDecisionFunction:
    n_features_in_ = len(FEATURE_COLUMNS)

    def predict(self, X):
        return np.ones(len(X), dtype=int)


def _valid_metadata():
    return {
        "metadata_version": model_schema.METADATA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "contamination": 0.04,
        "random_state": 42,
        "n_estimators": 200,
        "train_rows": 10,
        "trained_on_normal": True,
        "split_strategy": "time",
        "test_start": "2026-01-01T00:00:00",
        "evaluation_rows": 5,
        "evaluation_normal_rows": 3,
        "evaluation_anomaly_rows": 2,
        "score_calibration": {
            "method": model_schema.SCORE_CALIBRATION_METHOD,
            "score_min": -1.0,
            "score_max": 1.0,
            "medium_threshold": model_schema.RISK_MEDIUM_THRESHOLD,
            "high_threshold": model_schema.RISK_HIGH_THRESHOLD,
        },
    }


def test_model_schema_is_shared_and_has_expected_order():
    expected = (
        "temperature_filled",
        "rolling_mean",
        "rolling_std",
        "temp_diff",
        "abs_temp_diff",
        "abs_z_score",
        "is_missing",
        "is_stuck",
        "abs_diff_from_group_mean",
        "rolling_temp_diff_mean_20",
    )

    assert isinstance(model_schema.FEATURE_COLUMNS, tuple)
    assert model_schema.FEATURE_COLUMNS == expected
    assert len(model_schema.FEATURE_COLUMNS) == 10
    assert train_model.FEATURE_COLUMNS is model_schema.FEATURE_COLUMNS
    assert anomaly_detection.FEATURE_COLUMNS is model_schema.FEATURE_COLUMNS
    assert train_model.METADATA_VERSION is model_schema.METADATA_VERSION
    assert anomaly_detection.METADATA_VERSION is model_schema.METADATA_VERSION


def _write_bundle(tmp_path, metadata=None, scaler=None, model=None):
    dump(StubScaler() if scaler is None else scaler, tmp_path / "scaler.joblib")
    dump(StubModel() if model is None else model, tmp_path / "iforest.joblib")
    with open(tmp_path / "model_meta.json", "w", encoding="utf-8") as fh:
        json.dump(_valid_metadata() if metadata is None else metadata, fh)


def test_missing_scenario_stops_training_without_fallback_warning(
    preprocessed_synth,
):
    unlabelled = preprocessed_synth.drop(columns=["scenario"])

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(
            ValueError,
            match="подтверждённых normal-данных.*scenario",
        ):
            train_model.train(unlabelled)

    assert caught_warnings == []


def test_dataset_without_normal_rows_stops_training(preprocessed_synth):
    anomalies_only = preprocessed_synth.copy()
    anomalies_only["scenario"] = "sharp_jump"

    with pytest.raises(
        ValueError,
        match="нет строк scenario == 'normal'",
    ):
        train_model.train(anomalies_only)


def test_failed_main_creates_no_model_artifacts(
    preprocessed_synth,
    tmp_path,
    monkeypatch,
):
    invalid_input = tmp_path / "without_scenario.csv"
    preprocessed_synth.drop(columns=["scenario"]).to_csv(
        invalid_input,
        index=False,
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="подтверждённых normal-данных"):
        train_model.main(input_file=str(invalid_input))

    model_dir = tmp_path / "models"
    assert not model_dir.exists() or list(model_dir.iterdir()) == []

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


def test_custom_time_split_changes_sizes_predictably(
    preprocessed_full_synth,
):
    custom_start = "2026-06-06 13:00:00"
    train_df, evaluation_df, _x_train, _x_evaluation, info = (
        train_model.split_train_evaluation(
            preprocessed_full_synth,
            test_start=custom_start,
        )
    )

    assert info["test_start"] == "2026-06-06T13:00:00"
    assert info["train_rows"] == 360
    assert info["evaluation_rows"] == 2520
    assert len(train_df) == 360
    assert len(evaluation_df) == 2520
    assert (train_df["timestamp"] < pd.Timestamp(custom_start)).all()
    assert (evaluation_df["timestamp"] >= pd.Timestamp(custom_start)).all()


@pytest.mark.parametrize(
    "test_start",
    [
        "not-a-date",
        "2026-06-06",
        "2026-06-06T14:00:00",
        "2026-6-6 14:00:00",
        None,
    ],
)
def test_invalid_time_split_boundary_is_rejected(
    preprocessed_full_synth,
    test_start,
):
    with pytest.raises(
        ValueError,
        match="test_start.*YYYY-MM-DD HH:MM:SS",
    ):
        train_model.split_train_evaluation(
            preprocessed_full_synth,
            test_start=test_start,
        )


def test_time_split_requires_rows_on_both_sides(
    preprocessed_full_synth,
):
    timestamps = pd.to_datetime(preprocessed_full_synth["timestamp"])
    earliest = timestamps.min().strftime(train_model.TEST_START_FORMAT)
    after_latest = (
        timestamps.max() + pd.Timedelta(seconds=1)
    ).strftime(train_model.TEST_START_FORMAT)

    with pytest.raises(ValueError, match="В train нет строк"):
        train_model.split_train_evaluation(
            preprocessed_full_synth,
            test_start=earliest,
        )
    with pytest.raises(ValueError, match="В evaluation нет строк"):
        train_model.split_train_evaluation(
            preprocessed_full_synth,
            test_start=after_latest,
        )


def test_metadata_saves_actual_time_split(
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

    metadata = json.loads(
        (tmp_path / "model_meta.json").read_text(encoding="utf-8")
    )
    assert metadata["test_start"] == "2026-06-06T13:00:00"
    assert metadata["train_rows"] == 360
    assert metadata["evaluation_rows"] == 2520


def test_train_cli_accepts_time_split():
    args = train_model.parse_args(
        ["--test-start", "2026-06-06 13:00:00"]
    )

    assert args.test_start == "2026-06-06 13:00:00"


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
    expected = pd.Series(0, index=results.index, dtype=int)
    expected.loc[X.index] = (
        model.predict(scaler.transform(X)) == -1
    ).astype(int)
    np.testing.assert_array_equal(
        results["iforest_anomaly"].to_numpy(), expected.to_numpy()
    )
    assert {
        "iforest_prediction",
        "iforest_anomaly",
        "iforest_score_raw",
        "anomaly_score",
        "anomaly_score_norm",
        "analysis_mode",
        "final_anomaly",
    }.issubset(results.columns)


def test_rules_and_ml_mode_is_recorded_in_results_and_alarms(
    preprocessed_synth, tmp_path
):
    _write_bundle(tmp_path)

    results, alarm_log = detect_anomalies(
        preprocessed_synth, model_dir=str(tmp_path)
    )

    assert set(results["analysis_mode"].unique()) == {ANALYSIS_MODE_RULES_ML}
    assert "Режим_анализа" in alarm_log.columns
    assert set(alarm_log["Режим_анализа"].unique()) <= {ANALYSIS_MODE_RULES_ML}


def test_rules_only_mode_skips_model_and_marks_scores_unavailable(
    preprocessed_synth, tmp_path
):
    results, alarm_log = detect_anomalies(
        preprocessed_synth, model_dir=str(tmp_path), use_ml=False
    )

    assert list(tmp_path.iterdir()) == []
    assert set(results["analysis_mode"].unique()) == {ANALYSIS_MODE_RULES_ONLY}
    assert (results["iforest_anomaly"] == 0).all()
    assert results["iforest_prediction"].isna().all()
    assert results["anomaly_score_norm"].isna().all()
    assert set(results["ml_inference_status"].unique()) == {
        model_schema.ML_STATUS_NOT_APPLIED_RULES_ONLY
    }
    pd.testing.assert_series_equal(
        results["final_anomaly"],
        results["rule_anomaly"],
        check_names=False,
    )
    assert "Режим_анализа" in alarm_log.columns
    assert set(alarm_log["Режим_анализа"].unique()) <= {ANALYSIS_MODE_RULES_ONLY}


def test_missing_model_artifacts_raise_without_creating_files(
    preprocessed_synth, tmp_path
):
    with pytest.raises(ModelNotTrainedError) as exc_info:
        detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))

    message = str(exc_info.value)
    expected_names = ("scaler.joblib", "iforest.joblib", "model_meta.json")
    assert all(name in message for name in expected_names)
    assert [message.index(name) for name in expected_names] == sorted(
        message.index(name) for name in expected_names
    )
    assert "python preprocessing.py" in message
    assert "python train_model.py" in message
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "missing_name",
    [
        "scaler.joblib",
        "iforest.joblib",
        "model_meta.json",
    ],
)
def test_incomplete_model_artifacts_report_missing_file(
    preprocessed_synth, tmp_path, missing_name
):
    artifact_names = ("scaler.joblib", "iforest.joblib", "model_meta.json")
    for name in artifact_names:
        if name != missing_name:
            (tmp_path / name).write_bytes(b"existing artifact")

    with pytest.raises(ModelNotTrainedError) as exc_info:
        detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))

    assert missing_name in str(exc_info.value)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        name for name in artifact_names if name != missing_name
    )
    assert all(path.read_bytes() == b"existing artifact" for path in tmp_path.iterdir())


def test_corrupted_metadata_json_raises_compatibility_error(
    preprocessed_synth, tmp_path
):
    _write_bundle(tmp_path)
    (tmp_path / "model_meta.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(ModelCompatibilityError, match="model_meta.json"):
        detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))


def test_metadata_top_level_must_be_object(preprocessed_synth, tmp_path):
    _write_bundle(tmp_path, metadata=[])

    with pytest.raises(ModelCompatibilityError, match="JSON-объект"):
        detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))


def test_metadata_requires_feature_columns(preprocessed_synth, tmp_path):
    metadata = _valid_metadata()
    del metadata["feature_columns"]
    _write_bundle(tmp_path, metadata=metadata)

    with pytest.raises(ModelCompatibilityError, match="feature_columns"):
        detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))


@pytest.mark.parametrize(
    "feature_columns",
    [
        "not-a-list",
        list(FEATURE_COLUMNS[:-1]),
        [*FEATURE_COLUMNS, "extra_feature"],
        list(FEATURE_COLUMNS[::-1]),
    ],
    ids=("not-list", "missing-feature", "extra-feature", "changed-order"),
)
def test_metadata_feature_columns_must_match_exactly(
    preprocessed_synth, tmp_path, feature_columns
):
    metadata = _valid_metadata()
    metadata["feature_columns"] = feature_columns
    _write_bundle(tmp_path, metadata=metadata)

    with pytest.raises(ModelCompatibilityError) as exc_info:
        detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))

    message = str(exc_info.value)
    assert "Набор признаков модели несовместим" in message
    assert "Ожидался список" in message
    assert "Фактический список" in message
    assert "python preprocessing.py" in message
    assert "python train_model.py" in message


def test_score_normalization_is_independent_of_current_batch():
    """Одна и та же точка получает один score при разных соседних строках."""
    calibration = _valid_metadata()["score_calibration"]
    target_score = 0.25

    first_batch = anomaly_detection._normalize_anomaly_score(
        np.array([target_score, -100.0, 100.0]),
        calibration,
    )
    second_batch = anomaly_detection._normalize_anomaly_score(
        np.array([target_score, 0.20, 0.30]),
        calibration,
    )

    assert np.isclose(first_batch[0], second_batch[0])
    assert np.isclose(first_batch[0], 0.625)
    assert first_batch[1] == 0.0
    assert first_batch[2] == 1.0


def test_train_calibration_uses_only_training_baseline(preprocessed_synth):
    scaler, model, info = train_model.train(preprocessed_synth)
    _train_df, _evaluation_df, X_train, _X_evaluation, _ = (
        train_model.split_train_evaluation(preprocessed_synth)
    )
    train_scores = -model.decision_function(scaler.transform(X_train))
    calibration = info["score_calibration"]

    assert calibration["method"] == model_schema.SCORE_CALIBRATION_METHOD
    assert np.isclose(calibration["score_min"], train_scores.min())
    assert np.isclose(calibration["score_max"], train_scores.max())
    assert calibration["medium_threshold"] == model_schema.RISK_MEDIUM_THRESHOLD
    assert calibration["high_threshold"] == model_schema.RISK_HIGH_THRESHOLD


@pytest.mark.parametrize(
    "calibration",
    [
        None,
        {
            "method": model_schema.SCORE_CALIBRATION_METHOD,
            "score_min": 1.0,
            "score_max": 1.0,
            "medium_threshold": 0.60,
            "high_threshold": 0.85,
        },
        {
            "method": model_schema.SCORE_CALIBRATION_METHOD,
            "score_min": -1.0,
            "score_max": 1.0,
            "medium_threshold": 0.90,
            "high_threshold": 0.85,
        },
    ],
    ids=("not-object", "equal-bounds", "reversed-thresholds"),
)
def test_invalid_score_calibration_is_rejected(
    preprocessed_synth,
    tmp_path,
    calibration,
):
    metadata = _valid_metadata()
    metadata["score_calibration"] = calibration
    _write_bundle(tmp_path, metadata=metadata)

    with pytest.raises(ModelCompatibilityError, match="score_calibration|калибровки|Пороги"):
        detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))


def test_supported_metadata_version_is_accepted(preprocessed_synth, tmp_path):
    _write_bundle(tmp_path)

    detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))


def test_unsupported_metadata_version_raises(preprocessed_synth, tmp_path):
    metadata = _valid_metadata()
    metadata["metadata_version"] = model_schema.METADATA_VERSION + 1
    _write_bundle(tmp_path, metadata=metadata)

    with pytest.raises(ModelCompatibilityError, match="не поддерживается"):
        detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))


def test_corrupted_scaler_raises_before_predict(
    preprocessed_synth, tmp_path
):
    _write_bundle(tmp_path)
    (tmp_path / "scaler.joblib").write_bytes(b"not a joblib artifact")
    StubModel.predict_calls = 0

    with pytest.raises(ModelCompatibilityError, match="scaler.joblib"):
        detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))

    assert StubModel.predict_calls == 0


def test_corrupted_model_raises_compatibility_error(preprocessed_synth, tmp_path):
    _write_bundle(tmp_path)
    (tmp_path / "iforest.joblib").write_bytes(b"not a joblib artifact")

    with pytest.raises(ModelCompatibilityError, match="iforest.joblib"):
        detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))


def test_scaler_requires_transform(preprocessed_synth, tmp_path):
    _write_bundle(tmp_path, scaler=object())

    with pytest.raises(ModelCompatibilityError, match="transform"):
        detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))


def test_model_requires_predict(preprocessed_synth, tmp_path):
    _write_bundle(tmp_path, model=ModelWithoutPredict())

    with pytest.raises(ModelCompatibilityError, match="predict"):
        detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))


def test_model_requires_decision_function(preprocessed_synth, tmp_path):
    _write_bundle(tmp_path, model=ModelWithoutDecisionFunction())

    with pytest.raises(ModelCompatibilityError, match="decision_function"):
        detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))


def test_scaler_feature_count_must_match(preprocessed_synth, tmp_path):
    scaler = StubScaler()
    scaler.n_features_in_ = len(FEATURE_COLUMNS) - 1
    _write_bundle(tmp_path, scaler=scaler)

    with pytest.raises(ModelCompatibilityError, match="scaler.joblib"):
        detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))


def test_model_feature_count_must_match(preprocessed_synth, tmp_path):
    model = StubModel()
    model.n_features_in_ = len(FEATURE_COLUMNS) + 1
    _write_bundle(tmp_path, model=model)

    with pytest.raises(ModelCompatibilityError, match="iforest.joblib"):
        detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))


def test_valid_bundle_calls_inference_methods_and_stays_unchanged(
    preprocessed_synth, tmp_path
):
    _write_bundle(tmp_path)
    artifact_paths = [
        tmp_path / "scaler.joblib",
        tmp_path / "iforest.joblib",
        tmp_path / "model_meta.json",
    ]
    hashes_before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in artifact_paths
    }
    StubScaler.transform_calls = 0
    StubModel.predict_calls = 0
    StubModel.decision_function_calls = 0

    detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))

    assert StubScaler.transform_calls == 1
    assert StubModel.predict_calls == 1
    assert StubModel.decision_function_calls == 1
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in artifact_paths
    } == hashes_before


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

    expected_pred = pd.Series(0, index=evaluation_df.index, dtype=int)
    expected_pred.loc[X_evaluation.index] = (
        model.predict(scaler.transform(X_evaluation)) == -1
    ).astype(int)
    expected_gt = (evaluation_df["scenario"] != "normal").astype(int)
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
    assert meta["metadata_version"] == model_schema.METADATA_VERSION
    assert isinstance(meta["feature_columns"], list)
    assert meta["feature_columns"] == list(model_schema.FEATURE_COLUMNS)
    assert meta["trained_on_normal"] is True
    expected_fields = {
        "metadata_version",
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
    }
    assert expected_fields.issubset(meta)
    for key in expected_fields & info.keys():
        assert meta[key] == info[key]
    assert meta["score_calibration"] == info["score_calibration"]
