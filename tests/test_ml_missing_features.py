"""Регрессии безопасной политики пропусков в ML-признаках."""

import hashlib

import numpy as np
import pandas as pd

from anomaly_detection import detect_anomalies
from model_schema import (
    ML_STATUS_APPLIED,
    ML_STATUS_NOT_APPLIED_RULES_ONLY,
    ML_STATUS_SKIPPED_MISSING_TEMPERATURE,
    prepare_ml_features,
)
from preprocessing import preprocess_data
import train_model


def _save_trained_bundle(preprocessed_synth, model_dir):
    scaler, model, info = train_model.train(preprocessed_synth)
    train_model.save_model(scaler, model, info, model_dir=str(model_dir))


def _artifact_hashes(model_dir):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in model_dir.iterdir()
        if path.is_file()
    }


def test_fully_empty_channel_skips_ml_and_keeps_signal_loss(
    preprocessed_synth,
    tmp_path,
):
    _save_trained_bundle(preprocessed_synth, tmp_path)
    hashes_before = _artifact_hashes(tmp_path)
    timestamps = pd.date_range("2026-06-07", periods=3, freq="1min")
    raw = pd.DataFrame(
        {
            "timestamp": [*timestamps, *timestamps],
            "sensor_id": ["T-VALID"] * 3 + ["T-EMPTY"] * 3,
            "temperature": [70.0, 70.5, 71.0, np.nan, np.nan, np.nan],
        }
    )

    results, alarm_log = detect_anomalies(
        preprocess_data(raw),
        model_dir=str(tmp_path),
    )
    empty_rows = results[results["sensor_id"] == "T-EMPTY"]
    valid_rows = results[results["sensor_id"] == "T-VALID"]

    assert empty_rows["temperature_filled"].isna().all()
    assert set(empty_rows["ml_inference_status"]) == {
        ML_STATUS_SKIPPED_MISSING_TEMPERATURE
    }
    assert empty_rows["iforest_prediction"].isna().all()
    assert (empty_rows["iforest_anomaly"] == 0).all()
    assert empty_rows["anomaly_score_norm"].isna().all()
    assert (empty_rows["rule_anomaly"] == 1).all()
    assert (empty_rows["primary_reason"] == "signal_loss").all()
    assert (empty_rows["final_anomaly"] == 1).all()
    assert set(valid_rows["ml_inference_status"]) == {ML_STATUS_APPLIED}
    assert valid_rows["iforest_prediction"].notna().all()
    assert set(
        alarm_log.loc[alarm_log["Датчик"] == "T-EMPTY", "Статус_ML"]
    ) == {ML_STATUS_SKIPPED_MISSING_TEMPERATURE}
    assert _artifact_hashes(tmp_path) == hashes_before


def test_single_missing_between_valid_points_and_first_point_are_explicit(
    preprocessed_synth,
    tmp_path,
):
    _save_trained_bundle(preprocessed_synth, tmp_path)
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-06-07",
                periods=3,
                freq="1min",
            ),
            "sensor_id": ["T-01"] * 3,
            "temperature": [70.0, np.nan, 71.0],
        }
    )

    results, _ = detect_anomalies(
        preprocess_data(raw),
        model_dir=str(tmp_path),
    )
    first, missing, last = [results.iloc[index] for index in range(3)]

    assert first["temp_diff"] == 0
    assert first["ml_inference_status"] == ML_STATUS_APPLIED
    assert pd.notna(first["iforest_prediction"])
    assert missing["temperature_filled"] == 70.0
    assert missing["ml_inference_status"] == (
        ML_STATUS_SKIPPED_MISSING_TEMPERATURE
    )
    assert pd.isna(missing["iforest_prediction"])
    assert pd.isna(missing["anomaly_score_norm"])
    assert missing["primary_reason"] == "signal_loss"
    assert last["ml_inference_status"] == ML_STATUS_APPLIED


def test_rules_only_records_that_ml_was_not_applied(tmp_path):
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-07", periods=2, freq="1min"),
            "sensor_id": ["T-01", "T-01"],
            "temperature": [70.0, np.nan],
        }
    )

    results, alarm_log = detect_anomalies(
        preprocess_data(raw),
        model_dir=str(tmp_path),
        use_ml=False,
    )

    assert list(tmp_path.iterdir()) == []
    assert set(results["ml_inference_status"]) == {
        ML_STATUS_NOT_APPLIED_RULES_ONLY
    }
    assert set(alarm_log["Статус_ML"]) == {
        ML_STATUS_NOT_APPLIED_RULES_ONLY
    }


def test_train_and_shared_feature_preparation_match(preprocessed_synth):
    expected_df, expected_X, expected_mask = prepare_ml_features(
        preprocessed_synth
    )
    train_df, train_X = train_model.prepare_features(preprocessed_synth)

    pd.testing.assert_frame_equal(train_df, expected_df)
    pd.testing.assert_frame_equal(train_X, expected_X)
    assert expected_X.index.equals(expected_df.index[expected_mask])
    assert not expected_X.isna().any().any()