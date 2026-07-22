"""Тесты точности правил (PR2).

Проверяем главное — система находит реальные аномалии и не Raises ложных
тревог на штатном режиме. Используем размеченный синтетический набор: сценарий
'scenario' выступает ground truth, аномалия = scenario != 'normal'.
"""
import numpy as np
import pandas as pd

import anomaly_detection
import preprocessing
import rule_config
from preprocessing import preprocess_data
from anomaly_detection import detect_anomalies, RULE_PARAMS
import train_model


def _metrics(pred, gt):
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return tp, fp, fn, precision, recall, f1


def _full_pipeline(preprocessed_synth, tmp_path):
    """Обучает модель и возвращает позднюю evaluation-часть пайплайна."""
    scaler, model, info = train_model.train(preprocessed_synth)
    train_model.save_model(scaler, model, info, model_dir=str(tmp_path))
    results, _alarm = detect_anomalies(preprocessed_synth, model_dir=str(tmp_path))
    _train, evaluation, _X_train, _X_evaluation, _split_info = (
        train_model.split_train_evaluation(results)
    )
    return evaluation


def test_rule_params_present():
    """Все ключи порогов правил определены — единая точка настройки."""
    assert RULE_PARAMS is rule_config.RULE_PARAMS
    assert anomaly_detection.RULE_PARAMS is rule_config.RULE_PARAMS
    assert preprocessing.RULE_PARAMS is rule_config.RULE_PARAMS

    for key in (
        "sharp_jump_rate_c_per_min", "z_score", "group_deviation",
        "overheat_window", "overheat_slope",
    ):
        assert key in RULE_PARAMS


def test_known_scenarios_detected(preprocessed_full_synth, tmp_path):
    """Резкие и явные аномалии ловятся наверняка (recall = 1.0)."""
    results = _full_pipeline(preprocessed_full_synth, tmp_path)
    for must_hit in ("sharp_jump", "signal_loss"):
        sub = results[results["scenario"] == must_hit]
        assert sub["final_anomaly"].sum() == len(sub), (
            f"сценарий {must_hit} должен ловиться полностью"
        )


def test_subtle_scenarios_recall(preprocessed_full_synth, tmp_path):
    """Тонкие аномалии ловятся не хуже порога (правило дрейфа/перегрева)."""
    results = _full_pipeline(preprocessed_full_synth, tmp_path)
    floors = {
        "sensor_drift": 0.5,
        "slow_overheating": 0.5,
        "correlated_growth": 0.4,
        "stuck_sensor": 0.6,
        "high_noise": 0.7,
    }
    for scenario, floor in floors.items():
        sub = results[results["scenario"] == scenario]
        recall = sub["final_anomaly"].sum() / len(sub) if len(sub) else 0.0
        assert recall >= floor, f"{scenario}: recall={recall:.2f} < {floor}"


def test_false_positive_rate_on_normal(preprocessed_full_synth, tmp_path):
    """Независимая доля ложных тревог на штатном режиме < 12%."""
    results = _full_pipeline(preprocessed_full_synth, tmp_path)
    normal = results[results["scenario"] == "normal"]
    fp_rate = normal["final_anomaly"].sum() / len(normal)
    assert fp_rate < 0.12, f"FP rate на normal = {fp_rate:.3f}"


def test_overall_f1(preprocessed_full_synth, tmp_path):
    """Итоговый F1 (правила + ИИ) >= 0.60 — регрессионный порог точности."""
    results = _full_pipeline(preprocessed_full_synth, tmp_path)
    pred = results["final_anomaly"].astype(int).to_numpy()
    gt = (results["scenario"] != "normal").astype(int).to_numpy()
    *_, precision, recall, f1 = _metrics(pred, gt)
    assert f1 >= 0.60, f"F1={f1:.3f} precision={precision:.3f} recall={recall:.3f}"


def test_sharp_jump_rule_uses_temperature_rate_not_row_difference():
    raw = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:10",
                "2026-01-01 00:00:00",
                "2026-01-01 00:01:00",
                "2026-01-01 00:00:00",
                "2026-01-01 00:10:00",
            ],
            "sensor_id": [
                "T-FAST",
                "T-FAST",
                "T-MINUTE",
                "T-MINUTE",
                "T-SLOW",
                "T-SLOW",
            ],
            "temperature": [70.0, 71.0, 70.0, 76.0, 70.0, 71.0],
        }
    )

    prepared = preprocess_data(raw)
    results, _ = detect_anomalies(prepared, use_ml=False)
    last_rows = results.groupby("sensor_id").tail(1).set_index("sensor_id")

    assert np.isclose(last_rows.loc["T-FAST", "temp_rate_c_per_min"], 6.0)
    assert np.isclose(last_rows.loc["T-MINUTE", "temp_rate_c_per_min"], 6.0)
    assert np.isclose(last_rows.loc["T-SLOW", "temp_rate_c_per_min"], 0.1)
    assert last_rows.loc["T-FAST", "rule_event_type"] == "Резкий скачок температуры"
    assert last_rows.loc["T-MINUTE", "rule_event_type"] == "Резкий скачок температуры"
    assert last_rows.loc["T-SLOW", "rule_anomaly"] == 0


def test_preliminary_and_final_rules_use_same_config(monkeypatch):
    monkeypatch.setitem(RULE_PARAMS, "sharp_jump_rate_c_per_min", 10.0)
    raw = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:10",
            ],
            "sensor_id": ["T-01", "T-01"],
            "temperature": [70.0, 71.0],
        }
    )

    prepared = preprocess_data(raw)
    results, _ = detect_anomalies(prepared, use_ml=False)

    assert np.isclose(prepared["temp_rate_c_per_min"].iloc[1], 6.0)
    assert prepared["preliminary_warning"].iloc[1] == 0
    assert results["rule_anomaly"].iloc[1] == 0

def test_all_simultaneous_rule_reasons_are_retained():
    """Одна точка хранит все причины, а не только последнее правило."""
    row_count = 20
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=row_count,
                freq="1min",
            ),
            "sensor_id": ["T-ALL"] * row_count,
            "temperature": 70.0 + np.arange(row_count),
            "temperature_filled": 70.0 + np.arange(row_count),
            "is_missing": [0] * (row_count - 1) + [1],
            "temp_rate_c_per_min": [0.0] * (row_count - 1) + [6.0],
            "abs_z_score": [0.0] * (row_count - 1) + [4.0],
            "is_stuck": [0] * (row_count - 1) + [1],
            "abs_diff_from_group_mean": (
                [0.0] * (row_count - 1) + [9.0]
            ),
            "temp_diff": [0.0] * row_count,
            "rolling_mean": np.arange(row_count, dtype=float),
            "scenario": ["normal"] * (row_count - 1) + ["combined_fault"],
        }
    )

    results, alarm_log = detect_anomalies(frame, use_ml=False)
    target_rules = results.iloc[-1]["triggered_rules"]

    assert target_rules == [
        "signal_loss",
        "sharp_jump",
        "z_score",
        "stuck_sensor",
        "group_deviation",
        "sustained_overheat",
    ]
    assert results.iloc[0]["triggered_rules"] == []
    assert results.iloc[-1]["primary_reason"] == "signal_loss"
    assert results.iloc[-1]["rule_count"] == 6
    assert results.iloc[-1]["rule_event_type"] == "Устойчивый перегрев"
    assert (
        results["rule_anomaly"]
        == results["triggered_rules"].map(bool).astype(int)
    ).all()
    assert alarm_log.iloc[-1]["Сработавшие_правила"] == target_rules
    assert alarm_log.iloc[-1]["primary_reason"] == "signal_loss"
    assert alarm_log.iloc[-1]["rule_count"] == 6

def test_ml_only_alarm_has_primary_reason_and_zero_rule_count(
    preprocessed_synth,
    monkeypatch,
):
    """ML-only тревога объясняется моделью, а не выдуманным правилом."""
    def fake_inference(features, model_dir="models"):
        predictions = np.ones(len(features), dtype=int)
        predictions[0] = -1
        scores = np.zeros(len(features), dtype=float)
        scores[0] = -1.0
        return features, predictions, scores, True

    monkeypatch.setattr(
        anomaly_detection,
        "_load_or_fit_iforest",
        fake_inference,
    )

    results, alarm_log = detect_anomalies(preprocessed_synth)
    ai_row = results.iloc[0]

    assert ai_row["rule_anomaly"] == 0
    assert ai_row["iforest_anomaly"] == 1
    assert ai_row["triggered_rules"] == []
    assert ai_row["primary_reason"] == "iforest_anomaly"
    assert ai_row["rule_count"] == 0

    ai_alarm = alarm_log[
        alarm_log["primary_reason"] == "iforest_anomaly"
    ].iloc[0]
    assert ai_alarm["rule_count"] == 0
    assert ai_alarm["Сработавшие_правила"] == []

    normal_rows = results[results["final_anomaly"] == 0]
    assert normal_rows["primary_reason"].isna().all()
    assert (normal_rows["rule_count"] == 0).all()
