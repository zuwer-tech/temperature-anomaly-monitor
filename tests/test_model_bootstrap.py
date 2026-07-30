"""Проверки явного обучения демонстрационной ML-модели."""
import json
from pathlib import Path

import pandas as pd
import pytest

from anomaly_detection import ANALYSIS_MODE_RULES_ML, detect_anomalies
from model_bootstrap import MODEL_ARTIFACTS, train_demo_model


ROOT = Path(__file__).resolve().parents[1]
DEMO_DATASET = ROOT / "preprocessed_temperature_data.csv"
TRAINING_PAGE = ROOT / "pages" / "🤖_Обучить_модель.py"


def test_demo_training_creates_compatible_bundle_and_enables_inference(tmp_path):
    result = train_demo_model(dataset_path=DEMO_DATASET, model_dir=tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(MODEL_ARTIFACTS)
    assert result["artifacts"] == list(MODEL_ARTIFACTS)
    assert result["info"]["trained_on_normal"] is True
    assert result["info"]["split_strategy"] == "time"
    assert result["info"]["train_rows"] > 0
    assert result["info"]["evaluation_rows"] > 0

    metadata = json.loads(
        (tmp_path / "model_meta.json").read_text(encoding="utf-8")
    )
    assert metadata["trained_on_normal"] is True
    assert metadata["split_strategy"] == "time"
    assert metadata["test_start"] == result["info"]["test_start"]

    demo_df = pd.read_csv(DEMO_DATASET)
    results, _alarm_log = detect_anomalies(demo_df, model_dir=str(tmp_path))
    assert set(results["analysis_mode"].unique()) == {ANALYSIS_MODE_RULES_ML}
    assert results["iforest_prediction"].notna().any()


@pytest.mark.parametrize(
    "scenarios, message",
    [
        (None, "scenario"),
        (["sharp_jump"] * 30, "нет строк scenario == 'normal'"),
    ],
    ids=("missing-scenario", "no-normal-rows"),
)
def test_demo_training_rejects_unconfirmed_baseline(tmp_path, scenarios, message):
    dataset = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-06 12:00", periods=30, freq="1min"),
            "sensor_id": ["T-01"] * 30,
            "temperature_filled": [70.0] * 30,
            "rolling_mean": [70.0] * 30,
            "rolling_std": [1.0] * 30,
            "temp_diff": [0.0] * 30,
            "abs_temp_diff": [0.0] * 30,
            "abs_z_score": [0.0] * 30,
            "is_missing": [0] * 30,
            "is_stuck": [0] * 30,
            "abs_diff_from_group_mean": [0.0] * 30,
        }
    )
    if scenarios is not None:
        dataset["scenario"] = scenarios
    dataset_path = tmp_path / "unsafe_training.csv"
    dataset.to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match=message):
        train_demo_model(dataset_path=dataset_path, model_dir=tmp_path / "models")

    assert not (tmp_path / "models").exists()


def test_training_page_is_explicit_and_never_uses_uploaded_csv():
    source = TRAINING_PAGE.read_text(encoding="utf-8")

    assert 'st.button("Обучить демонстрационную ML-модель"' in source
    assert "train_demo_model()" in source
    assert "Ваш загруженный CSV в обучение не попадает" in source
    assert "uploaded_file" not in source
    assert "train_demo_model(uploaded" not in source
    assert "файловая система может очищаться" in source
