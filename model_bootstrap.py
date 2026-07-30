"""Явное обучение демонстрационной ML-модели из интерфейса.

Этот модуль использует только встроенный размеченный baseline проекта.
Пользовательский CSV сюда не передаётся: он остаётся неизвестной пробой для
последующего inference.
"""
from pathlib import Path

import pandas as pd

from anomaly_detection import _load_or_fit_iforest
from train_model import evaluate, prepare_features, save_model, train


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DEMO_DATASET = PROJECT_ROOT / "preprocessed_temperature_data.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
MODEL_ARTIFACTS = ("scaler.joblib", "iforest.joblib", "model_meta.json")


def train_demo_model(
    dataset_path=DEFAULT_DEMO_DATASET,
    model_dir=DEFAULT_MODEL_DIR,
):
    """Обучает и сохраняет модель только на встроенном размеченном baseline."""
    dataset_path = Path(dataset_path)
    model_dir = Path(model_dir)

    if not dataset_path.is_file():
        raise FileNotFoundError(
            "Не найден встроенный набор для обучения: "
            f"{dataset_path}."
        )

    training_df = pd.read_csv(dataset_path)
    scaler, model, info = train(training_df)
    report = evaluate(training_df, scaler, model)
    save_model(scaler, model, info, model_dir=str(model_dir))

    missing = [
        name for name in MODEL_ARTIFACTS if not (model_dir / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            "Обучение завершилось без полного комплекта артефактов: "
            + ", ".join(missing)
        )

    _prepared_df, feature_frame = prepare_features(training_df)
    _load_or_fit_iforest(feature_frame.head(1), model_dir=str(model_dir))

    return {
        "dataset_path": str(dataset_path),
        "model_dir": str(model_dir),
        "info": info,
        "report": report,
        "artifacts": list(MODEL_ARTIFACTS),
    }
