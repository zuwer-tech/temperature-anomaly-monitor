"""Тесты адаптера реальных данных Т2.csv (PR3)."""
import pandas as pd
import pytest

from data_adapters import load_t2, _parse_glued_body, CANONICAL_COLUMNS


def test_load_t2_real_file():
    """Реальный Т2.csv из репозитория грузится в каноническую схему."""
    df = load_t2("Т2.csv")
    assert list(df.columns) == CANONICAL_COLUMNS
    assert len(df) > 0
    # Один датчик, как и ожидаем от одного канала реальных данных.
    assert df["sensor_id"].nunique() == 1
    # Температура числовая.
    assert pd.api.types.is_numeric_dtype(df["temperature"])
    # Время монотонно растёт.
    assert df["timestamp"].is_monotonic_increasing
    # scenario заполнен.
    assert (df["scenario"] == "user_data").all()


def test_load_t2_runs_through_pipeline(tmp_path, preprocessed_synth):
    """Реальные данные проходят предобработку и детекцию без ошибок."""
    from preprocessing import preprocess_data
    from anomaly_detection import detect_anomalies
    import train_model

    df = load_t2("Т2.csv")
    pre = preprocess_data(df)
    scaler, model, info = train_model.train(preprocessed_synth)
    train_model.save_model(scaler, model, info, model_dir=str(tmp_path))
    results, alarm_log = detect_anomalies(pre, model_dir=str(tmp_path))
    assert len(results) == len(df)
    assert "final_anomaly" in results.columns
    # Журнал тревог содержит только колонки, которые отображает приложение.
    assert "Тип_события" in alarm_log.columns


def test_parse_glued_body():
    """Regex-fallback разбирает CSV без переносов строк (склеенное тело)."""
    body = "0,49.001,49.592,50.153,50.154,50.155,50.68"
    df = _parse_glued_body(body)
    assert list(df.columns) == ["time_s", "temp_C"]
    assert df["time_s"].tolist() == [0, 1, 2, 3, 4, 5]
    assert df["temp_C"].tolist() == [49.00, 49.59, 50.15, 50.15, 50.15, 50.68]


def test_load_t2_from_glued_file(tmp_path):
    """Полный load_t2 работает на синтетическом «склеенном» файле без \\n."""
    content = "time_s,temp_C0,49.001,49.592,50.15"
    path = tmp_path / "glued.csv"
    path.write_text(content, encoding="utf-8")
    df = load_t2(str(path), start_time="2026-01-01 00:00:00")
    assert len(df) == 3
    assert df["temperature"].iloc[0] == pytest.approx(49.00)
    assert df["scenario"].iloc[0] == "user_data"
