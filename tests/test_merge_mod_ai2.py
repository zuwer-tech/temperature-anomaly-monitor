"""Тесты объединения mod_AI_2 с основным пайплайном (PR #7).

Проверяем ключевые исправления для реальных данных:
- правило зависания не ловит ступени квантования АЦП (точное равенство + физическая длительность);
- настоящее зависание ловится;
- для одного датчика group-правило использует отклонение от rolling_mean;
- на реальном Т2.csv доля ложных «зависаний» низкая (регрессия к mod_AI_2).
"""
import numpy as np
import pandas as pd

from preprocessing import preprocess_data
from rule_config import RULE_PARAMS


def _make_df(temps, sensor_id="T-01"):
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-06-06 12:00", periods=len(temps), freq="1s"),
        "sensor_id": sensor_id,
        "temperature": temps,
    })


def test_stuck_ignores_quantization():
    """Короткие повторы значения (ступени квантования) — НЕ зависание."""
    # Плавный рост, квантованный до 0.5°: каждое значение повторяется 3 раза (короче физического порога).
    levels = np.repeat(np.arange(50, 70, 0.5), 3)
    df = preprocess_data(_make_df(levels))
    assert df["is_stuck"].sum() == 0, "квантованные ступени не должны считаться зависанием"


def test_stuck_detects_true_stall():
    """Длинный застывший участок — зависание."""
    temps = np.concatenate([
        np.linspace(50, 60, 40),
        np.full(int(RULE_PARAMS["stuck_duration_seconds"]) + 5, 60.0),
        np.linspace(60, 65, 20),
    ])
    df = preprocess_data(_make_df(temps))
    assert df["is_stuck"].sum() >= 1, "настоящий застой должен детектироваться"


def test_single_sensor_group_uses_rolling_mean():
    """Для одного датчика group-отклонение = |temp - rolling_mean|, а не 0."""
    temps = np.concatenate([np.full(20, 50.0), np.full(20, 90.0)])
    df = preprocess_data(_make_df(temps))
    # На участке 90° отклонение от rolling_mean заметно (не 0, как при кросс-сенсорном).
    hot = df[df["temperature_filled"] > 80]
    assert (hot["abs_diff_from_group_mean"] > 0).any()


def test_real_t2_stuck_not_overfiring():
    """На реальных Т2.csv доля 'зависаний' < 10% (раньше было ~86%)."""
    from data_adapters import load_t2
    df = preprocess_data(load_t2("Т2.csv"))
    stuck_rate = df["is_stuck"].mean()
    assert stuck_rate < 0.10, f"stuck rate = {stuck_rate:.3f}"
