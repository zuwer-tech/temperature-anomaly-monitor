"""Общие pytest-фикстуры.

Генерирует компактный синтетический набор температурных данных с размеченными
сценариями (аналог Data.py, но без побочных эффектов — без записи CSV и без
plt.show), чтобы тесты были быстрыми, детерминированными и независимыми от
больших закоммиченных артефактов.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def make_synth_df(n=300, num_sensors=6, seed=42):
    """Синтетический DataFrame с колонками timestamp, sensor_id, temperature, scenario.

    Воспроизводит ключевые сценарии Data.py: sharp_jump, slow_overheating,
    sensor_drift, stuck_sensor, high_noise, signal_loss, correlated_growth.
    """
    rng = np.random.RandomState(seed)
    base = 70.0
    t = np.arange(n)
    timestamps = pd.date_range("2026-06-06 12:00", periods=n, freq="1min")
    rows = []

    for s in range(1, num_sensors + 1):
        sid = f"T-{s:02d}"
        shift = rng.uniform(-1.5, 1.5)
        slow = 2 * np.sin(2 * np.pi * t / 180)
        noise = rng.normal(0, 1.0, n)
        temp = base + shift + slow + noise
        scen = np.array(["normal"] * n, dtype=object)

        if sid == "T-02":
            temp[120:140] += 15
            scen[120:140] = "sharp_jump"
            growth = np.linspace(0, 8, 30)
            temp[250:280] += growth
            scen[250:280] = "correlated_growth"

        if sid == "T-03":
            temp[160:240] += np.linspace(0, 12, 80)
            scen[160:240] = "slow_overheating"
            temp[250:280] += np.linspace(0, 8, 30)
            scen[250:280] = "correlated_growth"

        if sid == "T-04":
            temp[150:] += np.linspace(0, 10, n - 150)
            scen[150:] = "sensor_drift"

        if sid == "T-05":
            temp[180:220] = temp[180]
            scen[180:220] = "stuck_sensor"

        if sid == "T-06":
            temp[120:160] += rng.normal(0, 5, 40)
            scen[120:160] = "high_noise"
            miss = rng.choice(np.arange(120, 160), size=8, replace=False)
            temp[miss] = np.nan
            scen[miss] = "signal_loss"

        for i in range(n):
            rows.append({
                "timestamp": timestamps[i],
                "sensor_id": sid,
                "temperature": temp[i],
                "scenario": scen[i],
            })

    return pd.DataFrame(rows)


@pytest.fixture
def synth_df():
    """Сырой синтетический набор (до предобработки)."""
    return make_synth_df()


@pytest.fixture
def preprocessed_synth(synth_df):
    """Синтетический набор после preprocess_data."""
    from preprocessing import preprocess_data
    return preprocess_data(synth_df)


@pytest.fixture(scope="session")
def preprocessed_full_synth():
    """Полный текущий benchmark из synthetic_temperature_data.csv."""
    from preprocessing import preprocess_data

    repo_root = Path(__file__).resolve().parents[1]
    raw = pd.read_csv(repo_root / "synthetic_temperature_data.csv")
    return preprocess_data(raw)
