"""Причинные временные окна для рядов с реальными timestamp."""

import numpy as np
import pandas as pd


def _positive_seconds(value, name):
    seconds = float(value)
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{name} должно быть положительной длительностью в секундах.")
    return seconds


def _causal_segments(frame, max_gap_seconds):
    """Даёт непрерывные отрезки и явно отклоняет неположительные интервалы."""
    max_gap_seconds = _positive_seconds(max_gap_seconds, "max_gap_seconds")
    for sensor_id, group in frame.groupby("sensor_id", sort=False):
        timestamps = pd.to_datetime(group["timestamp"])
        deltas = timestamps.diff().dt.total_seconds()
        invalid = deltas.iloc[1:] <= 0
        if invalid.any():
            raise ValueError(
                "timestamp внутри датчика должен строго возрастать: "
                f"обнаружен нулевой или отрицательный интервал у {sensor_id}."
            )

        split_positions = np.flatnonzero(
            deltas.to_numpy()[1:] > max_gap_seconds
        ) + 1
        starts = np.r_[0, split_positions]
        stops = np.r_[split_positions, len(group)]
        for start, stop in zip(starts, stops):
            yield group.iloc[int(start):int(stop)]


def causal_time_rolling(
    frame,
    value_column,
    duration_seconds,
    aggregation,
    max_gap_seconds,
):
    """Считает mean/std в окне [t-duration, t] без будущих точек."""
    duration_seconds = _positive_seconds(duration_seconds, "duration_seconds")
    if aggregation not in {"mean", "std"}:
        raise ValueError("aggregation должна быть 'mean' или 'std'.")

    result = pd.Series(np.nan, index=frame.index, dtype=float)
    window = pd.Timedelta(seconds=duration_seconds)
    for segment in _causal_segments(frame, max_gap_seconds):
        timestamps = pd.DatetimeIndex(pd.to_datetime(segment["timestamp"]))
        values = pd.Series(
            pd.to_numeric(segment[value_column], errors="coerce").to_numpy(),
            index=timestamps,
        )
        rolling = values.rolling(window=window, min_periods=1, closed="both")
        aggregated = getattr(rolling, aggregation)()
        result.loc[segment.index] = aggregated.to_numpy()
    return result


def causal_time_slope(
    frame,
    value_column,
    duration_seconds,
    max_gap_seconds,
):
    """Возвращает наклон °C/мин по фактическому времени причинного окна."""
    duration_seconds = _positive_seconds(duration_seconds, "duration_seconds")
    result = pd.Series(np.nan, index=frame.index, dtype=float)

    for segment in _causal_segments(frame, max_gap_seconds):
        timestamps = pd.DatetimeIndex(pd.to_datetime(segment["timestamp"]))
        seconds = timestamps.asi8.astype(float) / 1_000_000_000
        values = pd.to_numeric(segment[value_column], errors="coerce").to_numpy(float)
        left = 0
        for right in range(len(segment)):
            if seconds[right] - seconds[0] < duration_seconds:
                continue
            window_start = seconds[right] - duration_seconds
            while left < right and seconds[left] < window_start:
                left += 1
            window_values = values[left:right + 1]
            valid = np.isfinite(window_values)
            if valid.sum() < 2:
                continue
            x_minutes = (seconds[left:right + 1][valid] - seconds[right]) / 60.0
            result.loc[segment.index[right]] = np.polyfit(
                x_minutes,
                window_values[valid],
                1,
            )[0]
    return result


def causal_stuck_flags(
    frame,
    value_column,
    duration_seconds,
    max_gap_seconds,
    absolute_tolerance,
):
    """Помечает сигнал, равный непрерывно заданную физическую длительность."""
    duration_seconds = _positive_seconds(duration_seconds, "duration_seconds")
    tolerance = float(absolute_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("absolute_tolerance должна быть конечной и неотрицательной.")

    result = pd.Series(0, index=frame.index, dtype=int)
    for segment in _causal_segments(frame, max_gap_seconds):
        timestamps = pd.DatetimeIndex(pd.to_datetime(segment["timestamp"]))
        seconds = timestamps.asi8.astype(float) / 1_000_000_000
        values = pd.to_numeric(segment[value_column], errors="coerce").to_numpy(float)
        run_start = 0
        for position, value in enumerate(values):
            if (
                position == 0
                or not np.isfinite(value)
                or not np.isfinite(values[position - 1])
                or abs(value - values[position - 1]) >= tolerance
            ):
                run_start = position
            if (
                np.isfinite(value)
                and seconds[position] - seconds[run_start] >= duration_seconds
            ):
                result.loc[segment.index[position]] = 1
    return result
