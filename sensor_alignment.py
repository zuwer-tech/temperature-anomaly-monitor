"""Причинное выравнивание измерений нескольких датчиков по времени."""

import numpy as np
import pandas as pd


def causal_group_mean(frame, value_column, tolerance_seconds):
    """Возвращает групповое среднее и число доступных соседних датчиков.

    Для строки в момент ``t`` используется текущее значение её датчика и
    последнее значение каждого другого датчика с timestamp <= t. Соседнее
    измерение участвует только пока его возраст не превышает допуск.
    """
    tolerance = float(tolerance_seconds)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError(
            "tolerance_seconds должна быть конечной неотрицательной длительностью."
        )

    required_columns = {"timestamp", "sensor_id", value_column}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            "Невозможно выровнять датчики: отсутствуют колонки "
            f"{', '.join(missing_columns)}."
        )

    group_mean = pd.Series(np.nan, index=frame.index, dtype=float)
    peer_count = pd.Series(0, index=frame.index, dtype=int)
    ordered = frame.assign(
        _alignment_timestamp=pd.to_datetime(frame["timestamp"]),
    ).sort_values(
        by=["_alignment_timestamp", "sensor_id"],
        kind="mergesort",
    )

    latest_by_sensor = {}
    for timestamp, simultaneous in ordered.groupby(
        "_alignment_timestamp",
        sort=False,
    ):
        # Точки с одинаковым timestamp одновременны, поэтому ни одна из них не
        # считается будущей относительно другой.
        for _index, row in simultaneous.iterrows():
            value = pd.to_numeric(row[value_column], errors="coerce")
            latest_by_sensor[row["sensor_id"]] = (timestamp, value)

        available = {
            sensor_id: value
            for sensor_id, (seen_at, value) in latest_by_sensor.items()
            if (
                0 <= (timestamp - seen_at).total_seconds() <= tolerance
                and np.isfinite(value)
            )
        }
        for index, row in simultaneous.iterrows():
            sensor_id = row["sensor_id"]
            own_value = pd.to_numeric(row[value_column], errors="coerce")
            peers = [
                value
                for peer_sensor, value in available.items()
                if peer_sensor != sensor_id
            ]
            peer_count.loc[index] = len(peers)
            if np.isfinite(own_value):
                group_mean.loc[index] = float(np.mean([own_value, *peers]))

    return group_mean, peer_count