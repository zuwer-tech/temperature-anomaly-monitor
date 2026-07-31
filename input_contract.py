"""Shared input contract for timestamps and optional measurement metadata."""

from dataclasses import dataclass

import pandas as pd


CELSIUS_UNIT_ALIASES = frozenset(
    {"c", "°c", "degc", "celsius", "градус c", "градусы c"}
)


@dataclass(frozen=True)
class TimestampInspection:
    """Parsed timestamps plus the policy facts needed by callers."""

    normalized: pd.Series
    invalid_count: int
    mode: str
    source_timezones: tuple[str, ...]
    normalized_timezone: str | None


def inspect_timestamp_values(values):
    """Parse timestamps without silently mixing zoned and unzoned values."""
    parsed = []
    kinds = set()
    source_timezones = set()
    invalid_count = 0

    for value in values:
        if pd.isna(value):
            parsed.append(pd.NaT)
            invalid_count += 1
            continue
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            parsed.append(pd.NaT)
            invalid_count += 1
            continue
        if pd.isna(timestamp):
            parsed.append(pd.NaT)
            invalid_count += 1
            continue

        is_aware = (
            timestamp.tzinfo is not None
            and timestamp.utcoffset() is not None
        )
        if is_aware:
            kinds.add("aware")
            source_timezones.add(str(timestamp.tzinfo))
        else:
            kinds.add("naive")
        parsed.append(timestamp)

    if kinds == {"aware"}:
        mode = "aware_utc"
        normalized = pd.Series(
            pd.to_datetime(parsed, utc=True),
            index=values.index,
            name=values.name,
        )
        normalized_timezone = "UTC"
    elif kinds == {"naive"}:
        mode = "naive"
        normalized = pd.Series(
            pd.to_datetime(parsed),
            index=values.index,
            name=values.name,
        )
        normalized_timezone = None
    elif kinds == {"aware", "naive"}:
        mode = "mixed"
        normalized = pd.Series(
            parsed,
            index=values.index,
            name=values.name,
            dtype="object",
        )
        normalized_timezone = None
    else:
        mode = "invalid"
        normalized = pd.Series(
            parsed,
            index=values.index,
            name=values.name,
            dtype="datetime64[ns]",
        )
        normalized_timezone = None

    return TimestampInspection(
        normalized=normalized,
        invalid_count=invalid_count,
        mode=mode,
        source_timezones=tuple(sorted(source_timezones)),
        normalized_timezone=normalized_timezone,
    )


def timestamp_policy_description(inspection):
    """Return a short user-facing explanation of the applied time policy."""
    if inspection.mode == "aware_utc":
        sources = ", ".join(inspection.source_timezones)
        return (
            f"Исходные зоны/смещения: {sources}; "
            "для расчётов время нормализовано в UTC."
        )
    if inspection.mode == "naive":
        return (
            "Часовой пояс не указан; все метки считаются единым "
            "локальным временем источника и не переводятся."
        )
    if inspection.mode == "mixed":
        return (
            "Смешаны метки с часовым поясом и без него; "
            "анализ заблокирован."
        )
    return "Корректный часовой пояс определить невозможно."


def _normalized_unit(value):
    return str(value).strip().lower().replace(" ", "")


def inspect_measurement_metadata(df):
    """Summarize optional metadata without changing measurements or ML features."""
    observed_units = []
    unsupported_units = []
    if "temperature_unit" in df.columns:
        observed_units = sorted(
            {
                str(value).strip()
                for value in df["temperature_unit"].dropna()
                if str(value).strip()
            }
        )
        normalized_aliases = {
            alias.replace(" ", "") for alias in CELSIUS_UNIT_ALIASES
        }
        unsupported_units = [
            value
            for value in observed_units
            if _normalized_unit(value) not in normalized_aliases
        ]

    accuracy_declared_count = 0
    invalid_accuracy_count = 0
    if "sensor_accuracy" in df.columns:
        accuracy = df["sensor_accuracy"]
        declared = accuracy.notna()
        accuracy_declared_count = int(declared.sum())
        numeric_accuracy = pd.to_numeric(accuracy, errors="coerce")
        invalid_accuracy_count = int(
            (declared & (numeric_accuracy.isna() | numeric_accuracy.lt(0))).sum()
        )

    quality_flags = []
    if "quality_flag" in df.columns:
        quality_flags = sorted(
            {
                str(value).strip()
                for value in df["quality_flag"].dropna()
                if str(value).strip()
            }
        )

    return {
        "temperature_units": observed_units,
        "unsupported_temperature_units": unsupported_units,
        "sensor_accuracy_declared_count": accuracy_declared_count,
        "invalid_sensor_accuracy_count": invalid_accuracy_count,
        "quality_flags": quality_flags,
    }


def validate_measurement_metadata(df):
    """Reject ambiguous units and invalid accuracy declarations."""
    metadata = inspect_measurement_metadata(df)
    unsupported = metadata["unsupported_temperature_units"]
    if unsupported:
        raise ValueError(
            "temperature_unit содержит неподдерживаемые единицы: "
            + ", ".join(unsupported)
            + ". Автоматический перевод из °F или K не выполняется; "
            "передайте температуру в °C."
        )
    invalid_accuracy_count = metadata["invalid_sensor_accuracy_count"]
    if invalid_accuracy_count:
        raise ValueError(
            "sensor_accuracy содержит "
            f"{invalid_accuracy_count} нечисловых или отрицательных значений."
        )
    return metadata
