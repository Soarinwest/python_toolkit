"""Pandas validation and QA/QC helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def ensure_columns(df: pd.DataFrame, columns: list[str], fill_value: Any = np.nan) -> pd.DataFrame:
    """Return a copy of df that includes all required columns."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = fill_value
    return out


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a copy of df with selected columns coerced to numeric."""
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_drop_reason(
    df: pd.DataFrame,
    rules: list[tuple[str, pd.Series | Callable[[pd.DataFrame], pd.Series]]],
    reason_col: str = "drop_reason",
) -> pd.DataFrame:
    """Assign the first matching drop reason to each row.

    Parameters
    ----------
    df:
        Input dataframe.
    rules:
        Ordered list of (reason, condition) pairs. Conditions can be boolean Series
        aligned to df or callables returning a boolean Series.
    reason_col:
        Column name where reasons are stored.
    """
    out = df.copy()
    out[reason_col] = pd.NA

    for reason, condition in rules:
        mask = condition(out) if callable(condition) else condition
        out.loc[out[reason_col].isna() & mask, reason_col] = reason

    return out


def drop_summary(df: pd.DataFrame, reason_col: str = "drop_reason") -> dict[str, int]:
    """Return counts by drop reason."""
    if reason_col not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[reason_col].value_counts(dropna=True).items()}


def validate_sensor_records(
    records: list[dict[str, Any]],
    id_col: str = "obs_id",
    lat_col: str = "lat",
    lon_col: str = "lon",
    reading_col: str = "reading",
    return_invalid: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]] | tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Validate messy sensor/geospatial observation records.

    Valid records must have:
    - non-empty id
    - numeric latitude in [-90, 90]
    - numeric longitude in [-180, 180]
    - numeric reading > 0
    """
    required_cols = [id_col, lat_col, lon_col, reading_col]
    df = pd.DataFrame(records)
    df = ensure_columns(df, required_cols)

    # Normalize core fields.
    df[id_col] = df[id_col].astype("string").str.strip()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[reading_col] = pd.to_numeric(df[reading_col], errors="coerce")

    empty_record = df[required_cols].isna().all(axis=1)
    missing_id = df[id_col].isna() | df[id_col].eq("")
    invalid_coordinates = (
        df[lat_col].isna()
        | df[lon_col].isna()
        | ~df[lat_col].between(-90, 90)
        | ~df[lon_col].between(-180, 180)
    )
    invalid_reading = df[reading_col].isna() | df[reading_col].le(0)

    rules = [
        ("empty_record", empty_record),
        ("missing_id", missing_id),
        ("invalid_coordinates", invalid_coordinates),
        ("invalid_reading", invalid_reading),
    ]
    qa_df = add_drop_reason(df, rules)

    valid_df = qa_df[qa_df["drop_reason"].isna()].drop(columns=["drop_reason"]).copy()
    invalid_df = qa_df[qa_df["drop_reason"].notna()].copy()
    summary = drop_summary(qa_df)

    if return_invalid:
        return valid_df, invalid_df, summary
    return valid_df, summary


def deduplicate_keep_best(
    df: pd.DataFrame,
    key_col: str,
    score_col: str,
    ascending: bool = False,
) -> pd.DataFrame:
    """Deduplicate by key, keeping the highest/lowest scoring row."""
    sorted_df = df.sort_values(score_col, ascending=ascending)
    return sorted_df.drop_duplicates(subset=[key_col], keep="first").copy()
