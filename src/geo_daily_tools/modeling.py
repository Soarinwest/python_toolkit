"""Modeling-prep utilities for geospatial/tabular ML data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def feature_missingness_report(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Return missingness by model feature."""
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    return pd.DataFrame(
        {
            "feature": feature_cols,
            "missing_count": [int(df[col].isna().sum()) for col in feature_cols],
            "missing_pct": [round(float(df[col].isna().mean() * 100), 2) for col in feature_cols],
            "dtype": [str(df[col].dtype) for col in feature_cols],
        }
    ).sort_values("missing_count", ascending=False)


def prepare_model_inputs(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    drop_missing: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare X/y and return the model-ready dataframe used."""
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required model columns: {missing_cols}")

    model_df = df.copy()
    if drop_missing:
        model_df = model_df.dropna(subset=feature_cols + [target_col]).copy()

    X = model_df[feature_cols].copy()
    y = model_df[target_col].copy()
    return X, y, model_df


def group_train_test_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by group to reduce spatial/scene leakage."""
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def random_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plain random split. Use carefully for geospatial data."""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df.copy(), test_df.copy()


def simple_group_holdout(
    df: pd.DataFrame,
    group_col: str,
    test_groups: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by an explicit list of held-out group values."""
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    test_set = set(test_groups)
    is_test = df[group_col].isin(test_set)
    return df.loc[~is_test].copy(), df.loc[is_test].copy()


def check_group_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str,
) -> dict[str, int | list]:
    """Report whether any group appears in both train and test."""
    train_groups = set(train_df[group_col].dropna().unique())
    test_groups = set(test_df[group_col].dropna().unique())
    overlap = sorted(train_groups & test_groups)
    return {
        "n_train_groups": len(train_groups),
        "n_test_groups": len(test_groups),
        "n_overlapping_groups": len(overlap),
        "overlapping_groups": overlap,
    }


def compare_distributions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Side-by-side mean/std/min/max for shared columns; flags suspicious gaps."""
    rows = []
    for col in columns:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        t = pd.to_numeric(train_df[col], errors="coerce")
        v = pd.to_numeric(test_df[col], errors="coerce")
        rows.append(
            {
                "column": col,
                "train_mean": float(t.mean()),
                "test_mean": float(v.mean()),
                "train_std": float(t.std()),
                "test_std": float(v.std()),
                "train_min": float(t.min()),
                "test_min": float(v.min()),
                "train_max": float(t.max()),
                "test_max": float(v.max()),
            }
        )
    return pd.DataFrame(rows)


def coordinate_grid_block(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    size_deg: float = 0.5,
    block_col: str = "grid_block",
) -> pd.DataFrame:
    """Add a coarse spatial-block id by floor-bucketing lat/lon.

    Useful as a `group_col` for spatial CV when no natural group exists.
    Block size is in degrees; for projected coords, pick a unit-appropriate size.
    """
    out = df.copy()
    lat_bin = np.floor(pd.to_numeric(out[lat_col], errors="coerce") / size_deg).astype("Int64")
    lon_bin = np.floor(pd.to_numeric(out[lon_col], errors="coerce") / size_deg).astype("Int64")
    out[block_col] = lat_bin.astype("string") + "_" + lon_bin.astype("string")
    return out
