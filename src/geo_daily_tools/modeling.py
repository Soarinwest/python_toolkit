"""Modeling-prep utilities for geospatial/tabular ML data."""

from __future__ import annotations

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
