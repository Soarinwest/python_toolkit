"""Initial pandas inspection utilities."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages by column."""
    if df.empty:
        return pd.DataFrame(columns=["column", "missing_count", "missing_pct", "dtype"])

    report = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values * 100).round(2),
            "dtype": [str(dtype) for dtype in df.dtypes.values],
        }
    )
    return report.sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)


def duplicate_report(df: pd.DataFrame, subset: Iterable[str] | None = None) -> dict[str, int]:
    """Return duplicate counts for an optional subset of columns."""
    duplicated_mask = df.duplicated(subset=list(subset) if subset else None, keep=False)
    return {
        "total_rows": int(len(df)),
        "duplicate_rows": int(duplicated_mask.sum()),
        "unique_rows": int((~duplicated_mask).sum()),
    }


def categorical_value_counts(
    df: pd.DataFrame,
    max_unique: int = 20,
    top_n: int = 10,
) -> dict[str, pd.Series]:
    """Return value counts for low-cardinality object/category columns."""
    results: dict[str, pd.Series] = {}
    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype) in {"category", "string"}:
            nunique = df[col].nunique(dropna=False)
            if nunique <= max_unique:
                results[col] = df[col].value_counts(dropna=False).head(top_n)
    return results


def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column overview: dtype, missing, unique, sample value."""
    if df.empty:
        return pd.DataFrame(
            columns=["column", "dtype", "n_missing", "missing_pct", "n_unique", "sample_value"]
        )

    samples = []
    for col in df.columns:
        non_null = df[col].dropna()
        samples.append(non_null.iloc[0] if len(non_null) else None)

    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(d) for d in df.dtypes.values],
            "n_missing": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values * 100).round(2),
            "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
            "sample_value": samples,
        }
    )


def dtype_summary(df: pd.DataFrame) -> pd.Series:
    """Return counts of columns grouped by dtype."""
    return df.dtypes.astype(str).value_counts().rename("n_columns")


def categorical_summary(df: pd.DataFrame, max_unique: int = 50) -> pd.DataFrame:
    """One-row-per-column summary for object/category/string columns."""
    rows = []
    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype) in {"category", "string"}:
            n_unique = df[col].nunique(dropna=True)
            if n_unique > max_unique:
                continue
            counts = df[col].value_counts(dropna=True)
            top_value = counts.index[0] if len(counts) else None
            top_count = int(counts.iloc[0]) if len(counts) else 0
            rows.append(
                {
                    "column": col,
                    "n_unique": int(n_unique),
                    "n_missing": int(df[col].isna().sum()),
                    "top_value": top_value,
                    "top_count": top_count,
                    "top_pct": round(top_count / len(df) * 100, 2) if len(df) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def grouped_numeric_summary(
    df: pd.DataFrame,
    group_cols: str | list[str],
    value_cols: str | list[str],
) -> pd.DataFrame:
    """Return count/mean/median/std/min/max per group for the given numeric columns."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if isinstance(value_cols, str):
        value_cols = [value_cols]

    aggs = ["count", "mean", "median", "std", "min", "max"]
    out = df.groupby(group_cols, dropna=False)[value_cols].agg(aggs)
    return out.reset_index()


def quick_inspect(df: pd.DataFrame, name: str = "df", print_output: bool = True) -> dict[str, object]:
    """Build a compact first-pass inspection report for a DataFrame.

    Returns a dictionary so the report can be printed, logged, or inspected in notebooks.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    report: dict[str, object] = {
        "name": name,
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missingness": missingness_report(df),
        "duplicates_all_columns": duplicate_report(df),
        "numeric_summary": df[numeric_cols].describe().T if numeric_cols else pd.DataFrame(),
        "categorical_value_counts": categorical_value_counts(df),
    }

    if print_output:
        print(f"\n=== {name}: quick inspection ===")
        print(f"Shape: {df.shape}")
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nDtypes:")
        print(df.dtypes)
        print("\nMissingness:")
        print(report["missingness"])
        print("\nDuplicate report:")
        print(report["duplicates_all_columns"])
        if numeric_cols:
            print("\nNumeric summary:")
            print(report["numeric_summary"])
        if report["categorical_value_counts"]:
            print("\nCategorical value counts:")
            for col, counts in report["categorical_value_counts"].items():
                print(f"\n{col}:")
                print(counts)

    return report
