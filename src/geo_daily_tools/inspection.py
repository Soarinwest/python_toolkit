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
