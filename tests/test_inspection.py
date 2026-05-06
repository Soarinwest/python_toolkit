import numpy as np
import pandas as pd

from geo_daily_tools.inspection import (
    categorical_summary,
    column_overview,
    dtype_summary,
    grouped_numeric_summary,
    missingness_report,
)


def _df():
    return pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", None],
            "kind": ["x", "x", "y", "y", "y"],
            "value": [1.0, 2.0, np.nan, 4.0, 5.0],
        }
    )


def test_missingness_report_columns():
    report = missingness_report(_df())
    assert {"column", "missing_count", "missing_pct", "dtype"} <= set(report.columns)
    assert report.set_index("column").loc["id", "missing_count"] == 1


def test_column_overview_basic():
    overview = column_overview(_df())
    assert set(overview["column"]) == {"id", "kind", "value"}
    assert overview.set_index("column").loc["value", "n_missing"] == 1
    assert overview.set_index("column").loc["kind", "n_unique"] == 2


def test_column_overview_empty():
    overview = column_overview(pd.DataFrame())
    assert list(overview.columns) == [
        "column", "dtype", "n_missing", "missing_pct", "n_unique", "sample_value"
    ]
    assert len(overview) == 0


def test_dtype_summary_returns_counts():
    summary = dtype_summary(_df())
    assert summary.sum() == 3


def test_categorical_summary_picks_object_columns():
    summary = categorical_summary(_df())
    assert "kind" in summary["column"].values
    assert "value" not in summary["column"].values


def test_grouped_numeric_summary_aggregates():
    grouped = grouped_numeric_summary(_df(), group_cols="kind", value_cols="value")
    assert ("value", "count") in grouped.columns
    assert ("value", "mean") in grouped.columns
