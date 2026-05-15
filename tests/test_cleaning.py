import numpy as np
import pandas as pd

from geo_daily_tools.cleaning import (
    apply_standard_cleaning,
    flag_blank_or_null,
    flag_duplicate_rows,
    normalize_null_like,
    required_columns_reason,
)


def test_normalize_null_like_handles_common_sentinels():
    df = pd.DataFrame(
        {
            "id": ["a", " ", "N/A", "null", "None", "-", "?", "real"],
            "value": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    out = normalize_null_like(df, columns=["id"])
    expected_missing = [False, True, True, True, True, True, True, False]
    assert out["id"].isna().tolist() == expected_missing
    assert out["value"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_normalize_null_like_case_insensitive_and_strips():
    df = pd.DataFrame({"x": ["  NA ", "n/a", "Null", "ok"]})
    out = normalize_null_like(df, columns=["x"])
    assert out["x"].isna().tolist() == [True, True, True, False]


def test_normalize_null_like_skips_numeric_columns_by_default():
    df = pd.DataFrame({"x": ["NA", "ok"], "n": [1, 2]})
    out = normalize_null_like(df)
    assert out["n"].tolist() == [1, 2]
    assert out["x"].isna().tolist() == [True, False]


def test_normalize_null_like_extra_tokens():
    df = pd.DataFrame({"x": ["pending", "ok", "PENDING"]})
    out = normalize_null_like(df, columns=["x"], extra_tokens=["pending"])
    assert out["x"].isna().tolist() == [True, False, True]


def test_flag_blank_or_null_strings_and_numerics():
    df = pd.DataFrame({"a": ["x", "  ", None, "y"], "b": [1.0, np.nan, 3.0, 4.0]})
    mask = flag_blank_or_null(df, ["a", "b"])
    assert mask.tolist() == [False, True, True, False]


def test_flag_blank_or_null_missing_column_flags_all_rows():
    df = pd.DataFrame({"a": ["x", "y"]})
    mask = flag_blank_or_null(df, ["a", "missing_col"])
    assert mask.tolist() == [True, True]


def test_required_columns_reason_first_hit_wins():
    df = pd.DataFrame(
        {
            "id": ["a", None, "c", "  "],
            "lat": [1.0, 2.0, np.nan, 4.0],
        }
    )
    reasons = required_columns_reason(df, ["id", "lat"])
    assert reasons.tolist() == [pd.NA, "missing:id", "missing:lat", "missing:id"]


def test_flag_duplicate_rows_subset():
    df = pd.DataFrame({"k": ["a", "a", "b", "a"], "v": [1, 2, 3, 4]})
    mask = flag_duplicate_rows(df, subset=["k"], keep="first")
    assert mask.tolist() == [False, True, False, True]


def test_apply_standard_cleaning_full_pipeline():
    df = pd.DataFrame(
        {
            "id": ["A", "  ", "N/A", "B", "B", None, "C"],
            "lat": [44.5, 44.6, 44.7, 44.8, 44.8, 44.9, np.nan],
            "note": ["ok", "", "null", "fine", "fine", "stuck", "ok"],
        }
    )
    clean, rejected, summary = apply_standard_cleaning(
        df,
        required=["id", "lat"],
        string_cols=["id", "note"],
        dedupe_subset=["id"],
    )

    assert summary.get("missing:id", 0) == 3
    assert summary.get("missing:lat", 0) == 1
    assert summary.get("duplicate", 0) == 1
    assert len(clean) == 2
    assert set(clean["id"]) == {"A", "B"}
    assert "drop_reason" not in clean.columns
    assert set(rejected["drop_reason"].unique()) == {"missing:id", "missing:lat", "duplicate"}


def test_apply_standard_cleaning_empty_row_takes_priority():
    df = pd.DataFrame(
        {
            "id": ["A", None],
            "lat": [44.5, np.nan],
        }
    )
    _, rejected, summary = apply_standard_cleaning(df, required=["id", "lat"])
    assert summary.get("empty_row") == 1
    assert "missing:id" not in summary
    assert rejected.iloc[0]["drop_reason"] == "empty_row"


def test_apply_standard_cleaning_no_required_returns_all_clean():
    df = pd.DataFrame({"x": ["a", "b"], "y": [1, 2]})
    clean, rejected, summary = apply_standard_cleaning(df)
    assert len(clean) == 2
    assert len(rejected) == 0
    assert summary == {}
