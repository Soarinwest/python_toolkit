"""Common dataframe cleaning: null-like normalization and drop-reason flagging.

Sits one layer above :mod:`geo_daily_tools.validation` and handles the everyday
"first pass" you almost always run before domain checks: turning placeholder
strings (``""``, ``"N/A"``, ``"null"``, ``"-"`` ...) into proper NA, flagging
rows that are empty or missing required fields, and surfacing duplicates.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from geo_daily_tools.validation import add_drop_reason, drop_summary, strip_string_columns


DEFAULT_NULL_LIKE_TOKENS: frozenset[str] = frozenset(
    {
        "",
        "na",
        "n/a",
        "null",
        "none",
        "nan",
        "nil",
        "-",
        "--",
        "?",
        "unknown",
        "tbd",
    }
)


def _string_like_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype) == "string"]


def _missing_mask(series: pd.Series) -> pd.Series:
    """True where a value is NA, or a whitespace-only / empty string."""
    if series.dtype == "object" or str(series.dtype) == "string":
        stripped = series.astype("string").str.strip()
        return stripped.isna() | stripped.eq("")
    return series.isna()


def normalize_null_like(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    extra_tokens: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Replace common null-like string sentinels with ``pd.NA``.

    Comparison is case-insensitive and whitespace-trimmed. Numeric columns are
    left untouched; only object/string-typed columns are normalized by default.
    """
    tokens = {t.casefold() for t in DEFAULT_NULL_LIKE_TOKENS}
    if extra_tokens:
        tokens.update(t.casefold() for t in extra_tokens)

    out = df.copy()
    target_cols = columns if columns is not None else _string_like_columns(out)

    for col in target_cols:
        if col not in out.columns:
            continue
        as_str = out[col].astype("string").str.strip()
        mask = as_str.str.casefold().isin(tokens)
        out.loc[mask.fillna(False), col] = pd.NA

    return out


def flag_blank_or_null(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Boolean mask: ``True`` where any listed column is NA / blank / whitespace."""
    result = pd.Series(False, index=df.index)
    for col in columns:
        if col not in df.columns:
            result = result | True
            continue
        result = result | _missing_mask(df[col])
    return result


def flag_duplicate_rows(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str | bool = "first",
) -> pd.Series:
    """Boolean mask for duplicated rows. See :meth:`pandas.DataFrame.duplicated`."""
    return df.duplicated(subset=subset, keep=keep)


def required_columns_reason(
    df: pd.DataFrame,
    required: list[str],
    prefix: str = "missing:",
) -> pd.Series:
    """Per-row reason for the first required column that is NA / blank.

    Walks ``required`` in order and assigns ``f"{prefix}{col}"`` to the first hit;
    rows where every required column is present get ``pd.NA``.
    """
    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    for col in required:
        if col not in df.columns:
            missing = pd.Series(True, index=df.index)
        else:
            missing = _missing_mask(df[col])
        reason = reason.mask(reason.isna() & missing, f"{prefix}{col}")
    return reason


def apply_standard_cleaning(
    df: pd.DataFrame,
    required: list[str] | None = None,
    string_cols: list[str] | None = None,
    dedupe_subset: list[str] | None = None,
    null_tokens: Iterable[str] | None = None,
    reason_col: str = "drop_reason",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Run the common cleaning pass and return ``(clean, rejected, summary)``.

    Pipeline:

    1. Strip whitespace on string columns; blank strings become NA.
    2. Normalize null-like sentinels (``"N/A"``, ``"null"``, ``"-"`` ...) to NA.
    3. Flag rows where every column is NA as ``empty_row``.
    4. Flag rows missing any required column as ``missing:<col>`` (first hit wins).
    5. If ``dedupe_subset`` is given, flag duplicates (after the first) as ``duplicate``.

    Reasons are applied in order: the first matching rule keeps its label, so an
    empty row is reported as ``empty_row`` rather than every missing column.
    """
    out = strip_string_columns(df, columns=string_cols)
    out = normalize_null_like(out, columns=string_cols, extra_tokens=null_tokens)

    rules: list[tuple[str, pd.Series]] = [("empty_row", out.isna().all(axis=1))]

    if required:
        for col in required:
            if col in out.columns:
                missing = _missing_mask(out[col])
            else:
                missing = pd.Series(True, index=out.index)
            rules.append((f"missing:{col}", missing))

    if dedupe_subset:
        rules.append(("duplicate", flag_duplicate_rows(out, subset=dedupe_subset)))

    flagged = add_drop_reason(out, rules, reason_col=reason_col)
    clean = flagged[flagged[reason_col].isna()].drop(columns=[reason_col]).copy()
    rejected = flagged[flagged[reason_col].notna()].copy()
    summary = drop_summary(flagged, reason_col=reason_col)
    return clean, rejected, summary
