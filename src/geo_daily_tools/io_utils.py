"""Lightweight file IO helpers."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read CSV, Parquet, or Excel based on extension."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)

    raise ValueError(f"Unsupported table format: {suffix}")


def read_geodata(path: str | Path, **kwargs) -> gpd.GeoDataFrame:
    """Read vector geospatial data with GeoPandas."""
    return gpd.read_file(path, **kwargs)


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False, **kwargs) -> None:
    """Write CSV or Parquet based on extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(path, index=index, **kwargs)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=index, **kwargs)
        return

    raise ValueError(f"Unsupported table output format: {suffix}")
