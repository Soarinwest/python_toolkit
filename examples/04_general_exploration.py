"""Example: general DataFrame exploration across file formats and PostGIS."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from geo_daily_tools.inspection import (
    categorical_summary,
    column_overview,
    dtype_summary,
    quick_inspect,
)
from geo_daily_tools.io_utils import read_geodata, read_table


def load_tabular(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV / Parquet / Excel file into a DataFrame."""
    df = read_table(path, **kwargs)
    print(f"Loaded tabular file '{path}' -> shape {df.shape}")
    return df


def load_vector(path: str | Path, **kwargs: Any) -> gpd.GeoDataFrame:
    """Load any GeoPandas-readable vector file (GeoJSON, SHP, GPKG, ...)."""
    gdf = read_geodata(path, **kwargs)
    print(f"Loaded vector file '{path}' -> shape {gdf.shape}, CRS={gdf.crs}")
    return gdf


def load_postgis(
    query: str,
    *,
    host: str,
    database: str,
    user: str,
    password: str,
    port: int = 5432,
    geom_col: str = "geom",
) -> gpd.GeoDataFrame:
    """Run a SQL query against PostGIS and return a GeoDataFrame.

    Requires ``sqlalchemy`` and ``psycopg2`` (or ``psycopg``) to be installed.
    """
    from sqlalchemy import create_engine

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(url)
    gdf = gpd.read_postgis(query, engine, geom_col=geom_col)
    print(f"Loaded PostGIS query -> shape {gdf.shape}, CRS={gdf.crs}")
    return gdf


def explore(df: pd.DataFrame, name: str = "df") -> None:
    """Run a general first-pass exploration on any DataFrame."""
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape}")

    print("\nHead:")
    print(df.head())

    print("\nDtype summary:")
    print(dtype_summary(df))

    print("\nColumn overview:")
    print(column_overview(df))

    cat_summary = categorical_summary(df)
    if not cat_summary.empty:
        print("\nCategorical summary:")
        print(cat_summary)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        print("\nNumeric describe:")
        print(df[numeric_cols].describe().T)

    if isinstance(df, gpd.GeoDataFrame):
        print(f"\nCRS: {df.crs}")
        print("Geometry types:")
        print(df.geom_type.value_counts(dropna=False))
        print("Total bounds:", df.total_bounds)


def main() -> None:
    # 1) Tabular file (auto-detected by extension)
    # df = load_tabular("data/some_file.csv")
    # explore(df, name="some_file.csv")

    # 2) Vector file (GeoJSON, Shapefile, GeoPackage, ...)
    # gdf = load_vector("data/parcels.geojson")
    # explore(gdf, name="parcels")

    # 3) PostGIS query
    # gdf = load_postgis(
    #     "SELECT id, name, geom FROM public.parcels LIMIT 1000",
    #     host="localhost",
    #     database="gis",
    #     user="postgres",
    #     password="postgres",
    # )
    # explore(gdf, name="parcels_postgis")

    # Fallback demo so the script runs out of the box:
    demo = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "category": ["a", "b", "a", "c", None],
            "value": [10.1, 12.4, None, 9.8, 15.0],
        }
    )
    explore(demo, name="demo")
    quick_inspect(demo, name="demo")


if __name__ == "__main__":
    main()
