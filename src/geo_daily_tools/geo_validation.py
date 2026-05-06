"""GeoPandas validation and geometry helpers."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd


def points_from_latlon(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame from latitude/longitude columns."""
    missing = [col for col in [lat_col, lon_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required coordinate columns: {missing}")

    out = df.copy()
    geometry = gpd.points_from_xy(out[lon_col], out[lat_col])
    return gpd.GeoDataFrame(out, geometry=geometry, crs=crs)


def geometry_quality_report(gdf: gpd.GeoDataFrame) -> dict[str, int | str | None]:
    """Return basic geometry QA/QC information."""
    if "geometry" not in gdf.columns:
        raise ValueError("Input GeoDataFrame has no geometry column.")

    return {
        "rows": int(len(gdf)),
        "crs": str(gdf.crs) if gdf.crs else None,
        "missing_geometry": int(gdf.geometry.isna().sum()),
        "empty_geometry": int(gdf.geometry.is_empty.sum()),
        "valid_geometry": int(gdf.geometry.is_valid.sum()),
        "invalid_geometry": int((~gdf.geometry.is_valid).sum()),
    }


def filter_bbox(
    df: pd.DataFrame,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """Filter latitude/longitude records to a bounding box."""
    mask = df[lat_col].between(min_lat, max_lat) & df[lon_col].between(min_lon, max_lon)
    return df.loc[mask].copy()


def reproject_for_metric_ops(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """Reproject before distance/area calculations."""
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS. Set CRS before reprojecting.")
    return gdf.to_crs(target_crs)
