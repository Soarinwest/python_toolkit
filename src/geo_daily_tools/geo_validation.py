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


def require_crs(gdf: gpd.GeoDataFrame, expected_crs: str | None = None) -> gpd.GeoDataFrame:
    """Raise if gdf has no CRS, or if it doesn't match expected_crs."""
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS set.")
    if expected_crs is not None and str(gdf.crs) != str(gpd.GeoSeries([], crs=expected_crs).crs):
        raise ValueError(f"Expected CRS {expected_crs}, got {gdf.crs}.")
    return gdf


def reproject_if_needed(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """Return gdf reprojected to target_crs, or unchanged if already there."""
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS. Set CRS before reprojecting.")
    target = gpd.GeoSeries([], crs=target_crs).crs
    if gdf.crs == target:
        return gdf
    return gdf.to_crs(target_crs)


def repair_invalid_geometries(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Run make_valid on invalid geometries; return (repaired_gdf, counts)."""
    invalid_before = int((~gdf.geometry.is_valid).sum())
    out = gdf.copy()
    invalid_mask = ~out.geometry.is_valid
    out.loc[invalid_mask, out.geometry.name] = out.loc[invalid_mask, out.geometry.name].make_valid()
    invalid_after = int((~out.geometry.is_valid).sum())
    return out, {
        "invalid_before": invalid_before,
        "invalid_after": invalid_after,
        "repaired": invalid_before - invalid_after,
    }


def geometry_type_summary(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Return counts by geometry type (Point, Polygon, etc.)."""
    return gdf.geometry.geom_type.value_counts(dropna=False).rename("n")


def bounds_summary(gdf: gpd.GeoDataFrame) -> dict[str, float | str | None]:
    """Return bounding box and CRS for a GeoDataFrame."""
    if gdf.empty:
        return {"crs": str(gdf.crs) if gdf.crs else None,
                "minx": None, "miny": None, "maxx": None, "maxy": None}
    minx, miny, maxx, maxy = gdf.total_bounds
    return {
        "crs": str(gdf.crs) if gdf.crs else None,
        "minx": float(minx),
        "miny": float(miny),
        "maxx": float(maxx),
        "maxy": float(maxy),
    }


def nearest_neighbor_join(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    distance_col: str = "distance",
    max_distance: float | None = None,
) -> gpd.GeoDataFrame:
    """Attach the nearest right-feature to each left row.

    Both inputs must share a CRS. For meaningful distance values, use a
    projected CRS (meters/feet) — not EPSG:4326 degrees.
    """
    if left.crs is None or right.crs is None:
        raise ValueError("Both GeoDataFrames must have a CRS set.")
    if left.crs != right.crs:
        raise ValueError(f"CRS mismatch: {left.crs} vs {right.crs}. Reproject first.")

    return gpd.sjoin_nearest(
        left,
        right,
        how="left",
        distance_col=distance_col,
        max_distance=max_distance,
    )
