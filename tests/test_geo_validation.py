import pytest
from shapely.geometry import Polygon

import geopandas as gpd

from geo_daily_tools.geo_validation import (
    bounds_summary,
    geometry_quality_report,
    geometry_type_summary,
    nearest_neighbor_join,
    points_from_latlon,
    repair_invalid_geometries,
    reproject_if_needed,
    require_crs,
)
from geo_daily_tools.sample_data import messy_sensor_records
from geo_daily_tools.validation import validate_sensor_records


def test_points_from_latlon():
    valid_df, _ = validate_sensor_records(messy_sensor_records())
    gdf = points_from_latlon(valid_df)

    assert str(gdf.crs) == "EPSG:4326"
    assert "geometry" in gdf.columns
    assert len(gdf) == len(valid_df)


def test_geometry_quality_report():
    valid_df, _ = validate_sensor_records(messy_sensor_records())
    gdf = points_from_latlon(valid_df)
    report = geometry_quality_report(gdf)

    assert report["rows"] == len(gdf)
    assert report["invalid_geometry"] == 0


def test_require_crs_raises_on_missing():
    gdf = gpd.GeoDataFrame(geometry=[])
    with pytest.raises(ValueError):
        require_crs(gdf)


def test_reproject_if_needed_noop_when_same_crs():
    valid_df, _ = validate_sensor_records(messy_sensor_records())
    gdf = points_from_latlon(valid_df)
    out = reproject_if_needed(gdf, "EPSG:4326")
    assert out.crs == gdf.crs


def test_reproject_if_needed_changes_crs():
    valid_df, _ = validate_sensor_records(messy_sensor_records())
    gdf = points_from_latlon(valid_df)
    out = reproject_if_needed(gdf, "EPSG:3857")
    assert str(out.crs) != str(gdf.crs)


def test_geometry_type_summary():
    valid_df, _ = validate_sensor_records(messy_sensor_records())
    gdf = points_from_latlon(valid_df)
    summary = geometry_type_summary(gdf)
    assert summary.get("Point", 0) == len(gdf)


def test_bounds_summary_keys():
    valid_df, _ = validate_sensor_records(messy_sensor_records())
    gdf = points_from_latlon(valid_df)
    bounds = bounds_summary(gdf)
    assert set(bounds) == {"crs", "minx", "miny", "maxx", "maxy"}
    assert bounds["minx"] is not None


def test_repair_invalid_geometries_fixes_bowtie():
    bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[bowtie], crs="EPSG:3857")
    repaired, counts = repair_invalid_geometries(gdf)
    assert counts["invalid_before"] == 1
    assert counts["invalid_after"] == 0
    assert bool(repaired.geometry.is_valid.iloc[0]) is True


def test_nearest_neighbor_join_attaches_distance():
    pts = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=gpd.points_from_xy([0, 100], [0, 0]),
        crs="EPSG:3857",
    )
    targets = gpd.GeoDataFrame(
        {"name": ["origin"]},
        geometry=gpd.points_from_xy([0], [0]),
        crs="EPSG:3857",
    )
    joined = nearest_neighbor_join(pts, targets)
    assert "distance" in joined.columns
    assert joined.iloc[0]["distance"] == 0
    assert joined.iloc[1]["distance"] == 100
