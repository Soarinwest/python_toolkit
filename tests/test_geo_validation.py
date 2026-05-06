from geo_daily_tools.geo_validation import geometry_quality_report, points_from_latlon
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
