"""Example: create a GeoDataFrame and run basic geometry checks."""

from geo_daily_tools.geo_validation import filter_bbox, geometry_quality_report, points_from_latlon
from geo_daily_tools.sample_data import messy_sensor_records
from geo_daily_tools.validation import validate_sensor_records


def main() -> None:
    valid_df, drop_summary = validate_sensor_records(messy_sensor_records())
    print("Drop summary:", drop_summary)

    gdf = points_from_latlon(valid_df)
    print("\nGeoDataFrame:")
    print(gdf)

    print("\nGeometry QA report:")
    print(geometry_quality_report(gdf))

    bbox_df = filter_bbox(valid_df, min_lat=44.50, max_lat=44.57, min_lon=-73.27, max_lon=-73.20)
    print("\nRecords inside bbox:")
    print(bbox_df)


if __name__ == "__main__":
    main()
