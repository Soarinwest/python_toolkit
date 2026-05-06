"""Example: pandas-first cleaning and QA/QC for messy geospatial sensor records."""

from geo_daily_tools.inspection import quick_inspect
from geo_daily_tools.sample_data import messy_sensor_records
from geo_daily_tools.validation import validate_sensor_records


def main() -> None:
    records = messy_sensor_records()

    valid_df, invalid_df, drop_summary = validate_sensor_records(records, return_invalid=True)

    print("\nDrop summary:")
    print(drop_summary)

    print("\nValid records:")
    print(valid_df)

    print("\nInvalid records with reasons:")
    print(invalid_df[["obs_id", "lat", "lon", "reading", "drop_reason"]])

    quick_inspect(valid_df, name="valid_sensor_records")

    if "sensor_type" in valid_df.columns:
        print("\nReading summary by sensor_type:")
        print(valid_df.groupby("sensor_type", dropna=False)["reading"].describe())


if __name__ == "__main__":
    main()
