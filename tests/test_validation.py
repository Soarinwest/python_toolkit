from geo_daily_tools.sample_data import messy_sensor_records
from geo_daily_tools.validation import validate_sensor_records


def test_validate_sensor_records_counts():
    valid_df, invalid_df, summary = validate_sensor_records(messy_sensor_records(), return_invalid=True)

    assert len(valid_df) == 2
    assert len(invalid_df) == 7
    assert summary["empty_record"] == 1
    assert summary["missing_id"] == 2
    assert summary["invalid_coordinates"] == 2
    assert summary["invalid_reading"] == 2


def test_validate_sensor_records_coerces_numeric_strings():
    valid_df, _ = validate_sensor_records(messy_sensor_records())
    row = valid_df[valid_df["obs_id"] == "OBS-0147"].iloc[0]

    assert row["lat"] == 44.56
    assert row["lon"] == -73.26
    assert row["reading"] == 40.1
