import pandas as pd

from geo_daily_tools.sample_data import messy_sensor_records
from geo_daily_tools.validation import (
    flag_outliers_iqr,
    range_check,
    strip_string_columns,
    valid_lat_lon_mask,
    validate_sensor_records,
    value_in_set_check,
)


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


def test_strip_string_columns_blank_to_na():
    df = pd.DataFrame({"id": ["  a ", "  ", "b"]})
    out = strip_string_columns(df, ["id"])
    assert out["id"].iloc[0] == "a"
    assert pd.isna(out["id"].iloc[1])
    assert out["id"].iloc[2] == "b"


def test_valid_lat_lon_mask():
    df = pd.DataFrame({"lat": [44.0, 91.0, "bad"], "lon": [-73.0, -73.0, -73.0]})
    mask = valid_lat_lon_mask(df)
    assert mask.tolist() == [True, False, False]


def test_flag_outliers_iqr_global():
    df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 100]})
    flags = flag_outliers_iqr(df, "v")
    assert flags.iloc[-1] is True or flags.iloc[-1] == True  # noqa: E712
    assert flags.iloc[:-1].sum() == 0


def test_flag_outliers_iqr_grouped():
    df = pd.DataFrame(
        {
            "g": ["a"] * 5 + ["b"] * 5,
            "v": [1, 2, 3, 4, 100, 10, 11, 12, 13, 14],
        }
    )
    flags = flag_outliers_iqr(df, "v", group_cols="g")
    assert bool(flags.iloc[4]) is True
    assert flags.iloc[5:].sum() == 0


def test_range_check_counts():
    df = pd.DataFrame({"v": [-1, 0, 5, 99]})
    rep = range_check(df, {"v": (0, 10)})
    row = rep.iloc[0]
    assert row["n_below"] == 1
    assert row["n_above"] == 1


def test_value_in_set_check():
    df = pd.DataFrame({"kind": ["a", "b", "c", None]})
    rep = value_in_set_check(df, {"kind": {"a", "b"}})
    row = rep.iloc[0]
    assert row["n_disallowed"] == 1
    assert row["examples"] == ["c"]
