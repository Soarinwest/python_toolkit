import pandas as pd

from geo_daily_tools.modeling import (
    check_group_leakage,
    compare_distributions,
    coordinate_grid_block,
    group_train_test_split,
    prepare_model_inputs,
    simple_group_holdout,
)
from geo_daily_tools.sample_data import messy_sensor_records
from geo_daily_tools.validation import validate_sensor_records


def test_prepare_model_inputs():
    valid_df, _ = validate_sensor_records(messy_sensor_records())
    X, y, model_df = prepare_model_inputs(valid_df, ["lat", "lon"], "reading")

    assert X.shape[1] == 2
    assert len(y) == len(model_df)


def test_group_train_test_split_no_group_overlap():
    valid_df, _ = validate_sensor_records(messy_sensor_records())
    train_df, test_df = group_train_test_split(valid_df, group_col="tile_id", test_size=0.5)

    assert set(train_df["tile_id"]).isdisjoint(set(test_df["tile_id"]))


def test_simple_group_holdout_splits_on_named_groups():
    df = pd.DataFrame({"tile": ["a", "a", "b", "c"], "v": [1, 2, 3, 4]})
    train, test = simple_group_holdout(df, "tile", ["b"])
    assert set(train["tile"]) == {"a", "c"}
    assert set(test["tile"]) == {"b"}


def test_check_group_leakage_detects_overlap():
    train = pd.DataFrame({"g": ["a", "b"]})
    test = pd.DataFrame({"g": ["b", "c"]})
    report = check_group_leakage(train, test, "g")
    assert report["n_overlapping_groups"] == 1
    assert report["overlapping_groups"] == ["b"]


def test_check_group_leakage_clean_split():
    train = pd.DataFrame({"g": ["a", "b"]})
    test = pd.DataFrame({"g": ["c", "d"]})
    report = check_group_leakage(train, test, "g")
    assert report["n_overlapping_groups"] == 0


def test_compare_distributions_shape():
    train = pd.DataFrame({"v": [1, 2, 3, 4]})
    test = pd.DataFrame({"v": [5, 6, 7, 8]})
    out = compare_distributions(train, test, ["v"])
    assert "train_mean" in out.columns and "test_mean" in out.columns
    assert out.iloc[0]["train_mean"] < out.iloc[0]["test_mean"]


def test_coordinate_grid_block_assigns_groups():
    df = pd.DataFrame({"lat": [44.1, 44.2, 45.5], "lon": [-73.1, -73.05, -72.0]})
    out = coordinate_grid_block(df, size_deg=1.0)
    assert "grid_block" in out.columns
    assert out["grid_block"].iloc[0] == out["grid_block"].iloc[1]
    assert out["grid_block"].iloc[0] != out["grid_block"].iloc[2]
