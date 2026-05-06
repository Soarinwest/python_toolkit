from geo_daily_tools.modeling import group_train_test_split, prepare_model_inputs
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
