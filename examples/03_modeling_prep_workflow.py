"""Example: model-ready features and group-aware train/test split."""

from geo_daily_tools.modeling import (
    feature_missingness_report,
    group_train_test_split,
    prepare_model_inputs,
)
from geo_daily_tools.sample_data import messy_sensor_records
from geo_daily_tools.validation import validate_sensor_records


def main() -> None:
    valid_df, _ = validate_sensor_records(messy_sensor_records())

    feature_cols = ["lat", "lon"]
    target_col = "reading"

    print("Feature missingness:")
    print(feature_missingness_report(valid_df, feature_cols))

    X, y, model_df = prepare_model_inputs(valid_df, feature_cols, target_col)
    print("\nX:")
    print(X)
    print("\ny:")
    print(y)

    train_df, test_df = group_train_test_split(model_df, group_col="tile_id", test_size=0.5)
    print("\nTrain tile IDs:", train_df["tile_id"].unique().tolist())
    print("Test tile IDs:", test_df["tile_id"].unique().tolist())


if __name__ == "__main__":
    main()
