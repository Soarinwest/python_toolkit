"""Build the demo notebooks. Run once: `python notebooks/_build_notebooks.py`."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT_DIR = Path(__file__).parent


def md(text: str) -> nbf.notebooknode.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.notebooknode.NotebookNode:
    return nbf.v4.new_code_cell(text)


def write(name: str, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    path = OUT_DIR / name
    nbf.write(nb, path)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# 01 Inspect a messy dataset
# ---------------------------------------------------------------------------

n01 = [
    md(
        "# 01 — Inspect a messy dataset\n\n"
        "First-pass helpers for any new DataFrame: per-column overview, dtype counts, "
        "missingness, categorical summary, grouped numeric summary, and a one-shot inspector."
    ),
    code(
        "import pandas as pd\n\n"
        "from geo_daily_tools.sample_data import messy_sensor_records\n"
        "from geo_daily_tools.inspection import (\n"
        "    column_overview,\n"
        "    dtype_summary,\n"
        "    missingness_report,\n"
        "    categorical_summary,\n"
        "    grouped_numeric_summary,\n"
        "    quick_inspect,\n"
        ")\n\n"
        "df = pd.DataFrame(messy_sensor_records())\n"
        "df"
    ),
    md("## Per-column overview — dtype, missing, unique, sample value"),
    code("column_overview(df)"),
    md("## Dtype distribution"),
    code("dtype_summary(df)"),
    md("## Missingness ranking"),
    code("missingness_report(df)"),
    md("## Categoricals at a glance"),
    code("categorical_summary(df)"),
    md("## Numeric summary by group"),
    code("grouped_numeric_summary(df, group_cols='sensor_type', value_cols='reading')"),
    md("## One-shot inspection (prints everything)"),
    code("_ = quick_inspect(df, name='messy_sensor_records')"),
]

# ---------------------------------------------------------------------------
# 02 Clean with drop reasons
# ---------------------------------------------------------------------------

n02 = [
    md(
        "# 02 — Clean with `drop_reason`\n\n"
        "Auditable cleaning: every dropped row gets a single explicit reason, and the function "
        "returns both the cleaned data and a summary of what was thrown away."
    ),
    code(
        "import pandas as pd\n\n"
        "from geo_daily_tools.sample_data import messy_sensor_records\n"
        "from geo_daily_tools.validation import (\n"
        "    strip_string_columns,\n"
        "    valid_lat_lon_mask,\n"
        "    validate_sensor_records,\n"
        "    flag_outliers_iqr,\n"
        "    range_check,\n"
        "    value_in_set_check,\n"
        ")\n\n"
        "df = pd.DataFrame(messy_sensor_records())\n"
        "df"
    ),
    md("## Strip whitespace and convert blanks to NA"),
    code("strip_string_columns(df, ['obs_id'])[['obs_id']]"),
    md("## Filter to rows with valid lat/lon"),
    code("df.loc[valid_lat_lon_mask(df)]"),
    md(
        "## End-to-end QA: `validate_sensor_records`\n\n"
        "Returns valid rows, invalid rows (with `drop_reason`), and a summary dict."
    ),
    code(
        "valid_df, invalid_df, summary = validate_sensor_records(\n"
        "    messy_sensor_records(), return_invalid=True\n"
        ")\n"
        "summary"
    ),
    code("valid_df"),
    code("invalid_df[['obs_id', 'lat', 'lon', 'reading', 'drop_reason']]"),
    md("## Range and allowed-value checks"),
    code(
        "range_check(\n"
        "    valid_df,\n"
        "    {'lat': (-90, 90), 'lon': (-180, 180), 'reading': (0, 100)},\n"
        ")"
    ),
    code("value_in_set_check(valid_df, {'sensor_type': {'type_a', 'type_b', 'type_c'}})"),
    md("## Flag IQR outliers, optionally per group"),
    code(
        "flagged = valid_df.copy()\n"
        "flagged['is_outlier'] = flag_outliers_iqr(flagged, 'reading', group_cols='sensor_type')\n"
        "flagged[['obs_id', 'sensor_type', 'reading', 'is_outlier']]"
    ),
]

# ---------------------------------------------------------------------------
# 03 GeoDataFrame QA
# ---------------------------------------------------------------------------

n03 = [
    md(
        "# 03 — GeoDataFrame QA\n\n"
        "Build a GeoDataFrame from lat/lon, run geometry checks, reproject before metric ops, "
        "and join points to nearest neighbors with a real distance column."
    ),
    code(
        "import geopandas as gpd\n\n"
        "from geo_daily_tools.sample_data import messy_sensor_records\n"
        "from geo_daily_tools.validation import validate_sensor_records\n"
        "from geo_daily_tools.geo_validation import (\n"
        "    points_from_latlon,\n"
        "    geometry_quality_report,\n"
        "    geometry_type_summary,\n"
        "    bounds_summary,\n"
        "    require_crs,\n"
        "    reproject_if_needed,\n"
        "    nearest_neighbor_join,\n"
        ")\n\n"
        "valid_df, _ = validate_sensor_records(messy_sensor_records())\n"
        "gdf = points_from_latlon(valid_df)\n"
        "gdf.head()"
    ),
    md("## Geometry QA"),
    code("geometry_quality_report(gdf)"),
    code("geometry_type_summary(gdf)"),
    code("bounds_summary(gdf)"),
    md(
        "## CRS hygiene\n\n"
        "EPSG:4326 is fine for *storing* lat/lon. Always reproject to a projected CRS "
        "(e.g. EPSG:3857 or a local UTM zone) before computing distance or area."
    ),
    code(
        "require_crs(gdf, expected_crs='EPSG:4326')\n"
        "gdf_m = reproject_if_needed(gdf, 'EPSG:3857')\n"
        "gdf_m.crs"
    ),
    md("## Nearest-neighbor join with a meaningful distance column"),
    code(
        "target = gpd.GeoDataFrame(\n"
        "    {'name': ['centroid']},\n"
        "    geometry=gpd.points_from_xy([gdf_m.geometry.x.mean()], [gdf_m.geometry.y.mean()]),\n"
        "    crs=gdf_m.crs,\n"
        ")\n"
        "joined = nearest_neighbor_join(gdf_m, target)\n"
        "joined[['obs_id', 'name', 'distance']]"
    ),
]

# ---------------------------------------------------------------------------
# 04 Modeling prep & spatial split
# ---------------------------------------------------------------------------

n04 = [
    md(
        "# 04 — Modeling prep & spatial split\n\n"
        "Build X/y, make a spatial-block group when no natural one exists, do a group-aware "
        "split, and sanity-check for leakage and distribution drift."
    ),
    code(
        "from geo_daily_tools.sample_data import messy_sensor_records\n"
        "from geo_daily_tools.validation import validate_sensor_records\n"
        "from geo_daily_tools.modeling import (\n"
        "    feature_missingness_report,\n"
        "    prepare_model_inputs,\n"
        "    coordinate_grid_block,\n"
        "    group_train_test_split,\n"
        "    check_group_leakage,\n"
        "    compare_distributions,\n"
        "    simple_group_holdout,\n"
        ")\n\n"
        "valid_df, _ = validate_sensor_records(messy_sensor_records())\n"
        "valid_df"
    ),
    md("## Feature missingness"),
    code("feature_missingness_report(valid_df, ['lat', 'lon'])"),
    md("## Build X / y"),
    code("X, y, model_df = prepare_model_inputs(valid_df, ['lat', 'lon'], 'reading')\nX"),
    md(
        "## Build a spatial-block group when no natural one exists\n\n"
        "Floor-buckets lat/lon into a coarse grid you can use as `group_col` for spatial CV."
    ),
    code(
        "blocked = coordinate_grid_block(valid_df, size_deg=0.05)\n"
        "blocked[['obs_id', 'lat', 'lon', 'grid_block']]"
    ),
    md(
        "## Group-aware train/test split\n\n"
        "Random row splits leak spatial structure between train and test. Splitting by "
        "tile/site/scene gives an honest validation set."
    ),
    code(
        "train, test = group_train_test_split(valid_df, group_col='tile_id', test_size=0.5)\n"
        "check_group_leakage(train, test, 'tile_id')"
    ),
    md("## Distribution sanity check"),
    code("compare_distributions(train, test, ['lat', 'lon', 'reading'])"),
    md("## Hold out specific groups by name"),
    code(
        "train2, test2 = simple_group_holdout(valid_df, 'tile_id', ['tile_001'])\n"
        "print('train tiles:', train2['tile_id'].unique().tolist())\n"
        "print('test tiles: ', test2['tile_id'].unique().tolist())"
    ),
]


def main() -> None:
    write("01_inspect_a_messy_dataset.ipynb", n01)
    write("02_clean_with_drop_reasons.ipynb", n02)
    write("03_geodataframe_qa.ipynb", n03)
    write("04_modeling_prep_and_split.ipynb", n04)


if __name__ == "__main__":
    main()
