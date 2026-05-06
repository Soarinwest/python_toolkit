# Geo Daily Tools

A small Python project for reusable geospatial/data-science daily utilities: initial dataset inspection, pandas validation, geospatial QA/QC, model-ready feature preparation, and interview-style practice scripts.

## Quick start

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -e .[dev,io]
```

Run the examples:

```bash
python examples/01_clean_sensor_records.py
python examples/02_geodataframe_workflow.py
python examples/03_modeling_prep_workflow.py
```

Run tests:

```bash
pytest
```

## Project layout

```text
geo-daily-tools/
├─ pyproject.toml
├─ README.md
├─ examples/
│  ├─ 01_clean_sensor_records.py
│  ├─ 02_geodataframe_workflow.py
│  └─ 03_modeling_prep_workflow.py
├─ src/
│  └─ geo_daily_tools/
│     ├─ __init__.py
│     ├─ inspection.py
│     ├─ validation.py
│     ├─ geo_validation.py
│     ├─ modeling.py
│     ├─ io_utils.py
│     └─ sample_data.py
└─ tests/
   ├─ test_validation.py
   ├─ test_geo_validation.py
   └─ test_modeling.py
```

## Interview/coding patterns this project is built around

### 1. Initial inspection

Use `quick_inspect(df)` to print shape, columns, dtypes, missingness, numeric summaries, and categorical summaries.

### 2. Drop-reason validation

Use `validate_sensor_records(records)` to create a cleaned dataframe and a drop summary. The function uses a `drop_reason` column so invalid rows are auditable.

### 3. GeoDataFrame creation

Use `points_from_latlon(df)` to create an EPSG:4326 GeoDataFrame from lat/lon columns.

### 4. Spatial modeling prep

Use `prepare_model_inputs(df, features, target)` and `group_train_test_split(...)` to avoid accidental spatial leakage.

## Notes

This is intentionally simple and readable. It is built for reusable daily work and live-coding practice, not as a heavyweight framework.
