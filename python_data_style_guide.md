# Style guide for Python / data prompts

How to handle Python and data work — pandas, GeoPandas, PostGIS, R, cloud-based geospatial pipelines, mixed-language workflows. Treat me as fluent — I want you tuned for that level, not for explaining what a DataFrame is.

These are principles, not strict rules. Use judgment.

---

## Framing the work

- **State assumptions before writing code.** Column names, dtypes, join cardinality, CRS — if you're guessing, say so. A one-line "assuming `id` is unique in the left table and many-to-one in the right" prevents a lot of wasted iteration.
- **Match the scale.** A 200-row exploration doesn't need a class hierarchy. A long-running ETL probably does need caching and restartability. Don't reach for either reflexively.
- **Show, don't lecture.** If you're explaining a concept I might not know, a four-line code example beats three paragraphs of prose.

## Data inspection

- **Inspect before transforming.** For any unfamiliar input: `shape`, `dtypes`, `head()`, null counts, `value_counts()` on a few categoricals. Don't assume schema from column names.
- **Flag, don't drop.** Surface bad rows with a reason column. Silent `.dropna()` is almost never what I want. If you're filtering, be loud about it and count what went.
- **Print row counts after joins.** Joins are where data quietly multiplies or vanishes. Show before/after.
- **`value_counts(dropna=False)`** — nulls are signal too.

## Geospatial

- **CRS is explicit, always.** Set it on construction, name it in variable names where ambiguous (`gdf_4326`, `gdf_utm`), and reproject before any metric (area, length, distance, buffer) operation. Never compute meters in degrees.
- **`estimate_utm_crs()` is fine for ad-hoc work**, but pick a stable projected CRS when the analysis will be revisited. Ask if it matters.
- **Geometry hygiene.** Check `is_valid` on incoming polygons. Use `make_valid`, not `buffer(0)`. Handle empties and nulls explicitly.
- **Spatial joins** — be intentional about the predicate (`within` vs `intersects` vs `contains`) and the index side. `sjoin_nearest` has a `max_distance` parameter; use it.
- **Raster ↔ point boundary.** Prefer server-side reductions (GEE `reduceRegions`, rasterio with windowed reads) over per-point loops. Chunk large extractions; don't pull everything into memory.
- **Coordinate precision and provenance matter.** When merging datasets with different positional accuracy, that mismatch is often the whole story — surface it.

## pandas

- **Method chaining is fine**, but break the chain when it stops fitting on screen or when a step deserves a comment.
- **`assign` over assignment** inside a pipe-style flow.
- **Avoid `inplace=True`.** It complicates reasoning and rarely saves what people think it saves.
- **`merge` with `validate=`** when the cardinality is known. Catches schema drift cheap.
- **Don't reach for `apply` reflexively.** Vectorize first; if `apply` is genuinely needed, say why.

## R / Python interop

- **Pick a paradigm per script.** Don't sprinkle base-R through tidyverse code, and don't mix pandas with raw loops where vectorization works. Be consistent within a script.
- **Don't reflexively port between languages.** Ask what the destination is — sometimes R is the right tool, sometimes Python, sometimes a mix is fine.
- **When generating R**, default to tidyverse + sf, and use `|>` (native pipe) over `%>%` unless I've shown otherwise.

## Databases & queries

- **Parameterized queries only.** SQLAlchemy `text(...)` with bound params, never f-strings into SQL. Same rule for any DBAPI.
- **PostGIS specifics** — GIST indexes on geometry columns, `geography` cast for true-distance/area where reprojecting isn't worth it, `ST_DWithin` over `ST_Distance < x` for index-aware proximity.
- **Idempotent writes.** Truncate-and-reload or upsert by key — defend the choice in a comment. Don't leave append-only writes that silently double on rerun.

## Long-running work

- **Cache the expensive step.** API pulls, raster extractions, simulations — write intermediate Parquet/Feather/RDS and check for it on rerun.
- **Make it restartable.** Long jobs should resume from the middle, not be all-or-nothing.
- **Progress visibility.** `tqdm` for loops, periodic log lines for chunked work. Not for 30-second jobs.

## Reproducibility

- **Set `random_state` / seed everywhere it matters** — sklearn splits, sampling, Monte Carlo work, anything stochastic.
- **`pathlib.Path`, not string concatenation.** Use a project-root anchor (`ROOT = Path(__file__).resolve().parent`).
- **Don't overwrite raw inputs.** Outputs go in a sibling directory. If versioning matters, datestamp or git-tag — don't rely on filenames alone.

## Code form

- **Type hints on function signatures** for anything I'd reuse. Skip them inside one-off notebooks.
- **Docstrings on non-trivial functions** — one-liner is fine, but say what the function assumes about inputs (CRS, schema, sort order).
- **Comments explain *why*.** "`# convert to meters before buffer`" earns its keep. "`# loop over rows`" does not.
- **Pure functions where you can.** Side effects (writes, plots, prints) belong at the edges of a script, not in the middle of a transformation.

## Visualization

- **Always label axes with units.** Title says what, subtitle says where/when if relevant.
- **Default to matplotlib for static, plotly for interactive, folium/leaflet for maps. ggplot2 in R.**
- **For figures I'll show stakeholders**: clean and minimal. **For figures I'll use to debug**: dense and informative. Ask which if it's not obvious.
- **Consistent palette across related figures** in the same analysis.

## ML / stats

- **Spatial autocorrelation is the default assumption** for spatial data. Random train/test splits are usually wrong; region-, tile-, or buffer-based holdouts are the starting point.
- **Temporal autocorrelation gets the same treatment** for time series — no random shuffling across time.
- **Uncertainty propagation matters.** When aggregating simulation runs or comparing distributions, don't collapse to point estimates without showing the spread.
- **Class and sampling imbalance is real** — coverage is rarely uniform across groups. Flag it.

## When in doubt

- **Ask before guessing big things.** Column names I can probably resolve, but design decisions are mine to make.
- **Surface tradeoffs, don't bury them.** If there are two reasonable approaches with different failure modes, say so — don't silently pick the conservative one.
- **Pseudocode is a smell.** If you're sketching an algorithm I'll implement, say "sketch." If you're handing me code, it should run.