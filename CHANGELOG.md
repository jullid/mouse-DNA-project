# Changelog

All notable changes to this project are documented here. Entries are in reverse chronological order.

---

## [0.5.0] — cfDNA Blood-Dominant Benchmark & Barplot Port

### Added
- `deconv_regimes.py` — cfDNA benchmark regime module
  - `REGIMES` dict — catalogue of all benchmark configs (baseline + easy/medium/hard/healthy)
  - `resolve_regime_names()` — expands `all`/`suite` shorthands, validates names
  - `generate_blood_dominant_mixtures()` — blood-dominant mixture generator; blood tissue always forced in, non-blood proportions via Dirichlet(α=0.5)
  - `compute_cfdna_metrics()` — blood MAE + non-blood recall (ε=0.02) per regime
  - `run_regime()` — runs steps 6–10 for one named regime; returns summary metrics
- `deconv_model.py`
  - `select_random_k_mixtures()` — selects mixture IDs by component count for barplot sampling
  - `plot_selected_mixture_barplots()` — grouped bar chart: true vs NNLS-estimated proportions for a selected subset of mixtures (ported from `notebooks/deconvolution_explore.ipynb`)
- `config.py` — cfDNA benchmark parameters: `DECONV_CFDNA_BLOOD_TISSUE`, `DECONV_CFDNA_K_NB_MIN/MAX`, `DECONV_CFDNA_NONBLOOD_ALPHA`, band-edge constants for each regime, `DECONV_MIXTURES_FIG_DIR`; barplot parameters: `DECONV_BARPLOT_K`, `DECONV_BARPLOT_N_PLOTS`, `DECONV_BARPLOT_SEED`
- `PLAN.md` — project-root planning log with dated entries for each significant planning round

### Changed
- `run_deconvolution.py` — refactored into `_run_shared_setup()`, `_run_legacy_balanced()`, `_run_regimes()`. Added `--regimes` CLI flag (see Usage). No-flag behaviour is identical to before.
- `deconv_model.py` — `select_random_k_mixtures()` extended with optional `k_max` parameter for range-based filtering (backwards compatible; existing calls use exact-k mode via `k_min` only).

### Design Decisions
- **Legacy path is unchanged:** running without `--regimes` executes the balanced benchmark at the existing flat figure paths. No files are moved or overwritten.
- **Regime outputs are isolated:** `--regimes` writes to `figures/deconvolution/mixtures/<regime>/` with the same 7-figure set per regime. A cross-regime summary table (median MAE, mean Pearson r, blood MAE, non-blood recall) is printed after all regimes complete.
- **Blood tissue proxy note:** `Blood_Spleen_Thymus` merges bulk blood, spleen, and thymus. Real cfDNA is granulocyte/lymphocyte-dominated; results are feasibility estimates, not quantitative cfDNA predictions.
- **Per-regime seed offsets** (0 / 1000 / 2000 / 3000 / 4000) ensure each regime is independently reproducible regardless of which other regimes are requested.

---

## [0.4.0] — Tissue Deconvolution Pipeline & Utils Refactor

### Added
- `utils.py` — shared utility functions extracted from `classifier_data.py` for reuse across pipelines
  - `build_label_map()` — derives flat label mapping from `MERGE_GROUPS`
  - `extract_sample_labels()` — parses tissue labels from df_raw column names
  - `apply_label_map()` — maps original labels to merged labels, removes excluded tissues
  - `build_centroids()` — computes merged tissue centroids from arbitrary sample subsets (renamed from `build_fold_centroids` for generality)
  - `select_probes_from_centroids()` — runs full diff scoring pipeline on a centroid matrix
  - `build_X_from_probes()` — masks df_raw to selected probes and transposes
- `deconv_data.py` — deconvolution data preparation module
  - `split_reference_pool()` — stratified split of replicates into reference and pool sets
  - `save_split()` / `load_split()` — persist and load split as JSON for reproducibility
  - `get_pool_tissue_counts()` — count pool replicates per merged tissue
  - `get_eligible_tissues()` — identify tissues with enough pool replicates for mixture generation
  - `build_reference_matrix()` — build signature matrix W from reference replicates, masked to probes
  - `validate_reference()` — condition number, NaN/Inf checks, per-tissue summary
  - `correlate_with_full_atlas()` — per-tissue Pearson correlation between reference and full-atlas centroids
  - `generate_synthetic_mixtures()` — generate N synthetic mixtures from pool replicates using Dirichlet proportions
  - `validate_mixtures()` — beta range, proportion sum, NaN/Inf, and distribution checks
  - `load_deconv_data()` — convenience loader for the full deconvolution data prep sequence
- `deconv_model.py` — deconvolution modelling module
  - `deconvolve_single()` — NNLS solver for a single mixture vector (designed for future single-sample inference)
  - `deconvolve_batch()` — NNLS solver for batch of synthetic mixtures
  - `compute_per_mixture_metrics()` — MAE, RMSE, Pearson r per mixture
  - `compute_per_tissue_metrics()` — mean MAE, Pearson r, false positive rate per tissue
  - `print_evaluation_summary()` — concise performance summary
  - `plot_reference_heatmap()` — heatmap of signature matrix W
  - `plot_reference_atlas_correlation()` — bar chart of per-tissue reference vs full-atlas correlation
  - `plot_proportion_distributions()` — histogram of non-zero mixture proportions
  - `plot_mixture_pca()` — PCA projection of reference centroids and synthetic mixtures
  - `plot_true_vs_estimated_scatter()` — per-tissue scatter panels (true vs estimated proportions)
  - `plot_mae_by_components()` — MAE box plot stratified by number of mixture components
  - `plot_residual_distribution()` — NNLS residual histogram
  - `plot_per_tissue_mae()` — per-tissue MAE bar chart
- `run_deconvolution.py` — orchestration script for the full deconvolution benchmarking pipeline
- `__init__.py` — makes `src/` an importable Python package
- Deconvolution-specific config parameters in `config.py`
- `README.md`, `USAGE.md`, `CHANGELOG.md` — project documentation

### Changed
- `classifier_data.py` — refactored to import shared functions from `utils.py` instead of defining them locally. Re-exports them under original names for backward compatibility (e.g. `build_fold_centroids` is an alias for `utils.build_centroids`). No change in behaviour or public interface.
- `config.py` — added deconvolution parameters (`DECONV_REFERENCE_FRACTION`, `DECONV_SPLIT_SEED`, `DECONV_MIN_POOL_REPLICATES`, `DECONV_N_MIXTURES`, `DECONV_K_MIN`, `DECONV_K_MAX`, `DECONV_DIRICHLET_ALPHA`, `DECONV_MIXTURE_SEED`, `DECONV_ATLAS_PROBES_PATH`, `DECONV_OUTPUT_DIR`, `DECONV_FIGURES_DIR`)

### Design Decisions
- **No cross-pipeline imports:** `deconv_data.py` imports from `utils.py`, not from `classifier_data.py`. The two pipelines are fully independent.
- **Probe selection runs once per pipeline execution** on the reference centroids. All 1000 mixtures use the same probe set. Changing the split ratio triggers a full re-run (new centroids → new probes → new mixtures).
- **Tissues with < 3 pool replicates** are excluded from mixture generation but remain in the reference atlas.
- **`deconvolve_single()` is designed for future single-sample inference** — takes a single beta vector and returns proportions + residual.
- **Split is saved to JSON** (`results/deconvolution/split.json`) with metadata for full reproducibility.

---

## [0.3.0] — Leakage-Free Tissue Classifier

### Added
- `classifier_data.py` — classifier data preparation module
  - `split_train_test()` — holds out 1 sample per original tissue as fixed test set
  - `load_classifier_data()` — convenience loader for the full data preparation sequence
- `classifier_model.py` — classifier modelling module
  - `check_cv_feasibility()` — auto-reduces fold count if smallest class is too small
  - `build_cv()` — creates StratifiedKFold cross-validator
  - `verify_fold_distributions()` — prints per-fold sample counts for stratification sanity check
  - `run_cv_with_probe_selection()` — leakage-free CV loop with per-fold probe selection
  - `final_evaluation()` — fits on all training data, evaluates on held-out test set
  - `report_per_class_metrics()` — prints full sklearn classification report
  - `plot_confusion_matrix()` — row-normalised confusion matrix plot
  - `plot_per_class_recall()` — horizontal bar chart with sample count annotations
  - `print_summary()` — end-of-run summary of CV and test results
- `run_classifier.py` — classifier orchestration script with main() guard

### Changed
- `data_loading.py` — exposed `MERGE_GROUPS` and `EXCLUDED_TISSUES` as module-level constants (previously hardcoded inside `build_merged_df`). No change in behaviour.

### Fixed
- Removed dead `label_map` parameter from `build_fold_centroids()` function signature and all call sites.

---

## [0.2.0] — HPC Compatibility & Figure Output

### Changed
- `visualization.py` — added `matplotlib.use("Agg")` for headless HPC rendering
- `visualization.py` — replaced all `plt.show()` with `_save_or_close()` helper
- `visualization.py` — `plot_clustered_tissue_correlation` uses `g.savefig()` for ClusterGrid
- `visualization.py` — all plot functions accept `output_path` and `dpi` parameters
- `visualization.py` — wrapped PCA into `plot_pca_validation()` function
- `run_pipeline.py` — all plot calls pass `output_path` with auto-generated filenames

### Added
- `config.py` — added `FIGURES_DIR`, `FIGURE_FORMAT`, `FIGURE_DPI`

---

## [0.1.0] — Initial Modularisation

### Added
- `config.py` — centralised paths and pipeline parameters
- `data_loading.py` — CSV loading, annotation filtering, celltype averaging, tissue merging
- `diff_analysis.py` — differential methylation scoring, probe selection, `create_heatmap_matrix()`
- `visualization.py` — all plot functions
- `annotation.py` — probe annotation join and Excel export
- `run_pipeline.py` — orchestration script with `main()` guard

### Changed
- Refactored monolithic Jupyter notebook into modular Python scripts
- Fixed deprecated `groupby(axis=1)` → `.T.groupby().mean().T` in `build_celltype_df()`
- Replaced fragile `getattr(row, col.replace(" ", "_"))` with `row._asdict()[col]` in Excel export

---

## [0.0.0] — Preprocessing

### Added
- `scripts/preprocessing.py` — one-shot column simplification script. Run once, not part of repeatable pipeline.
