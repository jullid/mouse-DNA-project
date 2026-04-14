# Project planning log

Most recent first. Each entry captures the context, decision, and scope of a planning round.

---

## 2026-04-13 — cfDNA blood-dominant benchmark extension

**Context**: The existing balanced benchmark uses Dirichlet(α=1.0) over all eligible tissues, producing mixtures without any blood dominance. Real cfDNA from blood draws is dominated by hematopoietic DNA (55–95 % in humans). A staged difficulty benchmark was needed to probe the blood-dominant regime while keeping the balanced benchmark intact.

**Decision**: Added four cfDNA-like regimes (easy/medium/hard/healthy) with non-overlapping-ish blood fraction bands, fixed k_nb=1–3 non-blood tissues, and Dirichlet(α=0.5) for non-blood sparsity. Blood tissue is always forced in. Implemented as an additive extension (no changes to existing balanced pipeline or its output paths).

| Regime        | Blood fraction | k_nonblood |
|---------------|----------------|------------|
| cfdna_easy    | 40–60 %        | 1–3        |
| cfdna_medium  | 60–80 %        | 1–3        |
| cfdna_hard    | 80–95 %        | 1–3        |
| cfdna_healthy | 90–99 %        | 1–2        |

**Scope**:
- `src/config.py` — cfDNA band constants, `DECONV_MIXTURES_FIG_DIR`
- `src/deconv_regimes.py` — new module: `REGIMES`, `generate_blood_dominant_mixtures`, `run_regime`, `compute_cfdna_metrics`, `resolve_regime_names`
- `src/deconv_model.py` — `select_random_k_mixtures` extended with `k_max` parameter
- `src/run_deconvolution.py` — `--regimes` CLI flag; legacy no-flag path unchanged

**Output layout** (regime mode only): `figures/deconvolution/mixtures/{balanced,easy,medium,hard,healthy}/` with the same 7-figure set per regime. Legacy balanced outputs at existing flat paths are untouched.

**CLI**:
```
python run_deconvolution.py                      # legacy (unchanged)
python run_deconvolution.py --regimes cfdna_easy # single regime
python run_deconvolution.py --regimes all        # easy+medium+hard
python run_deconvolution.py --regimes suite      # baseline+all cfDNA
```

**Metrics per regime**: median MAE, mean Pearson r, blood MAE, non-blood recall (threshold ε=0.02).

**Open / deferred**: per-regime results/ artefacts (CSV), adding technical noise model, restructuring `mixtures/` → `regimes/` or `benchmarks/` when non-mixture artefacts accumulate.

---

## 2026-04-13 — Port `plot_selected_mixture_barplots` into src/

**Context**: Notebook `notebooks/deconvolution_explore.ipynb` contained a stable grouped-bar plotting function comparing ground-truth vs. NNLS-estimated tissue proportions for a selected subset of synthetic mixtures. Ready to be moved into the main pipeline.

**Decision**: Added two functions to `src/deconv_model.py`:
- `select_random_k_mixtures(df_proportions, k_min, n_plots, k_max, random_seed)` — selects mixture IDs by component count (later extended with `k_max` range support for cfDNA regimes)
- `plot_selected_mixture_barplots(df_proportions, df_estimated, mixture_ids, title, output_path, dpi)` — faithful port of notebook logic; `plt.show()` replaced with `_save_or_close()`

**Scope**:
- `src/config.py` — `DECONV_BARPLOT_K`, `DECONV_BARPLOT_N_PLOTS`, `DECONV_BARPLOT_SEED`
- `src/deconv_model.py` — two new functions appended after `plot_per_tissue_mae`
- `src/run_deconvolution.py` — Step 10.5 block after existing eval plots

**Output**: `figures/deconvolution/mixtures/selected_barplots.{fmt}`
