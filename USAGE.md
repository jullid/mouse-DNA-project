# Usage Guide

All scripts are in `src/` and should be run from that directory.

## Prerequisites

```bash
cd ~/projects/mouse-DNA-project
source .venv/bin/activate
cd src
```

Verify data files exist:

```bash
python -c "import config; print('CSV:', config.CSV_FILE.exists()); print('Annot:', config.ANNOT_FILE.exists())"
```

---

## Pipeline 1 — Methylation Atlas

**Script:** `run_pipeline.py`

**What it does:**
1. Loads raw beta-value matrix and annotation file
2. Filters probes without genomic coordinates
3. Averages replicates into tissue centroids (29 → 20 merged tissues)
4. Computes differential methylation scores per tissue
5. Selects top tissue-specific marker probes (unique, capped at MAX_PER_TISSUE)
6. Generates heatmap, correlation, dendrogram, and clustered correlation plots
7. Joins selected probes with genomic annotation metadata
8. Exports annotated markers to Excel

```bash
python run_pipeline.py
```

**Outputs:** `figures/heatmap.{fmt}`, `figures/tissue_correlation.{fmt}`, `figures/dendrogram.{fmt}`, `figures/clustered_tissue_correlation.{fmt}`, `results/mouse_methylation_markers.xlsx`

**Key parameters:** `TOP_N`, `MAX_PER_TISSUE`, `REGION_MODE`, `USE_FILTERING` (all in `config.py`)

**Note:** To save the heatmap matrix for downstream use, uncomment Step 11 at the bottom of `run_pipeline.py`. This produces `data/processed/top_50_regions_df.csv`, which the deconvolution pipeline uses as its default probe panel.

---

## Pipeline 2 — Tissue Classifier

**Script:** `run_classifier.py`

**What it does:**
1. Loads raw replicate matrix, extracts tissue labels, applies merging
2. Holds out 1 sample per original tissue type as a fixed test set (26 test samples)
3. Runs stratified k-fold CV with probe selection inside each fold (no leakage)
4. Reports CV metrics (balanced accuracy, macro F1, per-fold probe counts)
5. Fits final model on all training data, evaluates on held-out test set
6. Generates confusion matrices and per-class recall plots for both CV and test

```bash
python run_classifier.py
```

**Outputs:** `figures/classifier/cv_confusion_matrix.{fmt}`, `figures/classifier/cv_per_class_recall.{fmt}`, `figures/classifier/test_confusion_matrix.{fmt}`, `figures/classifier/test_per_class_recall.{fmt}`

**Key parameters:** `N_FOLDS`, `RANDOM_SEED`, `C` (in `run_classifier.py`)

---

## Pipeline 3 — Tissue Deconvolution

**Script:** `run_deconvolution.py`

**Prerequisite:** The full-atlas probe panel must exist at `data/processed/top_50_regions_df.csv`. Run Pipeline 1 with Step 11 uncommented first to produce it (used only for the reference-vs-atlas correlation check; the deconvolution runs its own probe selection on the reference split).

**What it does:**
1. Loads raw replicate matrix, excludes dropped tissues
2. Splits replicates into reference set (50%) and mixture pool (50%), stratified by tissue
3. Saves the split to `results/deconvolution/split.json`
4. Builds tissue centroids from reference replicates only
5. Runs probe selection on the reference centroids (once, not per mixture)
6. Builds signature matrix W, validates it (condition number, correlation with full atlas)
7. Determines which tissues have enough pool replicates (>= 3) for mixture generation
8. Generates 1000 synthetic mixtures from pool replicates (2–4 tissues, Dirichlet proportions)
9. Validates mixtures (beta range, proportion sums, distribution checks, PCA projection)
10. Runs NNLS deconvolution for each mixture against W
11. Evaluates: per-mixture MAE/RMSE/correlation, per-tissue MAE and false positive rates
12. Generates evaluation plots (scatter per tissue, MAE by components, residuals, per-tissue MAE)

```bash
python run_deconvolution.py
```

**Outputs:**

| Output | Location |
|--------|----------|
| Reference/pool split | `results/deconvolution/split.json` |
| Reference heatmap | `figures/deconvolution/reference/reference_heatmap.{fmt}` |
| Reference vs atlas correlation | `figures/deconvolution/reference/reference_atlas_corr.{fmt}` |
| Proportion distributions | `figures/deconvolution/mixtures/proportion_dist.{fmt}` |
| Mixture PCA projection | `figures/deconvolution/mixtures/mixture_pca.{fmt}` |
| True vs estimated scatter | `figures/deconvolution/eval_scatter_per_tissue.{fmt}` |
| MAE by component count | `figures/deconvolution/eval_mae_by_k.{fmt}` |
| Residual histogram | `figures/deconvolution/eval_residuals.{fmt}` |
| Per-tissue MAE | `figures/deconvolution/eval_per_tissue_mae.{fmt}` |

**Key parameters** (all in `config.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DECONV_REFERENCE_FRACTION` | 0.5 | Fraction of replicates per tissue → reference |
| `DECONV_SPLIT_SEED` | 42 | Split reproducibility seed |
| `DECONV_MIN_POOL_REPLICATES` | 3 | Min pool replicates for mixture eligibility |
| `DECONV_N_MIXTURES` | 1000 | Number of synthetic mixtures |
| `DECONV_K_MIN` | 2 | Min tissues per mixture |
| `DECONV_K_MAX` | 4 | Max tissues per mixture |
| `DECONV_DIRICHLET_ALPHA` | 1.0 | Dirichlet concentration (1.0 = uniform) |
| `DECONV_MIXTURE_SEED` | 123 | Mixture generation seed |

**Changing the split:** Edit `DECONV_REFERENCE_FRACTION` or `DECONV_SPLIT_SEED` in `config.py` and re-run `run_deconvolution.py`. The entire pipeline re-executes: new split → new centroids → new probe selection → new mixtures → new evaluation. The split JSON is overwritten with the new configuration.

---

## Running from Notebooks

To import pipeline modules from a notebook in `notebooks/`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("../src")))

import data_loading
import config
```

---

## Quick Data Verification

```bash
python data_loading.py
```

Prints shapes and column names for df_raw, df_celltype, and df_merged.

---

## HPC Notes

- All plots use `matplotlib.use("Agg")` — no display server required.
- Figures are saved to disk and closed immediately to free memory.
- Figure format configurable: `"png"` for quick checks, `"pdf"` for publication quality.
