"""
Deconvolution benchmarking pipeline.

Splits replicates into a reference set and mixture pool, builds a
leakage-safe signature matrix, generates synthetic mixtures, runs
NNLS deconvolution, and evaluates performance.

The full atlas pipeline (run_pipeline.py) is NOT modified by this script.

Usage
-----
# Legacy: balanced baseline at current paths (unchanged behaviour)
python run_deconvolution.py

# Single cfDNA regime written to figures/deconvolution/mixtures/<subdir>/
python run_deconvolution.py --regimes cfdna_easy

# Multiple explicit regimes
python run_deconvolution.py --regimes cfdna_easy cfdna_hard

# All cfDNA regimes (easy, medium, hard)
python run_deconvolution.py --regimes all

# Balanced baseline + all cfDNA regimes
python run_deconvolution.py --regimes suite

# Add the optional healthy regime
python run_deconvolution.py --regimes cfdna_healthy
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import config
import data_loading
import diff_analysis
import visualization
import deconv_data
import deconv_model
import deconv_regimes
from utils import build_centroids


# ---------------------------------------------------------------------------
# Shared setup (steps 1–5) — runs once regardless of which regimes are selected
# ---------------------------------------------------------------------------

def _run_shared_setup(figures_dir: Path, fmt: str, dpi: int) -> tuple:
    """
    Run steps 1–5 and return (shared_inputs, common_params_base, W_masked).

    shared_inputs: dict fed to deconv_regimes.run_regime()
    Returns a dict containing both shared_inputs and cond_number for the summary.
    """

    # ---- Config -----------------------------------------------------------
    REFERENCE_FRACTION = config.DECONV_REFERENCE_FRACTION
    SPLIT_SEED         = config.DECONV_SPLIT_SEED
    MIN_POOL_REPS      = config.DECONV_MIN_POOL_REPLICATES
    K_MIN              = config.DECONV_K_MIN
    K_MAX              = config.DECONV_K_MAX
    DIRICHLET_ALPHA    = config.DECONV_DIRICHLET_ALPHA

    # ---- Step 1: Load and split -------------------------------------------
    print("=" * 60)
    print("STEP 1: Load data and split into reference / pool")
    print("=" * 60)

    (
        df_raw,
        reference_cols,
        pool_cols,
        original_labels,
        label_map,
    ) = deconv_data.load_deconv_data(
        reference_fraction=REFERENCE_FRACTION,
        split_seed=SPLIT_SEED,
    )

    # ---- Step 2: Build reference centroids --------------------------------
    print()
    print("=" * 60)
    print("STEP 2: Build reference centroid matrix")
    print("=" * 60)

    df_merged_ref = build_centroids(
        df_raw=df_raw,
        sample_cols=reference_cols,
        original_labels=original_labels,
        merge_groups=data_loading.MERGE_GROUPS,
    )
    print(f"Reference centroid matrix shape: {df_merged_ref.shape}")

    # ---- Step 3: Select probes --------------------------------------------
    print()
    print("=" * 60)
    print("STEP 3: Select probes from reference centroids")
    print("=" * 60)

    all_tissues_results = diff_analysis.build_tissue_diff_table(
        df=df_merged_ref,
        use_filtering=config.USE_FILTERING,
    )

    df_top_regions = diff_analysis.extract_top_regions(
        all_tissues_results,
        top_n=config.TOP_N,
    )

    ref_heatmap_matrix, ref_region_counts = diff_analysis.create_heatmap_matrix(
        df=df_merged_ref,
        df_top_regions=df_top_regions,
        region_mode=config.REGION_MODE,
        max_per_tissue=config.MAX_PER_TISSUE,
        verbose=config.USE_VERBOSE,
    )

    selected_probes = ref_heatmap_matrix.columns.tolist()
    print(f"Probes selected: {len(selected_probes)}")

    # ---- Step 4: Build and validate reference matrix ----------------------
    print()
    print("=" * 60)
    print("STEP 4: Build reference matrix W and validate")
    print("=" * 60)

    W_masked = deconv_data.build_reference_matrix(
        df_raw=df_raw,
        reference_cols=reference_cols,
        original_labels=original_labels,
        selected_probes=selected_probes,
    )

    cond_number = deconv_data.validate_reference(W_masked)

    atlas_path = config.DECONV_ATLAS_PROBES_PATH
    if atlas_path.exists():
        corr_series = deconv_data.correlate_with_full_atlas(W_masked, atlas_path)
    else:
        print(f"Full-atlas probe file not found at {atlas_path}, skipping correlation check.")
        corr_series = pd.Series(dtype=float)

    visualization.plot_heatmap(
        ref_heatmap_matrix,
        ref_region_counts,
        top_n=config.MAX_PER_TISSUE,
        title=f"Reference atlas heatmap ({len(reference_cols)} replicates, "
              f"{ref_heatmap_matrix.shape[1]} probes, reference split only)",
        output_path=figures_dir / "reference" / f"reference_heatmap.{fmt}",
        dpi=dpi,
    )

    if len(corr_series) > 0:
        deconv_model.plot_reference_atlas_correlation(
            corr_series,
            output_path=figures_dir / "reference" / f"reference_atlas_corr.{fmt}",
            dpi=dpi,
        )

    # ---- Step 5: Determine eligible tissues --------------------------------
    print()
    print("=" * 60)
    print("STEP 5: Determine eligible tissues for mixture generation")
    print("=" * 60)

    eligible_tissues = deconv_data.get_eligible_tissues(
        pool_cols=pool_cols,
        original_labels=original_labels,
        label_map=label_map,
        min_pool_replicates=MIN_POOL_REPS,
    )

    shared_inputs = {
        "df_raw":           df_raw,
        "reference_cols":   reference_cols,
        "pool_cols":        pool_cols,
        "original_labels":  original_labels,
        "label_map":        label_map,
        "selected_probes":  selected_probes,
        "eligible_tissues": eligible_tissues,
        "W_masked":         W_masked,
    }

    return shared_inputs, cond_number, K_MIN, K_MAX, DIRICHLET_ALPHA


# ---------------------------------------------------------------------------
# Legacy balanced run (no --regimes flag) — exact existing behaviour
# ---------------------------------------------------------------------------

def _run_legacy_balanced(shared_inputs: dict, figures_dir: Path, fmt: str, dpi: int,
                         k_min: int, k_max: int, dirichlet_alpha: float,
                         cond_number: float) -> None:
    """Run the original balanced benchmark writing to the current flat paths."""

    N_MIXTURES    = config.DECONV_N_MIXTURES
    MIXTURE_SEED  = config.DECONV_MIXTURE_SEED
    BARPLOT_K     = config.DECONV_BARPLOT_K
    BARPLOT_N     = config.DECONV_BARPLOT_N_PLOTS
    BARPLOT_SEED  = config.DECONV_BARPLOT_SEED

    df_raw           = shared_inputs["df_raw"]
    pool_cols        = shared_inputs["pool_cols"]
    original_labels  = shared_inputs["original_labels"]
    label_map        = shared_inputs["label_map"]
    selected_probes  = shared_inputs["selected_probes"]
    eligible_tissues = shared_inputs["eligible_tissues"]
    W_masked         = shared_inputs["W_masked"]
    reference_cols   = shared_inputs["reference_cols"]

    # ---- Step 6: Generate synthetic mixtures ------------------------------
    print()
    print("=" * 60)
    print("STEP 6: Generate synthetic mixtures")
    print("=" * 60)

    X_mixtures, df_proportions = deconv_data.generate_synthetic_mixtures(
        df_raw=df_raw,
        pool_cols=pool_cols,
        original_labels=original_labels,
        label_map=label_map,
        selected_probes=selected_probes,
        eligible_tissues=eligible_tissues,
        n_mixtures=N_MIXTURES,
        k_min=k_min,
        k_max=k_max,
        dirichlet_alpha=dirichlet_alpha,
        random_state=MIXTURE_SEED,
    )

    # ---- Step 7: Validate mixtures ----------------------------------------
    print()
    print("=" * 60)
    print("STEP 7: Validate synthetic mixtures")
    print("=" * 60)

    deconv_data.validate_mixtures(X_mixtures, df_proportions)

    deconv_model.plot_proportion_distributions(
        df_proportions,
        output_path=figures_dir / "mixtures" / f"proportion_dist.{fmt}",
        dpi=dpi,
    )

    deconv_model.plot_mixture_pca(
        W_masked,
        X_mixtures,
        df_proportions,
        output_path=figures_dir / "mixtures" / f"mixture_pca.{fmt}",
        dpi=dpi,
    )

    # ---- Step 8: NNLS deconvolution ---------------------------------------
    print()
    print("=" * 60)
    print("STEP 8: Run NNLS deconvolution")
    print("=" * 60)

    df_estimated, residuals = deconv_model.deconvolve_batch(W_masked, X_mixtures)

    # ---- Step 9: Evaluate -------------------------------------------------
    print()
    print("=" * 60)
    print("STEP 9: Evaluate deconvolution performance")
    print("=" * 60)

    df_mixture_metrics = deconv_model.compute_per_mixture_metrics(
        df_proportions, df_estimated,
    )
    df_tissue_metrics = deconv_model.compute_per_tissue_metrics(
        df_proportions, df_estimated,
    )
    deconv_model.print_evaluation_summary(
        df_mixture_metrics, df_tissue_metrics, residuals,
    )

    # ---- Step 10: Evaluation plots ----------------------------------------
    print()
    print("=" * 60)
    print("STEP 10: Generate evaluation plots")
    print("=" * 60)

    deconv_model.plot_true_vs_estimated_scatter(
        df_proportions, df_estimated,
        output_path=figures_dir / f"eval_scatter_per_tissue.{fmt}",
        dpi=dpi,
    )

    deconv_model.plot_mae_by_components(
        df_mixture_metrics,
        output_path=figures_dir / f"eval_mae_by_k.{fmt}",
        dpi=dpi,
    )

    deconv_model.plot_residual_distribution(
        residuals,
        output_path=figures_dir / f"eval_residuals.{fmt}",
        dpi=dpi,
    )

    deconv_model.plot_per_tissue_mae(
        df_tissue_metrics,
        output_path=figures_dir / f"eval_per_tissue_mae.{fmt}",
        dpi=dpi,
    )

    # qualitative per-mixture comparison for a random subset
    selected_ids = deconv_model.select_random_k_mixtures(
        df_proportions=df_proportions,
        k_min=BARPLOT_K,
        n_plots=BARPLOT_N,
        random_seed=BARPLOT_SEED,
    )

    deconv_model.plot_selected_mixture_barplots(
        df_proportions=df_proportions,
        df_estimated=df_estimated,
        mixture_ids=selected_ids,
        title=(
            f"True vs predicted tissue proportions for "
            f"{len(selected_ids)} random k={BARPLOT_K} mixtures"
        ),
        output_path=figures_dir / "mixtures" / f"selected_barplots.{fmt}",
        dpi=dpi,
    )

    # ---- Summary ----------------------------------------------------------
    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Reference: {len(reference_cols)} samples, {W_masked.shape[1]} tissues")
    print(f"Pool: {len(pool_cols)} samples")
    print(f"Probes: {W_masked.shape[0]}")
    print(f"Mixtures: {N_MIXTURES} ({k_min}–{k_max} tissues, alpha={dirichlet_alpha})")
    print(f"Eligible tissues: {len(eligible_tissues)}")
    print(f"Condition number: {cond_number:.1f}")
    print(f"Median MAE: {df_mixture_metrics['mae'].median():.4f}")
    print(f"Mean Pearson r: {df_mixture_metrics['pearson_r'].mean():.4f}")
    print(f"Figures: {figures_dir}")
    print(f"Split: {config.DECONV_OUTPUT_DIR / 'split.json'}")


# ---------------------------------------------------------------------------
# Multi-regime run (--regimes flag provided)
# ---------------------------------------------------------------------------

def _run_regimes(regime_names: list, shared_inputs: dict,
                 fmt: str, dpi: int, k_min: int, k_max: int,
                 dirichlet_alpha: float, cond_number: float) -> None:
    """Dispatch deconv_regimes.run_regime() for each requested regime."""

    common_params = {
        "n_mixtures":      config.DECONV_N_MIXTURES,
        "mixture_seed":    config.DECONV_MIXTURE_SEED,
        "figures_dir":     config.DECONV_MIXTURES_FIG_DIR,
        "fmt":             fmt,
        "dpi":             dpi,
        "barplot_n":       config.DECONV_BARPLOT_N_PLOTS,
        "barplot_seed":    config.DECONV_BARPLOT_SEED,
        "barplot_k":       config.DECONV_BARPLOT_K,
        "k_min":           k_min,
        "k_max":           k_max,
        "dirichlet_alpha": dirichlet_alpha,
    }

    summary_rows = []
    for name in regime_names:
        result = deconv_regimes.run_regime(name, shared_inputs, common_params)
        summary_rows.append(result)

    # ---- Cross-regime summary table ----------------------------------------
    print()
    print("=" * 60)
    print("BENCHMARK COMPLETE — CROSS-REGIME SUMMARY")
    print("=" * 60)

    header = (
        f"{'Regime':<35} {'N':>6} {'Med MAE':>9} {'MAE pres':>9} "
        f"{'MAE abs':>9} {'Mean r':>8} {'Blood MAE':>10} {'NB recall':>10}"
    )
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        blood_str   = f"{row['blood_mae']:.4f}" if row['blood_mae'] is not None else "     n/a"
        recall_str  = f"{row['nonblood_recall']:.3f}" if row['nonblood_recall'] is not None else "      n/a"
        present_str = f"{row['median_mae_present']:.4f}" if pd.notna(row['median_mae_present']) else "      n/a"
        absent_str  = f"{row['median_mae_absent']:.4f}"  if pd.notna(row['median_mae_absent'])  else "      n/a"
        print(
            f"{row['regime_name']:<35} "
            f"{row['n_mixtures']:>6} "
            f"{row['median_mae']:>9.4f} "
            f"{present_str:>9} "
            f"{absent_str:>9} "
            f"{row['mean_pearson_r']:>8.4f} "
            f"{blood_str:>10} "
            f"{recall_str:>10}"
        )

    print()
    print(f"Reference: {len(shared_inputs['reference_cols'])} samples, "
          f"{shared_inputs['W_masked'].shape[1]} tissues")
    print(f"Pool: {len(shared_inputs['pool_cols'])} samples | "
          f"Probes: {shared_inputs['W_masked'].shape[0]} | "
          f"Condition number: {cond_number:.1f}")
    print(f"Figures: {config.DECONV_MIXTURES_FIG_DIR}")
    print(f"Split: {config.DECONV_OUTPUT_DIR / 'split.json'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Deconvolution benchmarking pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Regime options:
  baseline       Balanced synthetic benchmark (same as no-flag legacy run,
                 but writes to figures/deconvolution/mixtures/balanced/)
  cfdna_easy     Blood-dominant cfDNA easy    (40–60%% blood)
  cfdna_medium   Blood-dominant cfDNA medium  (60–80%% blood)
  cfdna_hard     Blood-dominant cfDNA hard    (80–95%% blood)
  cfdna_healthy  Blood-dominant cfDNA healthy (90–99%% blood)
  all            Shorthand: cfdna_easy + cfdna_medium + cfdna_hard
  suite          Shorthand: baseline + all cfDNA regimes

Without --regimes, the legacy balanced benchmark runs at the existing paths.
        """,
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        metavar="REGIME",
        default=None,
        help="One or more regime names (or shorthands: all, suite). "
             "If omitted, runs the legacy balanced benchmark.",
    )
    args = parser.parse_args()

    FIGURES_DIR = config.DECONV_FIGURES_DIR
    FMT         = config.FIGURE_FORMAT
    DPI         = config.FIGURE_DPI

    # Steps 1–5 run once in all cases
    shared_inputs, cond_number, k_min, k_max, dirichlet_alpha = _run_shared_setup(
        figures_dir=FIGURES_DIR,
        fmt=FMT,
        dpi=DPI,
    )

    if args.regimes is None:
        # Legacy path: balanced benchmark at existing flat figure paths
        _run_legacy_balanced(
            shared_inputs=shared_inputs,
            figures_dir=FIGURES_DIR,
            fmt=FMT,
            dpi=DPI,
            k_min=k_min,
            k_max=k_max,
            dirichlet_alpha=dirichlet_alpha,
            cond_number=cond_number,
        )
    else:
        # Resolve shorthand and validate regime names
        regime_names = deconv_regimes.resolve_regime_names(args.regimes)
        _run_regimes(
            regime_names=regime_names,
            shared_inputs=shared_inputs,
            fmt=FMT,
            dpi=DPI,
            k_min=k_min,
            k_max=k_max,
            dirichlet_alpha=dirichlet_alpha,
            cond_number=cond_number,
        )


if __name__ == "__main__":
    main()
