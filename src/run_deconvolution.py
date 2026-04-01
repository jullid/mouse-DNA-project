"""
Deconvolution benchmarking pipeline.

Splits replicates into a reference set and mixture pool, builds a
leakage-safe signature matrix, generates synthetic mixtures, runs
NNLS deconvolution, and evaluates performance.

The full atlas pipeline (run_pipeline.py) is NOT modified by this script.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import config
import data_loading
import diff_analysis
import visualization
import deconv_data
import deconv_model
from utils import build_centroids


def main():

    # ---------------------------------- Configuration ------------------------------------
    REFERENCE_FRACTION = config.DECONV_REFERENCE_FRACTION
    SPLIT_SEED         = config.DECONV_SPLIT_SEED
    MIN_POOL_REPS      = config.DECONV_MIN_POOL_REPLICATES

    N_MIXTURES       = config.DECONV_N_MIXTURES
    K_MIN            = config.DECONV_K_MIN
    K_MAX            = config.DECONV_K_MAX
    DIRICHLET_ALPHA  = config.DECONV_DIRICHLET_ALPHA
    MIXTURE_SEED     = config.DECONV_MIXTURE_SEED

    FIGURES_DIR = config.DECONV_FIGURES_DIR
    FMT         = config.FIGURE_FORMAT
    DPI         = config.FIGURE_DPI
    # -------------------------------------------------------------------------------------


    # ---------------------------------- Step 1: Load and split ----------------------------
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
    # -------------------------------------------------------------------------------------


    # ---------------------------------- Step 2: Build reference centroids -----------------
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
    # -------------------------------------------------------------------------------------


    # ---------------------------------- Step 3: Select probes ----------------------------
    print()
    print("=" * 60)
    print("STEP 3: Select probes from reference centroids")
    print("=" * 60)

    # Run the same three-step sequence as the atlas pipeline so we get the
    # heatmap_matrix and region counts needed for the block-style heatmap plot.
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
    # -------------------------------------------------------------------------------------


    # ---------------------------------- Step 4: Build and validate reference matrix -------
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

    # diagnostic: condition number and per-tissue summary
    cond_number = deconv_data.validate_reference(W_masked)

    # correlation with full atlas
    atlas_path = config.DECONV_ATLAS_PROBES_PATH
    if atlas_path.exists():
        corr_series = deconv_data.correlate_with_full_atlas(W_masked, atlas_path)
    else:
        print(f"Full-atlas probe file not found at {atlas_path}, skipping correlation check.")
        corr_series = pd.Series(dtype=float)

    # reference heatmap — uses the same plotting function as the atlas pipeline
    visualization.plot_heatmap(
        ref_heatmap_matrix,
        ref_region_counts,
        top_n=config.MAX_PER_TISSUE,
        title=f"Reference atlas heatmap ({len(reference_cols)} replicates, "
              f"{ref_heatmap_matrix.shape[1]} probes, reference split only)",
        output_path=FIGURES_DIR / "reference" / f"reference_heatmap.{FMT}",
        dpi=DPI,
    )

    # reference vs atlas correlation plot
    if len(corr_series) > 0:
        deconv_model.plot_reference_atlas_correlation(
            corr_series,
            output_path=FIGURES_DIR / "reference" / f"reference_atlas_corr.{FMT}",
            dpi=DPI,
        )
    # -------------------------------------------------------------------------------------


    # ---------------------------------- Step 5: Determine eligible tissues ----------------
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
    # -------------------------------------------------------------------------------------


    # ---------------------------------- Step 6: Generate synthetic mixtures ---------------
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
        k_min=K_MIN,
        k_max=K_MAX,
        dirichlet_alpha=DIRICHLET_ALPHA,
        random_state=MIXTURE_SEED,
    )
    # -------------------------------------------------------------------------------------


    # ---------------------------------- Step 7: Validate mixtures ------------------------
    print()
    print("=" * 60)
    print("STEP 7: Validate synthetic mixtures")
    print("=" * 60)

    deconv_data.validate_mixtures(X_mixtures, df_proportions)

    # proportion distribution histogram
    deconv_model.plot_proportion_distributions(
        df_proportions,
        output_path=FIGURES_DIR / "mixtures" / f"proportion_dist.{FMT}",
        dpi=DPI,
    )

    # PCA sanity check: mixtures should lie between reference centroids
    deconv_model.plot_mixture_pca(
        W_masked,
        X_mixtures,
        df_proportions,
        output_path=FIGURES_DIR / "mixtures" / f"mixture_pca.{FMT}",
        dpi=DPI,
    )
    # -------------------------------------------------------------------------------------


    # ---------------------------------- Step 8: NNLS deconvolution -----------------------
    print()
    print("=" * 60)
    print("STEP 8: Run NNLS deconvolution")
    print("=" * 60)

    df_estimated, residuals = deconv_model.deconvolve_batch(W_masked, X_mixtures)
    # -------------------------------------------------------------------------------------


    # ---------------------------------- Step 9: Evaluate ---------------------------------
    print()
    print("=" * 60)
    print("STEP 9: Evaluate deconvolution performance")
    print("=" * 60)

    # per-mixture metrics
    df_mixture_metrics = deconv_model.compute_per_mixture_metrics(
        df_proportions, df_estimated,
    )

    # per-tissue metrics
    df_tissue_metrics = deconv_model.compute_per_tissue_metrics(
        df_proportions, df_estimated,
    )

    # summary printout
    deconv_model.print_evaluation_summary(
        df_mixture_metrics, df_tissue_metrics, residuals,
    )
    # -------------------------------------------------------------------------------------


    # ---------------------------------- Step 10: Evaluation plots -------------------------
    print()
    print("=" * 60)
    print("STEP 10: Generate evaluation plots")
    print("=" * 60)

    # true vs estimated scatter (per tissue panels)
    deconv_model.plot_true_vs_estimated_scatter(
        df_proportions, df_estimated,
        output_path=FIGURES_DIR / f"eval_scatter_per_tissue.{FMT}",
        dpi=DPI,
    )

    # MAE by number of components
    deconv_model.plot_mae_by_components(
        df_mixture_metrics,
        output_path=FIGURES_DIR / f"eval_mae_by_k.{FMT}",
        dpi=DPI,
    )

    # residual distribution
    deconv_model.plot_residual_distribution(
        residuals,
        output_path=FIGURES_DIR / f"eval_residuals.{FMT}",
        dpi=DPI,
    )

    # per-tissue MAE bar chart
    deconv_model.plot_per_tissue_mae(
        df_tissue_metrics,
        output_path=FIGURES_DIR / f"eval_per_tissue_mae.{FMT}",
        dpi=DPI,
    )
    # -------------------------------------------------------------------------------------


    # ---------------------------------- Summary ------------------------------------------
    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Reference: {len(reference_cols)} samples, {W_masked.shape[1]} tissues")
    print(f"Pool: {len(pool_cols)} samples")
    print(f"Probes: {W_masked.shape[0]}")
    print(f"Mixtures: {N_MIXTURES} ({K_MIN}–{K_MAX} tissues, alpha={DIRICHLET_ALPHA})")
    print(f"Eligible tissues: {len(eligible_tissues)}")
    print(f"Condition number: {cond_number:.1f}")
    print(f"Median MAE: {df_mixture_metrics['mae'].median():.4f}")
    print(f"Mean Pearson r: {df_mixture_metrics['pearson_r'].mean():.4f}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Split: {config.DECONV_OUTPUT_DIR / 'split.json'}")
    # -------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
