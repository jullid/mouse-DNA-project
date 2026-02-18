import warnings
import pandas as pd
from IPython.display import display


# --------------------------- FUNCTION DEFINITIONS -----------------------------

def build_tissue_diff_table(
    df: pd.DataFrame,
    target_threshold: float = 0.7,
    background_threshold: float = 0.8,
    diff_threshold: float = 0.3,
    use_filtering: bool = False,
    id_col: str = "probe_ID",
) -> list:
    """
    Build a long-format table of tissue-specific differential methylation scores.

    This function iterates over each tissue (column) in a tissue-averaged
    methylation matrix and, for every genomic region (row), computes:
      - target methylation (in the current tissue)
      - background methylation (mean across all other tissues)
      - a differential score (background − target)

    For each tissue, regions are ranked by the differential score, optionally
    applying simple biological filtering thresholds. Results from all tissues
    are stacked into a single DataFrame suitable for marker selection,
    downstream filtering, or heatmap construction.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with one row per genomic region and one column per tissue,
        plus an identifier column (default: "probe_ID").
    target_threshold : float, optional
        Threshold for low methylation in the target tissue (used only if
        filtering conditions are enabled).
    background_threshold : float, optional
        Threshold for high methylation in non-target tissues (used only if
        filtering conditions are enabled).
    diff_threshold : float, optional
        Minimum differential score (background − target), currently optional
        and commented out by default.
    use_filtering : bool, optional
        If False, all regions are ranked purely by differential score.
        If True, biological filtering is applied before ranking.
    id_col : str, optional
        Name of the region identifier column.

    Returns
    -------
    all_tissues_results : list of DataFrames.
    """
    # -------------------------------
    # config
    # -------------------------------

    TARGET_THRESHOLD     = target_threshold
    BACKGROUND_THRESHOLD = background_threshold
    DIFF_THRESHOLD       = diff_threshold

    # -------------------------------
    # setup
    # -------------------------------

    probe_ids = df[id_col]
    tissues   = df.columns.drop(id_col)

    all_tissues_results = []  # we will stack results here

    # -------------------------------
    # loop over tissues
    # -------------------------------

    for tissue in tissues:

        # target methylation (per row/region)
        target = df[tissue]

        # background methylation (mean of all OTHER tissues, per row)
        background = (
            df
            .drop(columns=[id_col, tissue])
            .mean(axis=1)
        )

        # difference
        diff = background - target

        # assemble per-tissue df
        tissue_df = pd.DataFrame({
            id_col:            probe_ids,
            "tissue":          tissue,
            "target_meth":     target,
            "background_meth": background,
            "diff":            diff
        })

        # switch between pure sorting on diff (use_filtering == FALSE) and filtering + sorting on diff (use_filtering == TRUE)
        if use_filtering:
            tissue_df_sorted = (
                tissue_df[
                    # (tissue_df["target_meth"] < TARGET_THRESHOLD) &
                    (tissue_df["background_meth"] > BACKGROUND_THRESHOLD)
                    # (tissue_df["diff"] > DIFF_THRESHOLD)
                ]
                .sort_values(by="diff", ascending=False)
            )

        else:
            tissue_df_sorted = tissue_df.sort_values(
                by="diff",
                ascending=False
            )

        # store results
        all_tissues_results.append(tissue_df_sorted)

    return all_tissues_results


def combine_tissue_results(all_tissues_results: list, verbose: bool = False) -> pd.DataFrame:
    """
    Concatenate per-tissue result tables into a single DataFrame.
    """

    final_df = pd.concat(all_tissues_results, ignore_index=True)

    if verbose:
        #print("Final shape:", final_df.shape)
        display(final_df.head())

    return final_df


def extract_top_regions(
    all_tissues_results: list,
    top_n: int = 25
) -> pd.DataFrame:
    """
    Extract the top N ranked regions per tissue from per-tissue result tables.

    Parameters
    ----------
    all_tissues_results : list of pandas.DataFrame
        List of per-tissue DataFrames, each sorted by differential score.
        Typically returned by build_tissue_diff_table().
    top_n : int
        Number of top regions to extract per tissue.

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame containing the top N regions per tissue.
    """

    top_regions_per_tissue = []

    for tissue_df in all_tissues_results:
        top_n_df = tissue_df.head(top_n)
        top_regions_per_tissue.append(top_n_df)

    df_top_regions = pd.concat(top_regions_per_tissue, ignore_index=True)

    # Sanity check
    #print("Top regions shape:", df_top_regions.shape)

    return df_top_regions


def stats_top_regions(df_top_regions: pd.DataFrame) -> pd.DataFrame:
    """
    Function that extracts some summary statistics for df_top_regions.
    Requires df_top_regions structure with columns = ["target_meth", "background_meth", "diff"].
    """
    metrics_cols = ["target_meth", "background_meth", "diff"]
    df_metrics   = df_top_regions[metrics_cols]

    summary_stats = df_metrics.describe().T
    summary_stats["range"] = summary_stats["max"] - summary_stats["min"]

    summary_stats = summary_stats.drop(columns=["count"]).T
    return summary_stats


def create_heatmap_matrix(
    df: pd.DataFrame,
    df_top_regions: pd.DataFrame,
    probe_col: str = "probe_ID",
    region_mode: str = "unique",
    max_per_tissue: int = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a heatmap-ready matrix (tissues x regions).

    This is the final region-selection step.  The resulting heatmap_matrix
    defines the exact set of probes used downstream (annotation, export,
    modelling, and visualisation of both real and synthetic data).

    Parameters:
    -----------
    region_mode : str
        - "all": use all regions (including duplicates)
        - "unique": keep only regions selected exactly once
    max_per_tissue : int, optional
        Maximum number of regions to keep per tissue. Only works with region_mode="unique".
        Regions are kept in their original ranking order from df_top_regions.
    verbose : bool
        If True, print detailed information about region selection
    """
    if region_mode not in {"all", "unique"}:
        raise ValueError("region_mode must be 'all' or 'unique'")

    if max_per_tissue is not None and region_mode != "unique":
        raise ValueError("max_per_tissue can only be used with region_mode='unique'. To resolve, set max_per_tissue=None.\n"
                         "Note: Using region_mode='all' will always return the same amount of regions as defined in top_N since there is no uniqueness constraint filtering.")

    if region_mode == "all":
        region_order = df_top_regions[probe_col].tolist()
    else:  # unique
        probe_counts  = df_top_regions[probe_col].value_counts()
        unique_probes = probe_counts[probe_counts == 1].index
        region_order  = (
            df_top_regions[df_top_regions[probe_col].isin(unique_probes)]
            [probe_col]
            .tolist()
        )

    # Calculate counts before capping
    counts_before = (
        df_top_regions[df_top_regions[probe_col].isin(region_order)]
        ["tissue"]
        .value_counts()
        .rename("n_regions")
        .to_frame()
        .sort_index()
    )

    # Apply max_per_tissue cap if specified
    if max_per_tissue is not None:
        # Check for tissues below max_per_tissue threshold
        tissues_below_max = counts_before[counts_before["n_regions"] < max_per_tissue]
        if not tissues_below_max.empty:
            warning_msg = "Some tissues have fewer regions than max_per_tissue:\n"
            for tissue, row in tissues_below_max.iterrows():
                warning_msg += f"  - {tissue}: {row['n_regions']} regions\n"
            warnings.warn(warning_msg, UserWarning)

        # Filter to unique regions and apply cap while preserving order
        df_filtered = df_top_regions[df_top_regions[probe_col].isin(region_order)]

        # Group by tissue and take first max_per_tissue regions (preserves df_top_regions order)
        capped_regions = (
            df_filtered
            .groupby("tissue", sort=False)
            .head(max_per_tissue)
            [probe_col]
            .tolist()
        )
        region_order = capped_regions

    # Calculate final counts
    counts = (
        df_top_regions[df_top_regions[probe_col].isin(region_order)]
        ["tissue"]
        .value_counts()
        .rename("n_regions")
        .to_frame()
        .sort_index()
    )

    # Reorder counts to match heatmap x-axis order
    ordered_tissues = (
        df_top_regions
        .loc[df_top_regions[probe_col].isin(region_order), "tissue"]
        .drop_duplicates()
        .tolist()
    )
    counts = counts.loc[ordered_tissues]

    if verbose:
        print(f"Region mode: {region_mode}")

        if max_per_tissue is not None:
            print(f"\n--- BEFORE capping (max_per_tissue={max_per_tissue}) ---")
            print(f"Total regions: {len(counts_before)}")
            display(counts_before)

            print(f"\n--- AFTER capping ---")
            print(f"Total regions: {len(region_order)}")
            display(counts)
        else:
            print(f"Total regions used: {len(region_order)}")
            display(counts)

    heatmap_df = (
        df
        .set_index(probe_col)
        .loc[region_order]
    )

    return heatmap_df.T, counts

# ------------------------------- END OF FUNCTION DEFINITIONS ----------------------------------
