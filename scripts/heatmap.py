# Import the modified CSV file for better visualization.
# Note that the added sufixes (".1", ".2", ..., ".25") is added by pandas when read by doing df = pd.read_csv(csv_file).
# The actual CSV does not contain those sufixes.
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from IPython.display import display, HTML
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from matplotlib.colors import LinearSegmentedColormap


DATA_DIR = Path.home() / "projects" / "data"

csv_file = DATA_DIR / "processed" / "GSE290585_SeSaMeBeta_MM285_BS_simplified_columns.csv" # path to mouse csv file

print(sys.executable)
print(csv_file.exists())

df = pd.read_csv(csv_file)

# with pd.option_context("display.max_columns", None,
#                        "display.width", None):
#         display(df.head(20))

# --- Colormap used in plots ---
blue_yellow = LinearSegmentedColormap.from_list(
    "blue_yellow",
    ["#0000FF", "#ffff00"]
    )

# ------------------------------- STATS -------------------------------------
# count how many cell type duplicates
# cell_type_counts = df.columns[1:].str.split(".", n=1).str[0].value_counts()
# print(cell_type_counts)

# # stats
# X = df.drop(columns=['probe_ID'])
# stats = pd.DataFrame({
#     'std': X.std(axis=0),
#     'range': X.max(axis=0) - X.min(axis=0)
# }).T

# print("BASIC STATS: STD & RANGE:")
# with pd.option_context("display.max_columns", None,
#                        "display.width", None):
#         display(stats)
# ----------------------------------------------------------------------------



# -------------------------------- df_cell_type ------------------------------
# Collapse cell type columns into one averaged column per cell type, resulting in 29 averaged cell types/columns
probe_ids = df["probe_ID"]
df_drop_ID = df.drop(columns="probe_ID")

# remove suffix .1 .2 etc. added by pandas
cell_type_labels = df_drop_ID.columns.str.split(".", n=1).str[0]

# collapse replicates
df_celltype = df_drop_ID.groupby(cell_type_labels, axis=1).mean()

# Reattach probe_ID
df_celltype.insert(0, "probe_ID", probe_ids)


# with pd.option_context("display.max_columns", None,
#                        "display.width", None):
#         display(df_celltype.head(20))

#df_celltype.shape
# ------------------------------------------------------------------------------



# --------------------------------- df_merged ----------------------------------
"""
After analysis of (725 region, 29 tissue) correlation matrix, we will do the following:

MERGE:
- Retina & Eye
- Brain_Cortex & Subcortinal_Brain
- Thymus & Blood & Spleen
- Tail & Ear & Skin

REMOVE:
- Optic_Nerve
- Sciatic_Nerve
(These are removed because cancer rarely metastasizes here)

This is done by filtering celltype_df and then merging celltypes. Expected: (525 regions, 21 tissues). HOWEVER, we will now use more than top25 regions -> Maybe top 100, 250 or even 500.
"""

# Drop columns that we don't need
df_filtered_tissues = df_celltype.drop(columns=['Sciatic_Nerve', 'Optic_Nerve'])

# Define "merge groups", the KEYS are the new merged tissue names.
merge_groups = {
    "Eye_Retina": ["Eye", "Retina"],
    "Brain_Cortex_Subcortical": ["Brain_Cortex", "Subcortical_Brain"],
    "Blood_Spleen_Thymus": ["Blood", "Spleen", "Thymus"],
    "Skin_Ears_Tail": ["Skin", "Ears", "Tail"],
}

# Build new merged df
df_merged = pd.DataFrame()
df_merged["probe_ID"] = df_filtered_tissues["probe_ID"] # copy over IDs

# add merged columns
for new_name, cols in merge_groups.items():
    df_merged[new_name] = df_filtered_tissues[cols].mean(axis=1)

# keep unmerged tissue as-is, flatten the merge_groups list
merged_cols_flat = [c for cols in merge_groups.values() for c in cols]

# keep non-merged columns
remaining_cols = []
for c in df_filtered_tissues.columns:
    if c not in merged_cols_flat and c != "probe_ID":
        remaining_cols.append(c)

# copy remaining columns into the merged df
df_merged[remaining_cols] = df_filtered_tissues[remaining_cols]

print(df_merged.shape)
#print(df_merged.columns)

# with pd.option_context("display.max_columns", None,
#                        "display.width", None):
#         display(df_merged.head(20))
# ------------------------------------------------------------------------------



# --------------------------- FUNCTION DEFINITIONS -----------------------------
def build_tissue_diff_table(
    df: pd.DataFrame,
    target_threshold: float = 0.7,
    background_threshold: float = 0.8,
    diff_threshold: float = 0.3,
    use_filtering: bool = False,
    id_col: str = "probe_ID",
) -> pd.DataFrame:
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
    df_celltype : pandas.DataFrame
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

    TARGET_THRESHOLD = target_threshold
    BACKGROUND_THRESHOLD = background_threshold
    DIFF_THRESHOLD = diff_threshold

    # -------------------------------
    # setup
    # -------------------------------

    probe_ids = df[id_col]
    tissues = df.columns.drop(id_col)

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
            id_col: probe_ids,
            "tissue": tissue,
            "target_meth": target,
            "background_meth": background,
            "diff": diff
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


def combine_tissue_results(all_tissues_results, verbose=False):
    """
    Concatenate per-tissue result tables into a single DataFrame.
    """

    final_df = pd.concat(all_tissues_results, ignore_index=True)

    if verbose:
        print("Final shape:", final_df.shape)
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
    print("Top regions shape:", df_top_regions.shape)
    return df_top_regions


def stats_top_regions(df_top_regions):
    """
    Function that extracts some summary statistics for df_top_regions.
    Requires df_top_regions structure with columns = ["target_meth", "background_meth", "diff"].
    """
    metrics_cols = ["target_meth", "background_meth", "diff"]
    df_metrics = df_top_regions[metrics_cols]

    summary_stats = df_metrics.describe().T
    summary_stats["range"] = summary_stats["max"] - summary_stats["min"]

    summary_stats = summary_stats.drop(columns=["count"]).T
    return summary_stats


def create_heatmap_matrix(df, df_top_regions, probe_col="probe_ID", region_mode="all", verbose=False):
    
    """
    Create a heatmap-ready matrix (tissues x regions).

    region_mode:
        - "all": use all regions (including duplicates)
        - "unique": keep only regions selected exactly once
    """

    if region_mode not in {"all", "unique"}:
        raise ValueError("region_mode must be 'all' or 'unique'")

    if region_mode == "all":
        region_order = df_top_regions[probe_col].tolist()

    else:  # unique
        probe_counts = df_top_regions[probe_col].value_counts()
        unique_probes = probe_counts[probe_counts == 1].index
        region_order = (
            df_top_regions[df_top_regions[probe_col].isin(unique_probes)]
            [probe_col]
            .tolist()
        )

    counts = (
        df_top_regions[df_top_regions[probe_col].isin(region_order)]
        ["tissue"]
        .value_counts()
        .rename("n_regions")
        .to_frame()
        .sort_index()
    )

    # reorder counts to match heatmap x-axis order
    ordered_tissues = (
        df_top_regions
        .loc[df_top_regions[probe_col].isin(region_order), "tissue"]
        .drop_duplicates()
        .tolist()
    )

    counts = counts.loc[ordered_tissues]

    # if verbose = True, display distribution of regions in tissues.
    if verbose:
        print(f"Region mode: {region_mode}")
        print(f"Total regions used: {len(region_order)}")
        display(counts)

    heatmap_df = (
        df
        .set_index(probe_col)
        .loc[region_order]
    )
    return heatmap_df.T, counts



def plot_heatmap(heatmap_matrix, counts, top_n=None, title=None, figsize=(20, 8)):
    """
    Plot methylation heatmap with automatic tissue block labels.

    Parameters
    ----------
    heatmap_matrix : pd.DataFrame
        DataFrame of shape (tissues × regions)
    counts : pd.DataFrame
        DataFrame with index = tissue names (in plot order)
        and column 'n_regions' giving number of regions per tissue
    """

    # extract labels and block sizes from counts
    # counts is a df with index (cell type) and n_regions in each tissue
    tissue_labels = counts.index.tolist()
    block_sizes = counts["n_regions"].values

    # compute xtick positions (center of each tissue block)
    xtick_positions = []
    current_pos = 0

    for size in block_sizes:
        xtick_positions.append(current_pos + size / 2)
        current_pos += size

    # plot
    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        heatmap_matrix,
        cmap=blue_yellow,
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=True,
        cbar_kws={"shrink": 0.75, "pad": 0.03}
    )

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(tissue_labels, rotation=90)
    ax.set_ylabel("Tissues")

    if top_n is not None:
        ax.set_xlabel(f"Tissue-specific regions (top {top_n} per tissue)")
    else:
        ax.set_xlabel("Tissue-specific regions")

    if title is not None:
        ax.set_title(title)
    plt.show()


def compute_tissue_correlation(heatmap_matrix, method="pearson"):
    """
    Compute tissue–tissue correlation matrix. Shared computation function, avoids recomputing T.corr
    """
    return heatmap_matrix.T.corr(method=method)


def plot_tissue_correlation(tissue_corr, figsize=(10, 8), title=None):
    """
    Plot tissue correlation, colormap between 0 and 1.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        tissue_corr,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75, "pad": 0.03}
    )
    plt.title(title or "Tissue–tissue methylation correlation")
    plt.tight_layout()
    plt.show()


def plot_tissue_dendrogram(tissue_corr, method="average", figsize=(10, 6), title=None):
    """
    Plot Dendrogram
    """
    # hierarchical clustering is based on distances, here we convert tissue_corr to distances giving us:
    # correlation = 1  -> distance = 0 (identical) 
    # correlation = 0  -> distance = 1 (unrelated)
    # correlation = -1 -> distance = 2 (opposite)
    distance_matrix = 1 - tissue_corr
    condensed_dist = squareform(distance_matrix.values)

    Z = linkage(condensed_dist, method=method)
    plt.figure(figsize=figsize)
    dendrogram(
        Z,
        labels=tissue_corr.index.tolist(),
        leaf_rotation=90
    )

    plt.ylabel("Distance (1 - correlation)")
    plt.title(title or "Hierarchical clustering of tissues")
    plt.tight_layout()
    plt.show()
    return Z


def plot_clustered_tissue_correlation(tissue_corr, figsize=(10, 10), title=None):
    """
    Plot clustered tissue correlation.
    """
    sns.clustermap(
        tissue_corr,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        figsize=figsize,
        cbar_kws={"shrink": 0.75, "pad": 0.03}
    )
    if title:
        plt.suptitle(title, y=1.02)
# ------------------------------- END OF FUNCTION DEFINITIONS ----------------------------------



# ---------------------------------- Pipeline Configuration ------------------------------------
""" PIPELINE CONFIG"""

# input dataframe
DATAFRAME = df_merged            # tissue-averaged beta matrix (probe_ID + tissues)

# use filtering + diff ranking (True) VS. Only diff ranking (False)
USE_FILTERING = True               # passed to build_tissue_diff_table
USE_VERBOSE = True                 # display the distribution of regions in cell types.

# region selection
TOP_N = 200                        # number of top regions per tissue
REGION_MODE = "unique"                # "all" or "unique"

# correlation / clustering
CORR_METHOD = "pearson"            # "pearson" or "spearman"
CLUSTER_METHOD = "average"         # linkage method

# plotting
HEATMAP_TITLE = "Mouse methylation atlas heatmap"
CORR_TITLE = "Tissue–tissue methylation correlation"
DENDRO_TITLE = "Hierarchical clustering of tissues (methylation correlation)"
CLUSTERED_CORR_TITLE = "Clustered tissue–tissue methylation correlation"
# ---------------------------------------------------------------------------------------------



# ----------------------------------------- PIPELINE ------------------------------------------
""" FULL PIPELINE"""

# Step 1: compute per-tissue differential tables
all_tissues_results = build_tissue_diff_table(
    df=DATAFRAME,
    use_filtering=USE_FILTERING
)

# Step 2: extract top N regions per tissue
df_top_regions = extract_top_regions(
    all_tissues_results,
    top_n=TOP_N
)

# Step 3: create heatmap matrix (tissues × regions)
heatmap_matrix, regions_per_tissue_count = create_heatmap_matrix(
    df=DATAFRAME,
    df_top_regions=df_top_regions,
    region_mode=REGION_MODE,
    verbose=USE_VERBOSE
)

# Step 4: plot methylation heatmap
plot_heatmap(
    heatmap_matrix,
    regions_per_tissue_count,
    top_n=TOP_N,
    title=f"{HEATMAP_TITLE} (top {TOP_N} regions per tissue) - {REGION_MODE} regions"
)

# Step 5: compute tissue–tissue correlation matrix
tissue_corr = compute_tissue_correlation(
    heatmap_matrix,
    method=CORR_METHOD
)

# Step 6: plot tissue correlation heatmap
plot_tissue_correlation(
    tissue_corr,
    title=CORR_TITLE
)

# Step 7: plot tissue dendrogram (hierarchical clustering)
Z = plot_tissue_dendrogram(
    tissue_corr,
    method=CLUSTER_METHOD,
    title=DENDRO_TITLE
)

# Step 8: plot clustered tissue correlation heatmap
plot_clustered_tissue_correlation(
    tissue_corr,
    title=CLUSTERED_CORR_TITLE
)


