"""
Deconvolution modelling module.

Contains the NNLS solver, per-mixture and per-tissue evaluation metrics,
and all deconvolution-specific plots.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import nnls


# --------------------------- FUNCTION DEFINITIONS -----------------------------

# ---- NNLS solver -------------------------------------------------------------

def deconvolve_single(
    W: np.ndarray,
    mixture_vec: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Deconvolve a single mixture vector against reference matrix W using NNLS.

    Returns normalised proportions (summing to 1) and the raw NNLS residual.
    The function is stateless and deterministic — suitable for both batch
    benchmarking and future single-sample inference.

    Parameters
    ----------
    W : np.ndarray
        Reference matrix, shape (n_probes, n_tissues).
    mixture_vec : np.ndarray
        Mixture beta vector, shape (n_probes,).

    Returns
    -------
    proportions : np.ndarray
        Estimated tissue proportions (normalised to sum to 1),
        shape (n_tissues,).
    residual : float
        NNLS residual norm.
    """
    raw_coeffs, residual = nnls(W, mixture_vec)

    # normalise to proportions (sum to 1)
    total = raw_coeffs.sum()
    if total > 0:
        proportions = raw_coeffs / total
    else:
        # edge case: all coefficients are zero
        proportions = raw_coeffs

    return proportions, residual


def deconvolve_batch(
    W_masked: pd.DataFrame,
    X_mixtures: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Deconvolve a batch of synthetic mixtures against reference matrix W.

    Parameters
    ----------
    W_masked : pd.DataFrame
        Reference matrix, shape (n_probes, n_tissues).
        Index = probe IDs, columns = tissue names.
    X_mixtures : np.ndarray
        Mixture beta vectors, shape (n_mixtures, n_probes).

    Returns
    -------
    df_estimated : pd.DataFrame
        Estimated proportions, shape (n_mixtures, n_tissues).
        Columns = tissue names, index = mixture_0, mixture_1, etc.
    residuals : np.ndarray
        NNLS residual per mixture, shape (n_mixtures,).
    """
    W = W_masked.values
    tissue_names = W_masked.columns.tolist()
    n_mixtures   = X_mixtures.shape[0]

    estimated = np.zeros((n_mixtures, len(tissue_names)))
    residuals = np.zeros(n_mixtures)

    for i in range(n_mixtures):
        estimated[i], residuals[i] = deconvolve_single(W, X_mixtures[i])

    df_estimated = pd.DataFrame(
        estimated,
        columns=tissue_names,
        index=[f"mixture_{i}" for i in range(n_mixtures)],
    )

    print(f"Deconvolved {n_mixtures} mixtures")
    print(f"  Mean residual: {residuals.mean():.4f}")
    print(f"  Max residual:  {residuals.max():.4f}")

    return df_estimated, residuals


# ---- Evaluation metrics ------------------------------------------------------

def compute_per_mixture_metrics(
    df_true: pd.DataFrame,
    df_estimated: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-mixture evaluation metrics.

    For each mixture: MAE, RMSE, and Pearson correlation between true
    and estimated proportion vectors (across all tissues including zeros).

    Parameters
    ----------
    df_true : pd.DataFrame
        Ground-truth proportions, shape (n_mixtures, n_tissues).
    df_estimated : pd.DataFrame
        Estimated proportions, shape (n_mixtures, n_tissues).

    Returns
    -------
    pd.DataFrame
        Per-mixture metrics with columns: mae, rmse, pearson_r.
    """
    # align columns (in case of ordering differences)
    shared_tissues = df_true.columns.intersection(df_estimated.columns)
    true_arr = df_true[shared_tissues].values
    est_arr  = df_estimated[shared_tissues].values

    n_mixtures = true_arr.shape[0]

    records = []
    for i in range(n_mixtures):
        diff = true_arr[i] - est_arr[i]
        mae  = np.abs(diff).mean()
        rmse = np.sqrt((diff ** 2).mean())

        # Pearson correlation (handle edge case where one vector is constant)
        if true_arr[i].std() == 0 or est_arr[i].std() == 0:
            pearson_r = np.nan
        else:
            pearson_r = np.corrcoef(true_arr[i], est_arr[i])[0, 1]

        records.append({"mae": mae, "rmse": rmse, "pearson_r": pearson_r})

    df_metrics = pd.DataFrame(records, index=df_true.index)

    # add number of true components per mixture
    df_metrics["n_components"] = (df_true > 0).sum(axis=1).values

    return df_metrics


def compute_per_tissue_metrics(
    df_true: pd.DataFrame,
    df_estimated: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-tissue evaluation metrics aggregated across all mixtures.

    For each tissue:
    - mean_mae: mean absolute error across all mixtures
    - pearson_r: correlation between true and estimated across mixtures
    - mean_false_positive: mean estimated proportion when true proportion is 0
    - n_present: number of mixtures where this tissue has proportion > 0

    Parameters
    ----------
    df_true : pd.DataFrame
        Ground-truth proportions.
    df_estimated : pd.DataFrame
        Estimated proportions.

    Returns
    -------
    pd.DataFrame
        Per-tissue metrics, indexed by tissue name.
    """
    shared_tissues = df_true.columns.intersection(df_estimated.columns)

    records = []
    for tissue in shared_tissues:
        true_vals = df_true[tissue].values
        est_vals  = df_estimated[tissue].values

        mae = np.abs(true_vals - est_vals).mean()

        # correlation (handle edge case)
        if true_vals.std() == 0 or est_vals.std() == 0:
            pearson_r = np.nan
        else:
            pearson_r = np.corrcoef(true_vals, est_vals)[0, 1]

        # false positive rate: mean estimated proportion when truth is 0
        zero_mask = true_vals == 0
        if zero_mask.sum() > 0:
            mean_fp = est_vals[zero_mask].mean()
        else:
            mean_fp = np.nan

        # number of mixtures where this tissue is present
        n_present = (true_vals > 0).sum()

        records.append({
            "tissue": tissue,
            "mean_mae": mae,
            "pearson_r": pearson_r,
            "mean_false_positive": mean_fp,
            "n_present": n_present,
        })

    df_tissue_metrics = (
        pd.DataFrame(records)
        .set_index("tissue")
        .sort_values("mean_mae", ascending=False)
    )

    return df_tissue_metrics


def print_evaluation_summary(
    df_mixture_metrics: pd.DataFrame,
    df_tissue_metrics: pd.DataFrame,
    residuals: np.ndarray,
) -> None:
    """
    Print a concise summary of deconvolution performance.

    Parameters
    ----------
    df_mixture_metrics : pd.DataFrame
        Per-mixture metrics from compute_per_mixture_metrics().
    df_tissue_metrics : pd.DataFrame
        Per-tissue metrics from compute_per_tissue_metrics().
    residuals : np.ndarray
        NNLS residuals per mixture.
    """
    print("=" * 60)
    print("DECONVOLUTION EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nMixtures evaluated: {len(df_mixture_metrics)}")

    print(f"\nOverall metrics (across all mixtures):")
    print(f"  Median MAE:          {df_mixture_metrics['mae'].median():.4f}")
    print(f"  Mean MAE:            {df_mixture_metrics['mae'].mean():.4f}")
    print(f"  Mean RMSE:           {df_mixture_metrics['rmse'].mean():.4f}")
    print(f"  Mean Pearson r:      {df_mixture_metrics['pearson_r'].mean():.4f}")

    # dominant tissue accuracy: does highest estimated match highest true?
    # (only for mixtures where there is a clear dominant tissue)
    n_correct_dominant = 0
    n_total = len(df_mixture_metrics)
    # need original proportion dataframes for this — compute from metrics index
    # Actually this needs the raw proportions, so we'll skip if not available

    print(f"\nMean NNLS residual:    {residuals.mean():.4f}")
    print(f"Max NNLS residual:     {residuals.max():.4f}")

    print(f"\nMAE by number of components:")
    for k, group in df_mixture_metrics.groupby("n_components"):
        print(f"  k={k}: median MAE={group['mae'].median():.4f}, "
              f"n={len(group)}")

    print(f"\nPer-tissue summary (sorted by MAE, worst first):")
    print(df_tissue_metrics.to_string())

    # tissues with highest false positive rates
    fp_sorted = df_tissue_metrics["mean_false_positive"].dropna().sort_values(ascending=False)
    print(f"\nHighest false positive rates (top 5):")
    for tissue, fp in fp_sorted.head(5).items():
        print(f"  {tissue:30s} {fp:.4f}")


# ---- Plots -------------------------------------------------------------------

def _save_or_close(output_path=None, dpi=150):
    """Save the current figure to disk if output_path is provided, then close it."""
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved → {output_path}")
    plt.close("all")


def plot_reference_heatmap(
    W_masked: pd.DataFrame,
    title: str = "Reference atlas heatmap (leakage-safe split)",
    output_path: Path = None,
    dpi: int = 150,
) -> None:
    """
    Plot heatmap of the reference matrix W.

    Parameters
    ----------
    W_masked : pd.DataFrame
        Reference matrix (probes × tissues).
    """
    from matplotlib.colors import LinearSegmentedColormap
    blue_yellow = LinearSegmentedColormap.from_list("blue_yellow", ["#0000FF", "#ffff00"])

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        W_masked.T,     # tissues × probes for display
        cmap=blue_yellow,
        vmin=0, vmax=1,
        xticklabels=False,
        yticklabels=True,
        cbar_kws={"shrink": 0.75, "pad": 0.03},
        ax=ax,
    )
    ax.set_xlabel("Selected probes")
    ax.set_ylabel("Tissues")
    ax.set_title(title)

    _save_or_close(output_path, dpi=dpi)


def plot_reference_atlas_correlation(
    corr_series: pd.Series,
    title: str = "Reference vs full-atlas tissue correlation",
    output_path: Path = None,
    dpi: int = 150,
) -> None:
    """
    Plot per-tissue correlation between reference and full atlas as a bar chart.

    Parameters
    ----------
    corr_series : pd.Series
        Per-tissue Pearson correlation, indexed by tissue name.
        Returned by deconv_data.correlate_with_full_atlas().
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(corr_series.index, corr_series.values, color="steelblue")
    ax.set_xlabel("Pearson correlation")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.axvline(x=0.95, color="red", linestyle="--", alpha=0.7, label="r = 0.95")
    ax.legend()

    # annotate bars
    for bar, val in zip(bars, corr_series.values):
        ax.text(
            val + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", fontsize=8,
        )

    plt.tight_layout()
    _save_or_close(output_path, dpi=dpi)


def plot_proportion_distributions(
    df_proportions: pd.DataFrame,
    title: str = "Distribution of non-zero mixture proportions",
    output_path: Path = None,
    dpi: int = 150,
) -> None:
    """
    Plot histogram of non-zero proportions across all mixtures.

    Parameters
    ----------
    df_proportions : pd.DataFrame
        Ground-truth proportions, shape (n_mixtures, n_tissues).
    """
    nonzero = df_proportions.values[df_proportions.values > 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(nonzero, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Proportion")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.axvline(x=nonzero.mean(), color="red", linestyle="--",
               label=f"mean = {nonzero.mean():.3f}")
    ax.legend()

    plt.tight_layout()
    _save_or_close(output_path, dpi=dpi)


def plot_mixture_pca(
    W_masked: pd.DataFrame,
    X_mixtures: np.ndarray,
    df_proportions: pd.DataFrame,
    title: str = "PCA: reference centroids and synthetic mixtures",
    output_path: Path = None,
    dpi: int = 150,
) -> None:
    """
    Project reference centroids and synthetic mixtures into PCA space.

    Mixtures should lie between the centroids of their component tissues.
    This is the strongest visual validation that the mixtures are
    mathematically correct.

    Parameters
    ----------
    W_masked : pd.DataFrame
        Reference matrix (probes × tissues).
    X_mixtures : np.ndarray
        Mixture beta vectors, shape (n_mixtures, n_probes).
    df_proportions : pd.DataFrame
        Ground-truth proportions (used for colouring mixtures by dominant tissue).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # reference centroids: transpose W to (tissues × probes), i.e. each tissue is a "sample"
    ref_matrix = W_masked.T.values
    tissue_names = W_masked.columns.tolist()

    # combine reference and mixtures for joint PCA
    combined = np.vstack([ref_matrix, X_mixtures])
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)

    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined_scaled)

    n_ref = len(tissue_names)
    ref_pca = combined_pca[:n_ref]
    mix_pca = combined_pca[n_ref:]

    # colour mixtures by dominant tissue
    dominant_tissue = df_proportions.idxmax(axis=1)

    fig, ax = plt.subplots(figsize=(12, 9))

    # plot mixtures first (behind)
    # create colour map for tissues
    unique_tissues = sorted(dominant_tissue.unique())
    cmap = plt.cm.get_cmap("tab20", len(unique_tissues))
    tissue_to_color = {t: cmap(i) for i, t in enumerate(unique_tissues)}

    mix_colors = [tissue_to_color.get(t, "grey") for t in dominant_tissue]
    ax.scatter(mix_pca[:, 0], mix_pca[:, 1],
               c=mix_colors, alpha=0.3, s=15, label="_nolegend_")

    # plot reference centroids on top
    ax.scatter(ref_pca[:, 0], ref_pca[:, 1],
               c="black", s=80, marker="^", zorder=5, edgecolors="white", linewidths=0.5)

    for i, tissue in enumerate(tissue_names):
        ax.annotate(tissue, (ref_pca[i, 0], ref_pca[i, 1]),
                    fontsize=7, ha="center", va="bottom")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title)

    plt.tight_layout()
    _save_or_close(output_path, dpi=dpi)


def plot_true_vs_estimated_scatter(
    df_true: pd.DataFrame,
    df_estimated: pd.DataFrame,
    title: str = "True vs estimated proportions",
    output_path: Path = None,
    dpi: int = 150,
) -> None:
    """
    Scatter plot of true vs estimated proportions, one panel per tissue.

    Parameters
    ----------
    df_true : pd.DataFrame
        Ground-truth proportions.
    df_estimated : pd.DataFrame
        Estimated proportions.
    """
    shared_tissues = sorted(df_true.columns.intersection(df_estimated.columns))
    n_tissues = len(shared_tissues)

    n_cols = 5
    n_rows = int(np.ceil(n_tissues / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for i, tissue in enumerate(shared_tissues):
        ax = axes[i]
        true_vals = df_true[tissue].values
        est_vals  = df_estimated[tissue].values

        ax.scatter(true_vals, est_vals, alpha=0.3, s=10, c="steelblue")
        ax.plot([0, 1], [0, 1], "r--", linewidth=0.8, alpha=0.7)

        # compute correlation for this tissue
        if true_vals.std() > 0 and est_vals.std() > 0:
            r = np.corrcoef(true_vals, est_vals)[0, 1]
            ax.set_title(f"{tissue}\nr={r:.3f}", fontsize=8)
        else:
            ax.set_title(tissue, fontsize=8)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.supxlabel("True proportion", fontsize=10)
    fig.supylabel("Estimated proportion", fontsize=10)
    plt.tight_layout()

    _save_or_close(output_path, dpi=dpi)


def plot_mae_by_components(
    df_mixture_metrics: pd.DataFrame,
    title: str = "MAE distribution by number of mixture components",
    output_path: Path = None,
    dpi: int = 150,
) -> None:
    """
    Box plot of MAE distribution stratified by number of components k.

    Parameters
    ----------
    df_mixture_metrics : pd.DataFrame
        Per-mixture metrics from compute_per_mixture_metrics().
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    groups = df_mixture_metrics.groupby("n_components")["mae"]
    positions = sorted(groups.groups.keys())
    data = [groups.get_group(k).values for k in positions]

    bp = ax.boxplot(data, positions=range(len(positions)), patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"k={k}" for k in positions])
    ax.set_xlabel("Number of tissue components")
    ax.set_ylabel("MAE")
    ax.set_title(title)

    plt.tight_layout()
    _save_or_close(output_path, dpi=dpi)


def plot_residual_distribution(
    residuals: np.ndarray,
    title: str = "NNLS residual distribution",
    output_path: Path = None,
    dpi: int = 150,
) -> None:
    """
    Histogram of NNLS residuals across all mixtures.

    Parameters
    ----------
    residuals : np.ndarray
        NNLS residual per mixture.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xlabel("NNLS residual")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.axvline(x=np.median(residuals), color="red", linestyle="--",
               label=f"median = {np.median(residuals):.4f}")
    ax.legend()

    plt.tight_layout()
    _save_or_close(output_path, dpi=dpi)


def plot_per_tissue_mae(
    df_tissue_metrics: pd.DataFrame,
    title: str = "Per-tissue mean absolute error",
    output_path: Path = None,
    dpi: int = 150,
) -> None:
    """
    Horizontal bar chart of per-tissue MAE.

    Parameters
    ----------
    df_tissue_metrics : pd.DataFrame
        Per-tissue metrics from compute_per_tissue_metrics().
    """
    df_sorted = df_tissue_metrics.sort_values("mean_mae", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(df_sorted.index, df_sorted["mean_mae"], color="steelblue")
    ax.set_xlabel("Mean absolute error")
    ax.set_title(title)

    # annotate with n_present
    for bar, (tissue, row) in zip(bars, df_sorted.iterrows()):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"n={int(row['n_present'])}",
            va="center", fontsize=8,
        )

    plt.tight_layout()
    _save_or_close(output_path, dpi=dpi)

# ------------------------------- END OF FUNCTION DEFINITIONS ----------------------------------
