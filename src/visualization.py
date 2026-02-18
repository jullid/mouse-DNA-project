import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required on headless HPC nodes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


# --- Colormap used in plots ---
blue_yellow = LinearSegmentedColormap.from_list(
    "blue_yellow",
    ["#0000FF", "#ffff00"]
)


# --------------------------- HELPER ------------------------------------------

def _save_or_close(output_path=None, dpi=150):
    """
    Save the current figure to disk if output_path is provided, then close it.

    Never calls plt.show() -- safe for headless HPC nodes.
    Always closes the figure to free memory, even if no path is given.

    Parameters
    ----------
    output_path : Path, optional
        Destination file. Format is inferred from the suffix (.pdf, .png, etc.).
    dpi : int
        Resolution for raster formats (png, etc.). Ignored for vector formats (pdf).
    """
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved -> {output_path}")
    plt.close("all")


# --------------------------- FUNCTION DEFINITIONS -----------------------------

def plot_heatmap(
    heatmap_matrix,
    counts,
    top_n=None,
    title=None,
    figsize=(20, 8),
    output_path=None,
    dpi=150,
):
    """
    Plot methylation heatmap with automatic tissue block labels.

    Accepts any tissues x regions matrix, including synthetic data matrices,
    as long as the index is tissue names and columns are region/probe IDs.

    Parameters
    ----------
    heatmap_matrix : pd.DataFrame
        DataFrame of shape (tissues x regions)
    counts : pd.DataFrame
        DataFrame with index = tissue names (in plot order)
        and column 'n_regions' giving number of regions per tissue
    output_path : Path, optional
        If provided, the figure is saved here. If None, the figure is closed
        without saving.
    dpi : int
        Resolution for raster formats.
    """

    # extract labels and block sizes from counts
    # counts is a df with index (cell type) and n_regions in each tissue
    tissue_labels = counts.index.tolist()
    block_sizes   = counts["n_regions"].values

    # compute xtick positions (center of each tissue block)
    xtick_positions = []
    current_pos     = 0

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

    _save_or_close(output_path, dpi=dpi)


def compute_tissue_correlation(heatmap_matrix, method="pearson"):
    """
    Compute tissue-tissue correlation matrix. Shared computation function, avoids recomputing T.corr
    """
    return heatmap_matrix.T.corr(method=method)


def plot_tissue_correlation(
    tissue_corr,
    figsize=(10, 8),
    title=None,
    output_path=None,
    dpi=150,
):
    """
    Plot tissue correlation, colormap between 0 and 1.

    Parameters
    ----------
    output_path : Path, optional
        If provided, the figure is saved here. If None, the figure is closed
        without saving.
    dpi : int
        Resolution for raster formats.
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
    plt.title(title or "Tissue-tissue methylation correlation")
    plt.tight_layout()
    _save_or_close(output_path, dpi=dpi)


def plot_tissue_dendrogram(
    tissue_corr,
    method="average",
    figsize=(10, 6),
    title=None,
    output_path=None,
    dpi=150,
):
    """
    Plot Dendrogram

    Parameters
    ----------
    output_path : Path, optional
        If provided, the figure is saved here. If None, the figure is closed
        without saving.
    dpi : int
        Resolution for raster formats.
    """
    # hierarchical clustering is based on distances, here we convert tissue_corr to distances giving us:
    # correlation = 1  -> distance = 0 (identical)
    # correlation = 0  -> distance = 1 (unrelated)
    # correlation = -1 -> distance = 2 (opposite)
    distance_matrix = 1 - tissue_corr
    condensed_dist  = squareform(distance_matrix.values)

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
    _save_or_close(output_path, dpi=dpi)
    return Z


def plot_clustered_tissue_correlation(
    tissue_corr,
    figsize=(10, 10),
    title=None,
    output_path=None,
    dpi=150,
):
    """
    Plot clustered tissue correlation.

    Note: sns.clustermap returns a ClusterGrid object, not a regular axes.
    Saving therefore uses clustermap_obj.savefig() rather than plt.savefig().

    Parameters
    ----------
    output_path : Path, optional
        If provided, the figure is saved here. If None, the figure is closed
        without saving.
    dpi : int
        Resolution for raster formats.
    """
    g = sns.clustermap(
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

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        g.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved -> {output_path}")
    plt.close("all")


# -------------------------------- VALIDATION & VALIDATION PLOTS --------------------------------

def plot_pca_validation(
    heatmap_matrix,
    remove_tissue=None,
    output_dir=None,
    dpi=150,
):
    """
    PCA QC Check - Do my extracted regions in heatmap_matrix encode useful information?

    Runs PCA on the tissues x regions matrix and plots the first four principal
    components in pairs (PC1/PC2, PC2/PC3, PC1/PC3, PC3/PC4).

    Parameters
    ----------
    heatmap_matrix : pd.DataFrame
        Tissues x regions matrix (real data or synthetic).
    remove_tissue : str, optional
        If provided, that tissue is dropped before PCA is run.
        Useful as a diagnostic -- e.g. remove_tissue="Testis" to check
        whether a single tissue is dominating PC1 ("testis vs all").
    output_dir : Path, optional
        Directory to save PCA plots into. Four files are written:
        pca_PC1_PC2.pdf, pca_PC2_PC3.pdf, pca_PC1_PC3.pdf, pca_PC3_PC4.pdf
        (with a _no_<tissue> suffix appended if remove_tissue is set).
        If None, figures are closed without saving.
    dpi : int
        Resolution for raster formats.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    matrix = heatmap_matrix.copy()

    if remove_tissue is not None:
        matrix = matrix.drop(index=remove_tissue)
        #print(f"Original shape: {heatmap_matrix.shape}")
        #print(f"No {remove_tissue} shape: {matrix.shape}")

    # convert df to numpy array
    X       = matrix.values
    tissues = matrix.index

    # try both with scaled and not scaled. We center in both cases.
    # X_centered = StandardScaler(with_std=False).fit_transform(X)
    X_scaled = StandardScaler(with_std=True).fit_transform(X)

    pca   = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)

    explained  = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print("Explained variance (first 10 PCs):")
    for i in range(10):
        print(f"PC{i+1}: {explained[i]*100:.1f}%")

    print("\nCumulative variance (first 10 PCs):")
    for i in range(10):
        print(f"PC1-PC{i+1}: {cumulative[i]*100:.1f}%")

    suffix   = f" ({remove_tissue} removed)" if remove_tissue else ""
    file_tag = f"_no_{remove_tissue}" if remove_tissue else ""

    def _pca_path(name):
        if output_dir is None:
            return None
        return output_dir / f"pca_{name}{file_tag}.pdf"

    # Plot PC1 vs PC2
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    for i, t in enumerate(tissues):
        plt.text(X_pca[i, 0], X_pca[i, 1], t, fontsize=6)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(f"PCA on atlas-selected regions{suffix}")
    plt.tight_layout()
    _save_or_close(_pca_path("PC1_PC2"), dpi=dpi)

    # plot PC2 vs PC3
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 1], X_pca[:, 2])
    for i, t in enumerate(tissues):
        plt.text(X_pca[i, 1], X_pca[i, 2], t, fontsize=9)
    plt.xlabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.ylabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
    plt.title(f"PCA on atlas-selected regions{suffix}")
    plt.tight_layout()
    _save_or_close(_pca_path("PC2_PC3"), dpi=dpi)

    # plot PC1 vs PC3
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 2])
    for i, t in enumerate(tissues):
        plt.text(X_pca[i, 0], X_pca[i, 2], t, fontsize=9)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
    plt.title(f"PCA on atlas-selected regions{suffix}")
    plt.tight_layout()
    _save_or_close(_pca_path("PC1_PC3"), dpi=dpi)

    # plot PC3 vs PC4
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 2], X_pca[:, 3])
    for i, t in enumerate(tissues):
        plt.text(X_pca[i, 2], X_pca[i, 3], t, fontsize=9)
    plt.xlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
    plt.ylabel(f"PC4 ({pca.explained_variance_ratio_[3]*100:.1f}%)")
    plt.title(f"PCA on atlas-selected regions{suffix}")
    plt.tight_layout()
    _save_or_close(_pca_path("PC3_PC4"), dpi=dpi)

# ------------------------------- END OF FUNCTION DEFINITIONS ----------------------------------
