"""
Deconvolution data preparation module.

Handles the reference/pool replicate split, signature matrix construction,
probe selection, and synthetic mixture generation for the NNLS deconvolution
benchmarking pipeline.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import config
import data_loading
from utils import (
    build_label_map,
    extract_sample_labels,
    apply_label_map,
    build_centroids,
    select_probes_from_centroids,
)


# --------------------------- FUNCTION DEFINITIONS -----------------------------

# ---- Split logic -------------------------------------------------------------

def split_reference_pool(
    original_labels: pd.Series,
    reference_fraction: float = config.DECONV_REFERENCE_FRACTION,
    random_state: int = config.DECONV_SPLIT_SEED,
) -> tuple[list, list]:
    """
    Split sample columns into reference and mixture pool sets.

    The split is stratified by original (pre-merge) tissue label: for each
    tissue, `reference_fraction` of its replicates go to the reference set
    and the rest go to the pool. Rounding is done via floor for the reference
    count, so the pool always gets at least as many as the reference.

    Parameters
    ----------
    original_labels : pd.Series
        Index = sample column names, values = original tissue labels.
        Should already have excluded tissues removed.
    reference_fraction : float
        Fraction of replicates per tissue assigned to the reference set.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    reference_cols : list
        Sample column names for the reference (atlas) set.
    pool_cols : list
        Sample column names for the mixture generation pool.
    """
    rng = np.random.RandomState(random_state)

    reference_cols = []
    pool_cols = []

    for tissue, group in original_labels.groupby(original_labels):
        n_total = len(group)
        n_ref   = max(1, int(np.floor(n_total * reference_fraction)))
        n_pool  = n_total - n_ref

        # shuffle and split
        indices = group.index.tolist()
        rng.shuffle(indices)

        reference_cols.extend(indices[:n_ref])
        pool_cols.extend(indices[n_ref:])

    # sanity checks
    assert len(set(reference_cols) & set(pool_cols)) == 0, \
        "Reference and pool sets must not overlap!"
    assert len(reference_cols) + len(pool_cols) == len(original_labels), \
        "All samples must be assigned to either reference or pool!"

    print(f"Reference samples: {len(reference_cols)}")
    print(f"Pool samples:      {len(pool_cols)}")

    return reference_cols, pool_cols


def get_pool_tissue_counts(
    pool_cols: list,
    original_labels: pd.Series,
    label_map: dict,
) -> pd.Series:
    """
    Count pool replicates per merged tissue.

    Used to determine which tissues have enough pool replicates for
    mixture generation (controlled by DECONV_MIN_POOL_REPLICATES).

    Parameters
    ----------
    pool_cols : list
        Sample column names in the mixture pool.
    original_labels : pd.Series
        Original tissue label per sample.
    label_map : dict
        Flat original -> merged label mapping.

    Returns
    -------
    pd.Series
        Counts indexed by merged tissue name, sorted ascending.
    """
    pool_labels = original_labels[pool_cols].map(lambda x: label_map.get(x, x))
    return pool_labels.value_counts().sort_values()


def get_eligible_tissues(
    pool_cols: list,
    original_labels: pd.Series,
    label_map: dict,
    min_pool_replicates: int = config.DECONV_MIN_POOL_REPLICATES,
) -> list:
    """
    Return merged tissue names eligible for mixture generation.

    A tissue is eligible if it has at least `min_pool_replicates` samples
    in the pool. Tissues below the threshold are excluded from mixture
    generation but remain in the reference atlas.

    Parameters
    ----------
    pool_cols : list
        Sample column names in the mixture pool.
    original_labels : pd.Series
        Original tissue label per sample.
    label_map : dict
        Flat original -> merged label mapping.
    min_pool_replicates : int
        Minimum pool replicates for a tissue to be eligible.

    Returns
    -------
    list
        Merged tissue names eligible for mixture generation.
    """
    counts = get_pool_tissue_counts(pool_cols, original_labels, label_map)

    eligible   = counts[counts >= min_pool_replicates].index.tolist()
    ineligible = counts[counts < min_pool_replicates].index.tolist()

    if ineligible:
        print(f"Tissues excluded from mixture generation (< {min_pool_replicates} pool replicates):")
        for t in ineligible:
            print(f"  {t}: {counts[t]} pool replicates")
    print(f"Tissues eligible for mixture generation: {len(eligible)}/{len(counts)}")

    return eligible


# ---- Split persistence -------------------------------------------------------

def save_split(
    reference_cols: list,
    pool_cols: list,
    output_path: Path,
    reference_fraction: float = config.DECONV_REFERENCE_FRACTION,
    random_state: int = config.DECONV_SPLIT_SEED,
) -> None:
    """
    Save the reference/pool split to a JSON file for reproducibility.

    The JSON includes metadata (seed, fraction) alongside the column lists
    so that the split is fully self-documenting.

    Parameters
    ----------
    reference_cols : list
        Sample column names for the reference set.
    pool_cols : list
        Sample column names for the mixture pool.
    output_path : Path
        Destination JSON file.
    reference_fraction : float
        The fraction used to generate this split.
    random_state : int
        The seed used to generate this split.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "reference_fraction": reference_fraction,
        "split_seed": random_state,
        "n_reference": len(reference_cols),
        "n_pool": len(pool_cols),
        "reference": reference_cols,
        "pool": pool_cols,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Split saved → {output_path}")


def load_split(split_path: Path) -> tuple[list, list]:
    """
    Load a previously saved reference/pool split from JSON.

    Parameters
    ----------
    split_path : Path
        Path to the split JSON file.

    Returns
    -------
    reference_cols : list
    pool_cols : list
    """
    with open(split_path) as f:
        payload = json.load(f)

    reference_cols = payload["reference"]
    pool_cols      = payload["pool"]

    print(f"Loaded split from {split_path}")
    print(f"  Reference fraction: {payload['reference_fraction']}")
    print(f"  Split seed: {payload['split_seed']}")
    print(f"  Reference: {len(reference_cols)} samples")
    print(f"  Pool: {len(pool_cols)} samples")

    return reference_cols, pool_cols


# ---- Reference matrix --------------------------------------------------------

def build_reference_matrix(
    df_raw: pd.DataFrame,
    reference_cols: list,
    original_labels: pd.Series,
    selected_probes: list,
) -> pd.DataFrame:
    """
    Build the signature matrix W from reference replicates, masked to probes.

    Returns W as a DataFrame with probe IDs as index and merged tissue
    names as columns, ready for NNLS.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Full raw replicate matrix with probe_ID as a column.
    reference_cols : list
        Sample column names in the reference set.
    original_labels : pd.Series
        Original tissue label per sample.
    selected_probes : list
        Probe IDs to mask W to.

    Returns
    -------
    pd.DataFrame
        W_masked of shape (n_probes × n_tissues). Index = probe IDs,
        columns = merged tissue names.
    """
    # build full centroid matrix from reference replicates
    df_merged_ref = build_centroids(
        df_raw=df_raw,
        sample_cols=reference_cols,
        original_labels=original_labels,
        merge_groups=data_loading.MERGE_GROUPS,
    )

    # mask to selected probes
    df_indexed = df_merged_ref.set_index("probe_ID")
    available  = [p for p in selected_probes if p in df_indexed.index]

    if len(available) < len(selected_probes):
        warnings.warn(
            f"{len(selected_probes) - len(available)} selected probes "
            f"not found in reference centroids and were skipped."
        )

    W_masked = df_indexed.loc[available]

    print(f"Reference matrix W shape: {W_masked.shape}")
    print(f"  Probes: {W_masked.shape[0]}")
    print(f"  Tissues: {W_masked.shape[1]}")

    return W_masked


def validate_reference(
    W_masked: pd.DataFrame,
) -> float:
    """
    Compute and print diagnostic statistics for the reference matrix W.

    Parameters
    ----------
    W_masked : pd.DataFrame
        Reference matrix (probes × tissues).

    Returns
    -------
    float
        Condition number of W_masked.
    """
    cond_number = np.linalg.cond(W_masked.values)
    print(f"Condition number of W: {cond_number:.1f}")

    if cond_number > 1000:
        warnings.warn(
            f"High condition number ({cond_number:.0f}) — NNLS estimates "
            f"may be numerically unstable for similar tissues."
        )

    # check for NaN or Inf
    n_nan = W_masked.isna().sum().sum()
    n_inf = np.isinf(W_masked.values).sum()
    if n_nan > 0:
        warnings.warn(f"W contains {n_nan} NaN values!")
    if n_inf > 0:
        warnings.warn(f"W contains {n_inf} Inf values!")

    # per-tissue beta value summary
    print("\nPer-tissue beta value summary (across selected probes):")
    summary = W_masked.describe().loc[["mean", "std", "min", "max"]].T
    print(summary.to_string())

    return cond_number


def correlate_with_full_atlas(
    W_masked: pd.DataFrame,
    atlas_probes_path: Path = config.DECONV_ATLAS_PROBES_PATH,
) -> pd.Series:
    """
    Compute per-tissue Pearson correlation between the reference-derived
    centroid (W_masked) and the full-atlas centroid.

    All correlations should be very high (>0.95). Low correlation for a
    tissue indicates the reference replicates may not be representative.

    Parameters
    ----------
    W_masked : pd.DataFrame
        Reference matrix (probes × tissues), index = probe IDs.
    atlas_probes_path : Path
        Path to the full-atlas heatmap matrix CSV (top_50_regions_df.csv).
        Tissues are rows (index), probes are columns.

    Returns
    -------
    pd.Series
        Per-tissue Pearson correlation, indexed by tissue name.
    """
    # load full atlas — tissues as index, probes as columns → transpose to probes × tissues
    atlas_full = pd.read_csv(atlas_probes_path, index_col=0).T

    # find shared probes
    shared_probes = W_masked.index.intersection(atlas_full.index)
    print(f"Shared probes between reference and full atlas: {len(shared_probes)}")

    if len(shared_probes) == 0:
        warnings.warn("No shared probes between W and full atlas!")
        return pd.Series(dtype=float)

    # find shared tissues
    shared_tissues = W_masked.columns.intersection(atlas_full.columns)

    correlations = {}
    for tissue in shared_tissues:
        ref_vals   = W_masked.loc[shared_probes, tissue]
        atlas_vals = atlas_full.loc[shared_probes, tissue]
        correlations[tissue] = ref_vals.corr(atlas_vals)

    corr_series = pd.Series(correlations).sort_values()

    print("\nReference vs full-atlas correlation per tissue:")
    for tissue, corr in corr_series.items():
        flag = " ⚠" if corr < 0.95 else ""
        print(f"  {tissue:30s} {corr:.4f}{flag}")

    return corr_series


# ---- Synthetic mixture generation --------------------------------------------

def _build_pool_lookup(
    df_raw: pd.DataFrame,
    pool_cols: list,
    original_labels: pd.Series,
    label_map: dict,
    selected_probes: list,
) -> dict:
    """
    Build a lookup: merged_tissue -> array of pool replicate beta vectors
    (already masked to selected probes).

    Internal helper for generate_synthetic_mixtures().

    Returns
    -------
    dict
        Keys = merged tissue names, values = np.ndarray of shape
        (n_pool_replicates, n_probes).
    """
    df_indexed = df_raw.set_index("probe_ID")

    # mask to selected probes
    available_probes = [p for p in selected_probes if p in df_indexed.index]
    df_masked = df_indexed.loc[available_probes, pool_cols]

    # map pool sample column names to merged tissue labels
    pool_labels = original_labels[pool_cols].map(lambda x: label_map.get(x, x))

    lookup = {}
    for tissue in pool_labels.unique():
        tissue_cols = pool_labels[pool_labels == tissue].index.tolist()
        lookup[tissue] = df_masked[tissue_cols].values.T   # shape (n_replicates, n_probes)

    return lookup


def generate_synthetic_mixtures(
    df_raw: pd.DataFrame,
    pool_cols: list,
    original_labels: pd.Series,
    label_map: dict,
    selected_probes: list,
    eligible_tissues: list,
    n_mixtures: int = config.DECONV_N_MIXTURES,
    k_min: int = config.DECONV_K_MIN,
    k_max: int = config.DECONV_K_MAX,
    dirichlet_alpha: float = config.DECONV_DIRICHLET_ALPHA,
    random_state: int = config.DECONV_MIXTURE_SEED,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Generate synthetic mixtures from pool replicates.

    For each mixture:
      1. Draw k tissues uniformly from eligible_tissues (k ~ Uniform(k_min, k_max)).
      2. For each tissue, draw 1 pool replicate at random.
      3. Draw proportions from Dirichlet(alpha).
      4. Compute mixture as weighted sum of the k replicate beta vectors.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Full raw replicate matrix with probe_ID as a column.
    pool_cols : list
        Sample column names in the mixture pool.
    original_labels : pd.Series
        Original tissue label per sample.
    label_map : dict
        Flat original -> merged label mapping.
    selected_probes : list
        Probe IDs defining the feature space.
    eligible_tissues : list
        Merged tissue names eligible for mixture generation.
    n_mixtures : int
        Number of synthetic mixtures to generate.
    k_min : int
        Minimum number of tissues per mixture.
    k_max : int
        Maximum number of tissues per mixture.
    dirichlet_alpha : float
        Dirichlet concentration parameter.
    random_state : int
        Random seed for mixture generation.

    Returns
    -------
    X_mixtures : np.ndarray
        Mixture beta vectors, shape (n_mixtures, n_probes).
    df_proportions : pd.DataFrame
        Ground-truth proportions, shape (n_mixtures, n_all_tissues).
        Columns are all merged tissue names (including those with 0 proportion).
        Index is mixture_0, mixture_1, etc.
    """
    rng = np.random.RandomState(random_state)

    # build lookup: tissue -> (n_replicates, n_probes) array
    pool_lookup = _build_pool_lookup(
        df_raw, pool_cols, original_labels, label_map, selected_probes
    )

    # all merged tissues (for the full proportion vector, including zeros)
    all_tissues = sorted(set(
        original_labels.map(lambda x: label_map.get(x, x)).unique()
    ))

    n_probes = len([p for p in selected_probes
                    if p in df_raw.set_index("probe_ID").index])

    X_mixtures   = np.zeros((n_mixtures, n_probes))
    proportions  = np.zeros((n_mixtures, len(all_tissues)))

    tissue_to_idx = {t: i for i, t in enumerate(all_tissues)}

    for mix_i in range(n_mixtures):
        # draw number of tissues
        k = rng.randint(k_min, k_max + 1)

        # draw which tissues
        chosen_tissues = rng.choice(eligible_tissues, size=k, replace=False).tolist()

        # draw proportions from Dirichlet
        alpha_vec = np.full(k, dirichlet_alpha)
        props     = rng.dirichlet(alpha_vec)

        # build mixture vector
        mixture_vec = np.zeros(n_probes)
        for j, tissue in enumerate(chosen_tissues):
            # draw 1 replicate from this tissue's pool
            n_available  = pool_lookup[tissue].shape[0]
            rep_idx      = rng.randint(0, n_available)
            replicate    = pool_lookup[tissue][rep_idx]

            mixture_vec += props[j] * replicate

            # record proportion in full vector
            proportions[mix_i, tissue_to_idx[tissue]] = props[j]

        X_mixtures[mix_i] = mixture_vec

    # assemble proportions dataframe
    df_proportions = pd.DataFrame(
        proportions,
        columns=all_tissues,
        index=[f"mixture_{i}" for i in range(n_mixtures)],
    )

    print(f"Generated {n_mixtures} synthetic mixtures")
    print(f"  Probes per mixture: {n_probes}")
    print(f"  Tissues per mixture: {k_min}–{k_max}")
    print(f"  Dirichlet alpha: {dirichlet_alpha}")
    print(f"  Eligible tissues: {len(eligible_tissues)}")

    return X_mixtures, df_proportions


# ---- Mixture validation ------------------------------------------------------

def validate_mixtures(
    X_mixtures: np.ndarray,
    df_proportions: pd.DataFrame,
) -> None:
    """
    Run quality checks on generated synthetic mixtures.

    Checks:
    1. All beta values are in [0, 1]
    2. Proportions sum to 1 for each mixture
    3. No NaN or Inf values
    4. Print summary statistics

    Parameters
    ----------
    X_mixtures : np.ndarray
        Mixture beta vectors, shape (n_mixtures, n_probes).
    df_proportions : pd.DataFrame
        Ground-truth proportions, shape (n_mixtures, n_tissues).
    """
    n_mixtures, n_probes = X_mixtures.shape

    # check beta value range
    beta_min = X_mixtures.min()
    beta_max = X_mixtures.max()
    assert beta_min >= 0.0 and beta_max <= 1.0, \
        f"Beta values out of range: [{beta_min:.4f}, {beta_max:.4f}]"
    print(f"Beta value range: [{beta_min:.4f}, {beta_max:.4f}] — OK")

    # check for NaN / Inf
    n_nan = np.isnan(X_mixtures).sum()
    n_inf = np.isinf(X_mixtures).sum()
    assert n_nan == 0, f"Mixtures contain {n_nan} NaN values!"
    assert n_inf == 0, f"Mixtures contain {n_inf} Inf values!"
    print("No NaN or Inf values — OK")

    # check proportions sum to 1
    prop_sums = df_proportions.sum(axis=1)
    max_deviation = (prop_sums - 1.0).abs().max()
    assert max_deviation < 1e-10, \
        f"Proportions do not sum to 1: max deviation = {max_deviation}"
    print(f"Proportion sums: max deviation from 1.0 = {max_deviation:.2e} — OK")

    # number of tissues per mixture
    n_components = (df_proportions > 0).sum(axis=1)
    print(f"\nComponents per mixture: min={n_components.min()}, "
          f"max={n_components.max()}, "
          f"mean={n_components.mean():.1f}")

    # proportion distribution summary (non-zero only)
    nonzero_props = df_proportions.values[df_proportions.values > 0]
    print(f"\nNon-zero proportion summary:")
    print(f"  min:    {nonzero_props.min():.4f}")
    print(f"  max:    {nonzero_props.max():.4f}")
    print(f"  mean:   {nonzero_props.mean():.4f}")
    print(f"  median: {np.median(nonzero_props):.4f}")

    # how often each tissue appears in mixtures
    tissue_freq = (df_proportions > 0).sum(axis=0).sort_values(ascending=False)
    print(f"\nTissue frequency in mixtures (top 5):")
    for tissue, freq in tissue_freq.head(5).items():
        print(f"  {tissue:30s} {freq:4d} ({freq/n_mixtures*100:.1f}%)")


# ---- Convenience loader ------------------------------------------------------

def load_deconv_data(
    reference_fraction: float = config.DECONV_REFERENCE_FRACTION,
    split_seed: int = config.DECONV_SPLIT_SEED,
) -> tuple:
    """
    Full data preparation sequence for the deconvolution pipeline.

    Loads df_raw, extracts labels, removes excluded tissues, splits into
    reference and pool sets, and saves the split to disk.

    Parameters
    ----------
    reference_fraction : float
        Fraction of replicates per tissue assigned to the reference set.
    split_seed : int
        Random seed for the reference/pool split.

    Returns
    -------
    df_raw : pd.DataFrame
        Full raw replicate matrix (probes × samples).
    reference_cols : list
        Sample column names for the reference (atlas) set.
    pool_cols : list
        Sample column names for the mixture generation pool.
    original_labels : pd.Series
        Original tissue label per non-excluded sample column.
    label_map : dict
        Flat original_label -> merged_label mapping.
    """
    # load raw replicate beta matrix
    df_raw, _, _, _ = data_loading.load_all()
    print(f"df_raw shape (probes × samples): {df_raw.shape}")

    # build label map
    label_map = build_label_map(data_loading.MERGE_GROUPS)

    # extract original tissue labels from column names
    original_labels_all = extract_sample_labels(df_raw)

    # remove excluded tissues
    _, _ = apply_label_map(
        original_labels_all,
        label_map,
        excluded_tissues=data_loading.EXCLUDED_TISSUES,
    )
    keep_mask       = ~original_labels_all.isin(data_loading.EXCLUDED_TISSUES)
    original_labels = original_labels_all[keep_mask]

    # split into reference and pool
    reference_cols, pool_cols = split_reference_pool(
        original_labels,
        reference_fraction=reference_fraction,
        random_state=split_seed,
    )

    # save split to disk
    split_path = config.DECONV_OUTPUT_DIR / "split.json"
    save_split(
        reference_cols, pool_cols, split_path,
        reference_fraction=reference_fraction,
        random_state=split_seed,
    )

    # print pool tissue counts
    print("\nPool replicates per merged tissue:")
    counts = get_pool_tissue_counts(pool_cols, original_labels, label_map)
    print(counts.to_string())

    return df_raw, reference_cols, pool_cols, original_labels, label_map

# ------------------------------- END OF FUNCTION DEFINITIONS ----------------------------------
