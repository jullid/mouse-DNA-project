import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

import config
import data_loading
import diff_analysis


# --------------------------- FUNCTION DEFINITIONS -----------------------------

def build_label_map(merge_groups: dict) -> dict:
    """
    Build a flat original_label -> merged_label lookup from a merge_groups dict.

    Tissues not present in any merge group are expected to map to themselves;
    this is handled implicitly downstream via dict.get(x, x).

    Parameters
    ----------
    merge_groups : dict
        Keys are merged tissue names; values are lists of original tissue names.
        Should be imported from data_loading.MERGE_GROUPS to stay in sync
        with the pipeline.

    Returns
    -------
    dict
        Flat mapping of original_label -> merged_label.
    """
    label_map = {}
    for merged_label, original_labels in merge_groups.items():
        for orig in original_labels:
            label_map[orig] = merged_label
    return label_map


def extract_sample_labels(df_raw: pd.DataFrame) -> pd.Series:
    """
    Extract the original tissue label for each sample from df_raw column names.

    Column names follow the pattern <TissueType>.<replicate_n> (pandas suffix
    added on read). The tissue label is recovered by splitting on '.' and
    taking the left side.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw replicate beta matrix with probe_ID as a column (not index).
        Columns are sample IDs of the form <TissueType>.<n>.

    Returns
    -------
    pd.Series
        Index = sample column names, values = original tissue labels.
    """
    # Column names follow the pattern <TissueType>.<replicate_n> (pandas suffix on read).
    # Strip the suffix to recover the original tissue label per sample.
    sample_cols = df_raw.columns[df_raw.columns != "probe_ID"].tolist()

    original_labels = pd.Series(
        [col.split(".")[0] for col in sample_cols],
        index=sample_cols,
        name="original_tissue",
    )
    return original_labels


def apply_label_map(
    original_labels: pd.Series,
    label_map: dict,
    excluded_tissues: set,
) -> tuple[pd.Series, pd.Series]:
    """
    Map original tissue labels to merged labels and remove excluded samples.

    Parameters
    ----------
    original_labels : pd.Series
        Index = sample column names, values = original tissue labels.
        Returned by extract_sample_labels().
    label_map : dict
        Flat original_label -> merged_label mapping.
        Built by build_label_map() from data_loading.MERGE_GROUPS.
    excluded_tissues : set
        Original tissue labels whose samples should be dropped entirely.
        Should be imported from data_loading.EXCLUDED_TISSUES.

    Returns
    -------
    sample_cols_use : pd.Series
        Sample column names after exclusion (index and values are both sample IDs).
    merged_labels : pd.Series
        Merged tissue label per retained sample.
    """
    # map original labels to merged labels; tissues not in label_map map to themselves
    merged_labels_full = original_labels.map(lambda x: label_map.get(x, x))
    merged_labels_full.name = "merged_tissue"

    # exclude samples whose original label is in excluded_tissues
    keep_mask       = ~original_labels.isin(excluded_tissues)
    sample_cols_use = original_labels[keep_mask].index.to_series()
    merged_labels   = merged_labels_full[keep_mask]

    n_excluded = (~keep_mask).sum()
    print(f"Excluded {n_excluded} samples from: {excluded_tissues}")
    print(f"Remaining samples: {len(sample_cols_use)}")
    print()
    print("Sample counts per merged tissue class:")
    print(merged_labels.value_counts().sort_index())

    return sample_cols_use, merged_labels


def split_train_test(
    original_labels: pd.Series,
    random_state: int = 42,
) -> tuple[list, list]:
    """
    Hold out exactly 1 sample per original tissue type as a fixed test set.

    Splitting is done at the original (pre-merge) tissue label level so that
    every original tissue contributes one test sample regardless of whether
    it belongs to a merged group. This gives ~26 test samples (one per
    non-excluded original tissue).

    Splitting happens before any centroid computation or probe selection so
    that the test set is never seen during feature selection or training.

    Parameters
    ----------
    original_labels : pd.Series
        Index = sample column names, values = original tissue labels.
        Should already have excluded tissues removed.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    train_cols : list
        Sample column names for the training set.
    test_cols : list
        Sample column names for the test set (1 per original tissue).
    """
    # pick exactly 1 sample per original tissue class for the test set
    test_cols = (
        original_labels
        .groupby(original_labels)
        .apply(lambda grp: grp.sample(n=1, random_state=random_state))
        .index.get_level_values(1)   # get sample column names (not tissue names)
        .tolist()
    )

    train_cols = [c for c in original_labels.index if c not in test_cols]

    print(f"Train samples: {len(train_cols)}")
    print(f"Test samples:  {len(test_cols)}  (1 per original tissue)")

    # sanity check: 1 per class in test set
    test_tissue_counts = original_labels[test_cols].value_counts()
    assert (test_tissue_counts == 1).all(), "Test set should have exactly 1 sample per tissue!"

    return train_cols, test_cols


def build_fold_centroids(
    df_raw: pd.DataFrame,
    sample_cols: list,
    original_labels: pd.Series,
    merge_groups: dict,
) -> pd.DataFrame:
    """
    Compute a merged tissue centroid matrix from a subset of samples.

    Used inside the CV loop to build a fold-specific centroid matrix from
    training samples only — ensuring probe selection at each fold has no
    access to held-out validation samples.

    The output format matches df_merged from data_loading.build_merged_df():
    probe_ID as a column, one column per merged tissue.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Full raw replicate matrix with probe_ID as a column.
    sample_cols : list
        Sample column names to include (e.g. fold training samples).
    original_labels : pd.Series
        Index = sample column names, values = original tissue labels.
    merge_groups : dict
        Keys = merged tissue names, values = lists of original tissue names.
        Should be data_loading.MERGE_GROUPS.

    Returns
    -------
    pd.DataFrame
        Centroid matrix with probe_ID column + one column per merged tissue,
        built only from the supplied sample_cols.
    """
    # subset to the supplied samples and index by probe_ID
    df_subset = df_raw.set_index("probe_ID")[sample_cols]

    # get original tissue label for each sample in this subset
    fold_orig_labels = original_labels[sample_cols]

    # group columns by original tissue label -> mean centroid per original tissue
    # .T groups columns (samples) by tissue label, .mean() averages, .T restores orientation
    # result: index=probe_IDs, columns=original tissue names present in sample_cols
    df_orig_centroids = df_subset.T.groupby(fold_orig_labels.values).mean().T

    # build merged centroid dataframe
    df_merged_fold = pd.DataFrame(index=df_orig_centroids.index)

    # add merged tissue columns (average of constituent original tissues present in subset)
    for new_name, orig_cols in merge_groups.items():
        available = [c for c in orig_cols if c in df_orig_centroids.columns]
        if available:
            df_merged_fold[new_name] = df_orig_centroids[available].mean(axis=1)

    # keep unmerged tissue columns as-is
    merged_flat = {c for cols in merge_groups.values() for c in cols}
    for c in df_orig_centroids.columns:
        if c not in merged_flat:
            df_merged_fold[c] = df_orig_centroids[c]

    # reset index: restores probe_ID as a column (format expected by diff_analysis functions)
    df_merged_fold = df_merged_fold.reset_index()

    return df_merged_fold


def select_probes_from_centroids(
    df_merged_fold: pd.DataFrame,
    top_n: int = config.TOP_N,
    use_filtering: bool = config.USE_FILTERING,
    max_per_tissue: int = config.MAX_PER_TISSUE,
    region_mode: str = config.REGION_MODE,
    suppress_warnings: bool = False,
) -> list:
    """
    Run the full differential methylation probe selection pipeline on a
    centroid matrix and return the selected probe IDs.

    This is the core of the leakage-free design: called inside the CV loop
    with fold-training centroids only, so the validation fold never influences
    which probes are selected.

    Uses the same parameters as the atlas pipeline (TOP_N, USE_FILTERING,
    MAX_PER_TISSUE, REGION_MODE from config) so that fold-level selection is
    consistent with the full-data atlas selection.

    Parameters
    ----------
    df_merged_fold : pd.DataFrame
        Centroid matrix returned by build_fold_centroids().
    suppress_warnings : bool
        If True, suppress UserWarnings from create_heatmap_matrix (useful
        inside the CV loop where small fold sizes routinely trigger the
        max_per_tissue warning).

    Returns
    -------
    list
        Selected probe IDs (columns of the resulting heatmap_matrix).
    """
    all_tissues_results = diff_analysis.build_tissue_diff_table(
        df=df_merged_fold,
        use_filtering=use_filtering,
    )

    df_top_regions = diff_analysis.extract_top_regions(
        all_tissues_results,
        top_n=top_n,
    )

    with warnings.catch_warnings():
        if suppress_warnings:
            warnings.simplefilter("ignore", UserWarning)
        heatmap_matrix_fold, _ = diff_analysis.create_heatmap_matrix(
            df=df_merged_fold,
            df_top_regions=df_top_regions,
            region_mode=region_mode,
            max_per_tissue=max_per_tissue,
            verbose=False,
        )

    return heatmap_matrix_fold.columns.tolist()


def build_X_from_probes(
    df_raw: pd.DataFrame,
    sample_cols: list,
    selected_probes: list,
) -> pd.DataFrame:
    """
    Build a (samples x probes) feature matrix from a subset of samples and probes.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Full raw replicate matrix with probe_ID as a column.
    sample_cols : list
        Sample column names to include.
    selected_probes : list
        Probe IDs to use as features.

    Returns
    -------
    pd.DataFrame
        Feature matrix of shape (len(sample_cols) x len(selected_probes)).
    """
    df_indexed = df_raw.set_index("probe_ID")

    # guard against any selected probe missing from df_raw (should not happen
    # after annotation filtering, but defensive check is cheap)
    available_probes = [p for p in selected_probes if p in df_indexed.index]
    if len(available_probes) < len(selected_probes):
        warnings.warn(
            f"{len(selected_probes) - len(available_probes)} selected probes "
            f"not found in df_raw and were skipped."
        )

    return df_indexed.loc[available_probes, sample_cols].T


def load_classifier_data(
    random_state: int = 42,
) -> tuple:
    """
    Full data preparation sequence for the leakage-free classifier pipeline.

    Loads df_raw, extracts and maps sample labels, removes excluded tissues,
    and performs a stratified train/test split (1 sample per original tissue
    held out as a fixed test set). Does NOT build X or select probes here —
    those steps happen inside the CV loop and final evaluation to prevent
    feature-selection leakage.

    Parameters
    ----------
    random_state : int
        Passed to split_train_test() for reproducibility.

    Returns
    -------
    df_raw : pd.DataFrame
        Full raw replicate matrix (probes x samples).
    train_cols : list
        Sample column names for the training set.
    test_cols : list
        Sample column names for the test set.
    original_labels : pd.Series
        Original tissue label per non-excluded sample column.
    label_map : dict
        Flat original_label -> merged_label mapping.
    le : LabelEncoder
        Fitted on merged training labels. Use le.classes_ to recover tissue names.
    merged_labels_train : pd.Series
        Merged tissue label per training sample (used for CV feasibility checks).
    """
    # load raw replicate beta matrix (probes x samples)
    df_raw, _, _, _ = data_loading.load_all()
    print(f"df_raw shape (probes x samples): {df_raw.shape}")

    # build label map from module-level constants (single source of truth)
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

    # stratified train/test split: 1 sample per original tissue held out
    train_cols, test_cols = split_train_test(original_labels, random_state=random_state)

    # merged labels for training samples (used by check_cv_feasibility and CV loop)
    merged_labels_train = original_labels[train_cols].map(lambda x: label_map.get(x, x))
    merged_labels_train.name = "merged_tissue"

    print("\nTrain set class counts (merged labels):")
    print(merged_labels_train.value_counts().sort_index())

    # fit label encoder on all training merged labels — consistent encoding across folds
    le = LabelEncoder()
    le.fit(merged_labels_train)

    print("\nClass encoding:")
    for i, cls in enumerate(le.classes_):
        print(f"  {i:2d}  {cls}")

    return df_raw, train_cols, test_cols, original_labels, label_map, le, merged_labels_train

# ------------------------------- END OF FUNCTION DEFINITIONS ----------------------------------
