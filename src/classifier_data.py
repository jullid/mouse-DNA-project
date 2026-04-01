import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

import config
import data_loading
import diff_analysis

# Shared utilities — imported and re-exported for backward compatibility.
# classifier_model.py and run_classifier.py can continue to import from here.
from utils import (
    build_label_map,
    extract_sample_labels,
    apply_label_map,
    build_centroids as build_fold_centroids,   # alias preserves existing call sites
    select_probes_from_centroids,
    build_X_from_probes,
)


# --------------------------- FUNCTION DEFINITIONS -----------------------------

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
