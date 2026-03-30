import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required on headless HPC nodes
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder

import config
import classifier_data


# --------------------------- FUNCTION DEFINITIONS -----------------------------

def check_cv_feasibility(
    merged_labels: pd.Series,
    n_folds: int,
) -> int:
    """
    Warn if any class has fewer samples than n_folds and suggest a safe fold count.

    Parameters
    ----------
    merged_labels : pd.Series
        Merged tissue label per training sample.
    n_folds : int
        Requested number of CV folds.

    Returns
    -------
    int
        Safe number of folds to use (may be lower than n_folds if classes are small).
    """
    class_counts   = merged_labels.value_counts()
    min_class_size = class_counts.min()

    if min_class_size < n_folds:
        safe_folds = int(min_class_size)
        warnings.warn(
            f"Smallest class has only {min_class_size} samples, which is less than "
            f"n_folds={n_folds}. Reducing to {safe_folds} folds to avoid empty "
            f"validation splits.\nSmall classes:\n"
            f"{class_counts[class_counts < n_folds].to_string()}"
        )
        return safe_folds

    print(f"All classes have >= {n_folds} samples. Stratified {n_folds}-fold CV is safe.")
    return n_folds


def build_cv(n_folds: int, random_seed: int) -> StratifiedKFold:
    """
    Build a StratifiedKFold cross-validator.

    Parameters
    ----------
    n_folds : int
        Number of CV folds. Should be the value returned by check_cv_feasibility().
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    StratifiedKFold
    """
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)


def verify_fold_distributions(
    cv: StratifiedKFold,
    y: np.ndarray,
) -> None:
    """
    Print per-fold sample counts as a sanity check for stratification.

    Note: takes y directly rather than (X, y) since in the leakage-free
    pipeline X is not pre-built — it is constructed inside the CV loop
    per fold after probe selection.

    Parameters
    ----------
    cv : StratifiedKFold
    y : np.ndarray
        Integer-encoded labels for the training set.
    """
    # StratifiedKFold only uses y for splitting; a dummy index array suffices for X
    dummy_X = np.arange(len(y))

    print(f"Fold class distribution check ({cv.n_splits} folds):")
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(dummy_X, y)):
        val_counts = pd.Series(y[val_idx]).value_counts().sort_index()
        print(
            f"  Fold {fold_idx + 1}: train={len(train_idx)} samples, "
            f"val={len(val_idx)} samples, "
            f"min class in val={val_counts.min()}"
        )


def run_cv_with_probe_selection(
    model: LogisticRegression,
    df_raw: pd.DataFrame,
    train_cols: list,
    original_labels: pd.Series,
    label_map: dict,
    le: LabelEncoder,
    cv: StratifiedKFold,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Stratified k-fold CV with probe selection performed inside each fold.

    This is the leakage-free CV loop. At each fold:
      1. Tissue centroids are computed from the fold's training samples only.
      2. Differential methylation scoring selects probes from those centroids.
      3. The model is trained on the fold-training samples masked to those probes.
      4. The fold-validation samples are masked to the same probes and predicted.

    Probe sets differ slightly fold-to-fold (expected and correct). They are
    transient — never written to disk — so top_50_regions_df.csv is untouched.

    Parameters
    ----------
    model : LogisticRegression
        Unfitted sklearn estimator.
    df_raw : pd.DataFrame
        Full raw replicate matrix (probes x samples).
    train_cols : list
        Training sample column names (post train/test split).
    original_labels : pd.Series
        Original tissue label per non-excluded sample column.
    label_map : dict
        Flat original_label -> merged_label mapping.
    le : LabelEncoder
        Fitted on all merged training labels for consistent encoding.
    cv : StratifiedKFold

    Returns
    -------
    all_y_true : np.ndarray
        Out-of-fold true labels (integer-encoded).
    all_y_pred : np.ndarray
        Out-of-fold predicted labels (integer-encoded).
    fold_metrics : pd.DataFrame
        Per-fold balanced accuracy, macro F1, and probe count.
    """
    train_cols_arr = np.array(train_cols)

    # integer-encode training labels for CV splitting
    y_train = le.transform(
        original_labels[train_cols].map(lambda x: label_map.get(x, x))
    )

    all_y_true   = []
    all_y_pred   = []
    fold_records = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(train_cols_arr, y_train)):

        fold_train_cols = train_cols_arr[train_idx].tolist()
        fold_val_cols   = train_cols_arr[val_idx].tolist()

        # Step 1: build centroids from fold training samples only
        df_merged_fold = classifier_data.build_fold_centroids(
            df_raw=df_raw,
            sample_cols=fold_train_cols,
            original_labels=original_labels,
            merge_groups=data_loading_merge_groups(),
        )

        # Step 2: select probes from fold training centroids (transient — not saved)
        selected_probes = classifier_data.select_probes_from_centroids(
            df_merged_fold,
            suppress_warnings=True,   # suppress max_per_tissue warnings in the loop
        )

        # Step 3: build feature matrices for this fold (samples x selected_probes)
        X_fold_train = classifier_data.build_X_from_probes(
            df_raw, fold_train_cols, selected_probes
        )
        X_fold_val = classifier_data.build_X_from_probes(
            df_raw, fold_val_cols, selected_probes
        )

        # Step 4: encode labels for this fold
        y_fold_train = le.transform(
            original_labels[fold_train_cols].map(lambda x: label_map.get(x, x))
        )
        y_fold_val = le.transform(
            original_labels[fold_val_cols].map(lambda x: label_map.get(x, x))
        )

        # Step 5: fit and predict
        model.fit(X_fold_train.values, y_fold_train)
        y_pred = model.predict(X_fold_val.values)

        # accumulate predictions
        all_y_true.extend(y_fold_val)
        all_y_pred.extend(y_pred)

        # per-fold scalar metrics
        bal_acc = balanced_accuracy_score(y_fold_val, y_pred)
        mac_f1  = f1_score(y_fold_val, y_pred, average="macro", zero_division=0)
        fold_records.append({
            "fold":               fold_idx + 1,
            "n_probes_selected":  len(selected_probes),
            "balanced_accuracy":  bal_acc,
            "macro_f1":           mac_f1,
        })
        print(
            f"  Fold {fold_idx + 1}: "
            f"balanced_acc={bal_acc:.3f}, "
            f"macro_f1={mac_f1:.3f}, "
            f"n_probes={len(selected_probes)}"
        )

    fold_metrics = pd.DataFrame(fold_records)

    print()
    print("CV summary (mean +/- std across folds):")
    print(f"  Balanced accuracy : {fold_metrics['balanced_accuracy'].mean():.3f} "
          f"(+/- {fold_metrics['balanced_accuracy'].std():.3f})")
    print(f"  Macro F1          : {fold_metrics['macro_f1'].mean():.3f} "
          f"(+/- {fold_metrics['macro_f1'].std():.3f})")
    print(f"  Probes selected   : {fold_metrics['n_probes_selected'].mean():.0f} "
          f"(+/- {fold_metrics['n_probes_selected'].std():.0f})")

    return np.array(all_y_true), np.array(all_y_pred), fold_metrics


def final_evaluation(
    model: LogisticRegression,
    df_raw: pd.DataFrame,
    train_cols: list,
    test_cols: list,
    original_labels: pd.Series,
    label_map: dict,
    le: LabelEncoder,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Fit a final model on all training data and evaluate on the held-out test set.

    Probe selection is performed on centroids built from all training samples —
    the test set is never used in probe selection or training.

    Parameters
    ----------
    model : LogisticRegression
        Unfitted sklearn estimator.
    df_raw : pd.DataFrame
        Full raw replicate matrix.
    train_cols : list
        All training sample column names.
    test_cols : list
        Held-out test sample column names (1 per original tissue).
    original_labels : pd.Series
        Original tissue label per non-excluded sample column.
    label_map : dict
        Flat original_label -> merged_label mapping.
    le : LabelEncoder
        Fitted label encoder.

    Returns
    -------
    y_test : np.ndarray
        True labels for the test set (integer-encoded).
    y_pred_test : np.ndarray
        Predicted labels for the test set (integer-encoded).
    selected_probes : list
        Probe IDs selected from the full training centroid (for reference).
    """
    print("Building final model on all training data...")

    # build centroids from all training samples
    df_merged_train = classifier_data.build_fold_centroids(
        df_raw=df_raw,
        sample_cols=train_cols,
        original_labels=original_labels,
        merge_groups=data_loading_merge_groups(),
    )

    # select probes from full training centroids
    selected_probes = classifier_data.select_probes_from_centroids(df_merged_train)
    print(f"Probes selected from full training centroids: {len(selected_probes)}")

    # build feature matrices
    X_train = classifier_data.build_X_from_probes(df_raw, train_cols, selected_probes)
    X_test  = classifier_data.build_X_from_probes(df_raw, test_cols, selected_probes)

    # encode labels
    y_train = le.transform(
        original_labels[train_cols].map(lambda x: label_map.get(x, x))
    )
    y_test = le.transform(
        original_labels[test_cols].map(lambda x: label_map.get(x, x))
    )

    # fit final model and predict on test set
    model.fit(X_train.values, y_train)
    y_pred_test = model.predict(X_test.values)

    return y_test, y_pred_test, selected_probes


def data_loading_merge_groups() -> dict:
    """
    Thin wrapper to import MERGE_GROUPS from data_loading without a
    top-level circular-import risk.  Called at runtime inside CV functions.
    """
    import data_loading as _dl
    return _dl.MERGE_GROUPS


def report_per_class_metrics(
    all_y_true: np.ndarray,
    all_y_pred: np.ndarray,
    le: LabelEncoder,
    header: str = "Classification report:",
) -> None:
    """
    Print a full sklearn classification report.

    Parameters
    ----------
    header : str
        Label printed above the report — use to distinguish CV vs test reports.
    """
    print(header)
    print(classification_report(
        all_y_true,
        all_y_pred,
        target_names=le.classes_,
        zero_division=0,
    ))


def plot_confusion_matrix(
    all_y_true: np.ndarray,
    all_y_pred: np.ndarray,
    le: LabelEncoder,
    title: str,
    output_path: Path = None,
    dpi: int = 150,
) -> None:
    """
    Plot and save a row-normalised confusion matrix.

    Row normalisation (normalize='true') shows recall per true class, which
    is appropriate for imbalanced data — raw counts would visually overrepresent
    large classes.

    Parameters
    ----------
    title : str
        Figure title — use to distinguish CV vs test confusion matrices.
    output_path : Path, optional
        If provided, figure is saved here. If None, figure is closed without saving.
    dpi : int
        Resolution for raster formats.
    """
    # confusion matrix -- normalised by true class (row) to account for imbalance
    cm    = confusion_matrix(all_y_true, all_y_pred, normalize="true")
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_df,
        cmap="Blues",
        vmin=0, vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved -> {output_path}")
    plt.close("all")


def plot_per_class_recall(
    all_y_true: np.ndarray,
    all_y_pred: np.ndarray,
    le: LabelEncoder,
    merged_labels: pd.Series,
    title: str,
    output_path: Path = None,
    dpi: int = 150,
) -> None:
    """
    Plot per-class recall as a horizontal bar chart.

    Bars are annotated with sample count (n) to contextualise low recall
    for small classes. A vertical dashed line marks mean recall.

    Parameters
    ----------
    merged_labels : pd.Series
        Merged tissue label per sample — used to retrieve per-class sample counts.
    title : str
        Figure title — use to distinguish CV vs test recall plots.
    output_path : Path, optional
        If provided, figure is saved here. If None, figure is closed without saving.
    dpi : int
        Resolution for raster formats.
    """
    cm               = confusion_matrix(all_y_true, all_y_pred, normalize="true")
    per_class_recall = cm.diagonal()
    class_counts     = merged_labels.value_counts()

    recall_df = pd.DataFrame({
        "tissue":    le.classes_,
        "recall":    per_class_recall,
        "n_samples": [class_counts.get(cls, 0) for cls in le.classes_],
    }).sort_values("recall")

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(recall_df["tissue"], recall_df["recall"], color="steelblue")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.axvline(
        x=recall_df["recall"].mean(),
        color="red", linestyle="--", label="Mean recall"
    )
    ax.legend()

    # annotate bars with sample count
    for bar, n in zip(bars, recall_df["n_samples"]):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"n={n}",
            va="center", fontsize=8,
        )

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved -> {output_path}")
    plt.close("all")


def print_summary(
    train_cols: list,
    test_cols: list,
    le: LabelEncoder,
    fold_metrics: pd.DataFrame,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    n_folds: int,
    figures_dir: Path,
) -> None:
    """Print a concise end-of-run summary of CV and test set results."""
    print("===== SUMMARY =====")
    print(f"Training samples:     {len(train_cols)}")
    print(f"Test samples:         {len(test_cols)}")
    print(f"Classes:              {len(le.classes_)}")
    print(f"CV folds:             {n_folds}")
    print()
    print("CV results (out-of-fold, probe selection inside loop):")
    print(f"  Balanced accuracy : {fold_metrics['balanced_accuracy'].mean():.3f} "
          f"(+/- {fold_metrics['balanced_accuracy'].std():.3f})")
    print(f"  Macro F1          : {fold_metrics['macro_f1'].mean():.3f} "
          f"(+/- {fold_metrics['macro_f1'].std():.3f})")
    print(f"  Probes selected   : {fold_metrics['n_probes_selected'].mean():.0f} avg per fold")
    print()
    print("Final test set results (1 sample per original tissue):")
    print(f"  Balanced accuracy : {balanced_accuracy_score(y_test, y_pred_test):.3f}")
    print(f"  Macro F1          : {f1_score(y_test, y_pred_test, average='macro', zero_division=0):.3f}")
    print()
    print("Figures written to:", figures_dir)

# ------------------------------- END OF FUNCTION DEFINITIONS ----------------------------------
