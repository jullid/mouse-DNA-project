from sklearn.linear_model import LogisticRegression

import config
import classifier_data
import classifier_model


def main():

    # ---------------------------------- Configuration ------------------------------------
    N_FOLDS     = 5     # reduced automatically by check_cv_feasibility if classes are too small
    RANDOM_SEED = 42

    FIGURES_DIR = config.FIGURES_DIR / "classifier"
    FMT         = config.FIGURE_FORMAT
    DPI         = config.FIGURE_DPI
    # -------------------------------------------------------------------------------------


    # ---------------------------------- Data preparation ---------------------------------
    # Loads df_raw, extracts labels, removes excluded tissues, performs train/test split.
    # Probe selection is NOT done here — it happens inside the CV loop and final evaluation
    # to prevent feature-selection leakage.
    (
        df_raw,
        train_cols,
        test_cols,
        original_labels,
        label_map,
        le,
        merged_labels_train,
    ) = classifier_data.load_classifier_data(random_state=RANDOM_SEED)
    # -------------------------------------------------------------------------------------


    # ---------------------------------- CV setup -----------------------------------------
    # Reduce n_folds automatically if any class has fewer samples than requested
    n_folds = classifier_model.check_cv_feasibility(merged_labels_train, N_FOLDS)

    cv = classifier_model.build_cv(n_folds, RANDOM_SEED)

    # verify stratification using training labels only (X not yet built — probe selection
    # is inside the loop, so we pass y directly)
    y_train_for_check = le.transform(merged_labels_train)
    classifier_model.verify_fold_distributions(cv, y_train_for_check)
    # -------------------------------------------------------------------------------------


    # ---------------------------------- Model definition ---------------------------------
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",   # compensates for imbalanced class sizes
        random_state=RANDOM_SEED,
        C=1.0,                     # inverse regularisation strength; tune later
    )
    # -------------------------------------------------------------------------------------


    # ---------------------------------- CV with probe selection --------------------------
    # Probe selection (diff scoring -> top regions -> create_heatmap_matrix) runs inside
    # each fold using fold-training centroids only. Probe sets are transient and never
    # written to disk — top_50_regions_df.csv is not touched.
    print("\nRunning cross-validation with per-fold probe selection...")
    all_y_true, all_y_pred, fold_metrics = classifier_model.run_cv_with_probe_selection(
        model=model,
        df_raw=df_raw,
        train_cols=train_cols,
        original_labels=original_labels,
        label_map=label_map,
        le=le,
        cv=cv,
    )
    # -------------------------------------------------------------------------------------


    # ---------------------------------- CV metrics & plots -------------------------------
    # Per-class classification report aggregated across all folds
    classifier_model.report_per_class_metrics(
        all_y_true, all_y_pred, le,
        header="CV classification report (out-of-fold, aggregated):",
    )

    # Normalised confusion matrix — CV
    classifier_model.plot_confusion_matrix(
        all_y_true, all_y_pred, le,
        title=f"Normalised confusion matrix — CV (stratified {n_folds}-fold, probe selection inside loop)",
        output_path=FIGURES_DIR / f"cv_confusion_matrix.{FMT}",
        dpi=DPI,
    )

    # Per-class recall bar chart — CV
    classifier_model.plot_per_class_recall(
        all_y_true, all_y_pred, le,
        merged_labels=merged_labels_train,
        title="Per-class recall — CV (logistic regression, probe selection inside loop)",
        output_path=FIGURES_DIR / f"cv_per_class_recall.{FMT}",
        dpi=DPI,
    )
    # -------------------------------------------------------------------------------------


    # ---------------------------------- Final test evaluation ----------------------------
    # Fit on all training data (probe selection from full training centroids),
    # evaluate on the held-out test set (1 sample per original tissue).
    print("\nRunning final evaluation on held-out test set...")
    y_test, y_pred_test, selected_probes = classifier_model.final_evaluation(
        model=model,
        df_raw=df_raw,
        train_cols=train_cols,
        test_cols=test_cols,
        original_labels=original_labels,
        label_map=label_map,
        le=le,
    )

    # Test set report — note: 1 sample per class means per-class recall is 0 or 1
    merged_labels_test = original_labels[test_cols].map(lambda x: label_map.get(x, x))
    merged_labels_test.name = "merged_tissue"

    classifier_model.report_per_class_metrics(
        y_test, y_pred_test, le,
        header="Test set classification report (1 sample per original tissue):",
    )

    # Confusion matrix — test set
    classifier_model.plot_confusion_matrix(
        y_test, y_pred_test, le,
        title="Normalised confusion matrix — held-out test set",
        output_path=FIGURES_DIR / f"test_confusion_matrix.{FMT}",
        dpi=DPI,
    )

    # Per-class recall — test set
    classifier_model.plot_per_class_recall(
        y_test, y_pred_test, le,
        merged_labels=merged_labels_test,
        title="Per-class recall — held-out test set",
        output_path=FIGURES_DIR / f"test_per_class_recall.{FMT}",
        dpi=DPI,
    )
    # -------------------------------------------------------------------------------------


    # ---------------------------------- Summary ------------------------------------------
    classifier_model.print_summary(
        train_cols=train_cols,
        test_cols=test_cols,
        le=le,
        fold_metrics=fold_metrics,
        y_test=y_test,
        y_pred_test=y_pred_test,
        n_folds=n_folds,
        figures_dir=FIGURES_DIR,
    )
    # -------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
