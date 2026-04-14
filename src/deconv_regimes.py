"""
cfDNA-like blood-dominant benchmark regimes for the deconvolution pipeline.

Provides:
  - REGIMES: catalogue of all benchmark configurations (baseline + cfDNA variants)
  - resolve_regime_names(): expand CLI shorthand to a list of regime keys
  - generate_blood_dominant_mixtures(): mixture generator for blood-dominant regimes
  - run_regime(): run steps 6–10 (generate → validate → deconvolve → eval → plot)
                  for a single named regime; return summary metrics dict

All code builds on deconv_data, deconv_model, and config. The leakage-safe
reference/pool split produced in steps 1–5 of run_deconvolution.py is passed in
unchanged.

Realism note: Blood_Spleen_Thymus merges bulk blood, spleen, and thymus replicates.
Real cfDNA is granulocyte/lymphocyte-dominated; this is a first-order proxy.
Mixtures are built from bulk beta profiles, not fragmented/size-selected cfDNA.
Results are feasibility estimates, not quantitative cfDNA predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import config
import deconv_data
import deconv_model

# ---------------------------------------------------------------------------
# Regime catalogue
# ---------------------------------------------------------------------------

BLOOD_TISSUE: str = config.DECONV_CFDNA_BLOOD_TISSUE

# Per-regime seed offsets ensure each regime is independently reproducible
# regardless of which other regimes are requested in the same run.
_SEED_OFFSET: dict = {
    "baseline":      0,
    "cfdna_easy":    1000,
    "cfdna_medium":  2000,
    "cfdna_hard":    3000,
    "cfdna_healthy": 4000,
}

REGIMES: dict = {
    "baseline": {
        "label":  "Balanced baseline",
        "subdir": "balanced",
        "type":   "balanced",
    },
    "cfdna_easy": {
        "label":       "cfDNA easy (40–60% blood)",
        "subdir":      "easy",
        "type":        "blood_dominant",
        "blood_range": config.DECONV_CFDNA_EASY_RANGE,
        "k_nb_min":    config.DECONV_CFDNA_K_NB_MIN,
        "k_nb_max":    config.DECONV_CFDNA_K_NB_MAX,
        "alpha_nb":    config.DECONV_CFDNA_NONBLOOD_ALPHA,
    },
    "cfdna_medium": {
        "label":       "cfDNA medium (60–80% blood)",
        "subdir":      "medium",
        "type":        "blood_dominant",
        "blood_range": config.DECONV_CFDNA_MEDIUM_RANGE,
        "k_nb_min":    config.DECONV_CFDNA_K_NB_MIN,
        "k_nb_max":    config.DECONV_CFDNA_K_NB_MAX,
        "alpha_nb":    config.DECONV_CFDNA_NONBLOOD_ALPHA,
    },
    "cfdna_hard": {
        "label":       "cfDNA hard (80–95% blood)",
        "subdir":      "hard",
        "type":        "blood_dominant",
        "blood_range": config.DECONV_CFDNA_HARD_RANGE,
        "k_nb_min":    config.DECONV_CFDNA_K_NB_MIN,
        "k_nb_max":    config.DECONV_CFDNA_K_NB_MAX,
        "alpha_nb":    config.DECONV_CFDNA_NONBLOOD_ALPHA,
    },
    "cfdna_healthy": {
        "label":       "cfDNA healthy (90–99% blood)",
        "subdir":      "healthy",
        "type":        "blood_dominant",
        "blood_range": config.DECONV_CFDNA_HEALTHY_RANGE,
        "k_nb_min":    config.DECONV_CFDNA_K_NB_MIN,
        "k_nb_max":    min(2, config.DECONV_CFDNA_K_NB_MAX),  # cap at 2 for extreme blood %
        "alpha_nb":    config.DECONV_CFDNA_NONBLOOD_ALPHA,
    },
}

# CLI shorthand expansions
_SHORTHAND: dict = {
    "all":   ["cfdna_easy", "cfdna_medium", "cfdna_hard"],
    "suite": ["baseline", "cfdna_easy", "cfdna_medium", "cfdna_hard"],
}


def resolve_regime_names(requested: list) -> list:
    """
    Expand shorthand keys and validate regime names.

    Parameters
    ----------
    requested : list of str
        Names from the --regimes CLI argument.
        Accepts individual regime keys (e.g. 'cfdna_easy') or shorthands:
          'all'   → all cfDNA regimes (easy, medium, hard)
          'suite' → baseline + all cfDNA regimes

    Returns
    -------
    list of str
        Ordered, deduplicated list of canonical regime keys.
    """
    resolved = []
    for name in requested:
        if name in _SHORTHAND:
            resolved.extend(_SHORTHAND[name])
        elif name in REGIMES:
            resolved.append(name)
        else:
            valid = list(REGIMES.keys()) + list(_SHORTHAND.keys())
            raise ValueError(
                f"Unknown regime '{name}'. Valid values: {valid}"
            )

    # deduplicate while preserving order
    seen = set()
    unique = []
    for r in resolved:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


# ---------------------------------------------------------------------------
# Blood-dominant mixture generator
# ---------------------------------------------------------------------------

def generate_blood_dominant_mixtures(
    df_raw: pd.DataFrame,
    pool_cols: list,
    original_labels: pd.Series,
    label_map: dict,
    selected_probes: list,
    eligible_tissues: list,
    regime_cfg: dict,
    n_mixtures: int,
    random_state: int,
) -> tuple:
    """
    Generate blood-dominant synthetic mixtures for one cfDNA regime.

    For each mixture:
      1. Draw blood proportion p_blood ~ Uniform(blood_lo, blood_hi).
      2. Draw k_nb ~ Uniform{k_nb_min, k_nb_max} non-blood tissues.
      3. Allocate remaining (1 - p_blood) across k_nb tissues via
         Dirichlet(alpha_nb * ones(k_nb)).
      4. Compute mixture as weighted sum of one pool replicate per tissue
         (identical replicate-sampling logic to generate_synthetic_mixtures).

    Parameters
    ----------
    df_raw : pd.DataFrame
        Full raw replicate matrix with probe_ID column.
    pool_cols : list
        Sample column names in the mixture pool.
    original_labels : pd.Series
        Original tissue label per sample.
    label_map : dict
        Flat original → merged label mapping.
    selected_probes : list
        Probe IDs defining the feature space.
    eligible_tissues : list
        Merged tissue names eligible for mixture generation.
    regime_cfg : dict
        One entry from REGIMES (must be of type 'blood_dominant').
    n_mixtures : int
        Number of mixtures to generate.
    random_state : int
        Random seed (base seed + per-regime offset from caller).

    Returns
    -------
    X_mixtures : np.ndarray, shape (n_mixtures, n_probes)
    df_proportions : pd.DataFrame, shape (n_mixtures, n_all_tissues)
        Same column structure as generate_synthetic_mixtures output.
    """
    if BLOOD_TISSUE not in eligible_tissues:
        raise ValueError(
            f"Blood tissue '{BLOOD_TISSUE}' has insufficient pool replicates "
            f"and is not in eligible_tissues. Cannot generate blood-dominant mixtures."
        )

    non_blood_eligible = [t for t in eligible_tissues if t != BLOOD_TISSUE]
    if not non_blood_eligible:
        raise ValueError("No non-blood eligible tissues available.")

    blood_lo, blood_hi = regime_cfg["blood_range"]
    k_nb_min = regime_cfg["k_nb_min"]
    k_nb_max = regime_cfg["k_nb_max"]
    alpha_nb = regime_cfg["alpha_nb"]

    rng = np.random.RandomState(random_state)

    # pool replicate lookup — identical to generate_synthetic_mixtures internals
    pool_lookup = deconv_data._build_pool_lookup(
        df_raw, pool_cols, original_labels, label_map, selected_probes
    )

    # full tissue list for proportion DataFrame columns (includes zeros)
    all_tissues = sorted(set(
        original_labels.map(lambda x: label_map.get(x, x)).unique()
    ))
    tissue_to_idx = {t: i for i, t in enumerate(all_tissues)}
    n_probes = pool_lookup[BLOOD_TISSUE].shape[1]

    X_mixtures  = np.zeros((n_mixtures, n_probes))
    proportions = np.zeros((n_mixtures, len(all_tissues)))

    for mix_i in range(n_mixtures):
        # blood proportion
        p_blood = rng.uniform(blood_lo, blood_hi)

        # draw k_nb non-blood tissues
        k_nb = rng.randint(k_nb_min, k_nb_max + 1)
        chosen_nb = rng.choice(non_blood_eligible, size=k_nb, replace=False).tolist()

        # split remaining proportion across non-blood tissues via Dirichlet
        props_nb = (1.0 - p_blood) * rng.dirichlet(np.full(k_nb, alpha_nb))

        # blood replicate
        n_blood = pool_lookup[BLOOD_TISSUE].shape[0]
        mixture_vec = p_blood * pool_lookup[BLOOD_TISSUE][rng.randint(0, n_blood)]
        proportions[mix_i, tissue_to_idx[BLOOD_TISSUE]] = p_blood

        # non-blood replicates
        for j, tissue in enumerate(chosen_nb):
            n_reps = pool_lookup[tissue].shape[0]
            mixture_vec += props_nb[j] * pool_lookup[tissue][rng.randint(0, n_reps)]
            proportions[mix_i, tissue_to_idx[tissue]] = props_nb[j]

        X_mixtures[mix_i] = mixture_vec

    df_proportions = pd.DataFrame(
        proportions,
        columns=all_tissues,
        index=[f"mixture_{i}" for i in range(n_mixtures)],
    )

    print(f"Generated {n_mixtures} blood-dominant mixtures — {regime_cfg['label']}")
    print(f"  Blood fraction: {blood_lo:.0%}–{blood_hi:.0%}")
    print(f"  Non-blood tissues per mixture: {k_nb_min}–{k_nb_max}")
    print(f"  Non-blood Dirichlet alpha: {alpha_nb}")
    print(f"  Eligible non-blood tissues: {len(non_blood_eligible)}")

    return X_mixtures, df_proportions


# ---------------------------------------------------------------------------
# cfDNA-specific metrics
# ---------------------------------------------------------------------------

def compute_cfdna_metrics(
    df_proportions: pd.DataFrame,
    df_estimated: pd.DataFrame,
    nonblood_recall_threshold: float = 0.02,
) -> dict:
    """
    Compute cfDNA-relevant scalar metrics for one regime run.

    Parameters
    ----------
    df_proportions : pd.DataFrame
        Ground-truth proportions (n_mixtures × n_tissues).
    df_estimated : pd.DataFrame
        NNLS-estimated proportions (n_mixtures × n_tissues).
    nonblood_recall_threshold : float
        Minimum estimated proportion to count a non-blood tissue as detected.

    Returns
    -------
    dict with keys:
      blood_mae : float
          Mean |true_blood − pred_blood| across all mixtures.
      nonblood_recall : float or None
          Fraction of truly-present non-blood tissues whose estimated
          proportion exceeds the threshold. None if no non-blood tissues
          are present (e.g. degenerate regime).
    """
    if BLOOD_TISSUE not in df_proportions.columns:
        return {"blood_mae": None, "nonblood_recall": None}

    # blood MAE
    blood_mae = (
        (df_proportions[BLOOD_TISSUE] - df_estimated[BLOOD_TISSUE]).abs().mean()
    )

    # non-blood recall
    nonblood_cols = [c for c in df_proportions.columns if c != BLOOD_TISSUE]
    true_nb  = df_proportions[nonblood_cols]
    pred_nb  = df_estimated[nonblood_cols]

    present  = (true_nb > 0)
    detected = (pred_nb > nonblood_recall_threshold) & present

    n_present = present.values.sum()
    if n_present == 0:
        nonblood_recall = None
    else:
        nonblood_recall = detected.values.sum() / n_present

    return {"blood_mae": blood_mae, "nonblood_recall": nonblood_recall}


# ---------------------------------------------------------------------------
# Regime runner
# ---------------------------------------------------------------------------

def run_regime(
    regime_name: str,
    shared_inputs: dict,
    common_params: dict,
) -> dict:
    """
    Run deconvolution benchmark steps 6–10 for one named regime.

    Parameters
    ----------
    regime_name : str
        Key in REGIMES (e.g. 'baseline', 'cfdna_easy').
    shared_inputs : dict
        Outputs of steps 1–5, keyed by:
          df_raw, pool_cols, original_labels, label_map,
          selected_probes, eligible_tissues, W_masked
    common_params : dict
        Pipeline-wide settings keyed by:
          n_mixtures, mixture_seed, figures_dir, fmt, dpi,
          barplot_n, barplot_seed, k_min, k_max, dirichlet_alpha

    Returns
    -------
    dict with keys:
      regime_name, n_mixtures, median_mae, mean_pearson_r,
      blood_mae, nonblood_recall
    """
    if regime_name not in REGIMES:
        raise ValueError(f"Unknown regime '{regime_name}'. Valid: {list(REGIMES)}")

    cfg = REGIMES[regime_name]

    # unpack shared inputs
    df_raw           = shared_inputs["df_raw"]
    pool_cols        = shared_inputs["pool_cols"]
    original_labels  = shared_inputs["original_labels"]
    label_map        = shared_inputs["label_map"]
    selected_probes  = shared_inputs["selected_probes"]
    eligible_tissues = shared_inputs["eligible_tissues"]
    W_masked         = shared_inputs["W_masked"]

    # unpack params
    n_mixtures      = common_params["n_mixtures"]
    base_seed       = common_params["mixture_seed"]
    figures_dir     = common_params["figures_dir"]
    fmt             = common_params["fmt"]
    dpi             = common_params["dpi"]
    barplot_n       = common_params["barplot_n"]
    barplot_seed    = common_params["barplot_seed"]
    k_min           = common_params["k_min"]
    k_max           = common_params["k_max"]
    dirichlet_alpha = common_params["dirichlet_alpha"]

    regime_seed = base_seed + _SEED_OFFSET[regime_name]
    out_dir = figures_dir / cfg["subdir"]

    print()
    print("=" * 60)
    print(f"REGIME: {cfg['label']}")
    print("=" * 60)

    # ---- Step 6: Generate mixtures ----------------------------------------
    print()
    print("STEP 6: Generate synthetic mixtures")

    if cfg["type"] == "balanced":
        X_mixtures, df_proportions = deconv_data.generate_synthetic_mixtures(
            df_raw=df_raw,
            pool_cols=pool_cols,
            original_labels=original_labels,
            label_map=label_map,
            selected_probes=selected_probes,
            eligible_tissues=eligible_tissues,
            n_mixtures=n_mixtures,
            k_min=k_min,
            k_max=k_max,
            dirichlet_alpha=dirichlet_alpha,
            random_state=regime_seed,
        )
    else:
        X_mixtures, df_proportions = generate_blood_dominant_mixtures(
            df_raw=df_raw,
            pool_cols=pool_cols,
            original_labels=original_labels,
            label_map=label_map,
            selected_probes=selected_probes,
            eligible_tissues=eligible_tissues,
            regime_cfg=cfg,
            n_mixtures=n_mixtures,
            random_state=regime_seed,
        )

    # ---- Step 7: Validate + mixture-level plots ---------------------------
    print()
    print("STEP 7: Validate mixtures")

    deconv_data.validate_mixtures(X_mixtures, df_proportions)

    deconv_model.plot_proportion_distributions(
        df_proportions,
        title=f"Proportion distributions — regime: {cfg['label']}",
        output_path=out_dir / f"proportion_dist.{fmt}",
        dpi=dpi,
    )

    deconv_model.plot_mixture_pca(
        W_masked,
        X_mixtures,
        df_proportions,
        title=f"Mixture PCA — regime: {cfg['label']}",
        output_path=out_dir / f"mixture_pca.{fmt}",
        dpi=dpi,
    )

    # ---- Step 8: NNLS deconvolution ----------------------------------------
    print()
    print("STEP 8: Run NNLS deconvolution")

    df_estimated, residuals = deconv_model.deconvolve_batch(W_masked, X_mixtures)

    # ---- Step 9: Evaluate --------------------------------------------------
    print()
    print("STEP 9: Evaluate deconvolution performance")

    df_mixture_metrics = deconv_model.compute_per_mixture_metrics(
        df_proportions, df_estimated,
    )
    df_tissue_metrics = deconv_model.compute_per_tissue_metrics(
        df_proportions, df_estimated,
    )
    deconv_model.print_evaluation_summary(
        df_mixture_metrics, df_tissue_metrics, residuals,
    )

    # ---- Step 10: Evaluation plots -----------------------------------------
    print()
    print("STEP 10: Generate evaluation plots")

    deconv_model.plot_true_vs_estimated_scatter(
        df_proportions, df_estimated,
        title=f"True vs estimated proportions — regime: {cfg['label']}",
        output_path=out_dir / f"eval_scatter_per_tissue.{fmt}",
        dpi=dpi,
    )

    deconv_model.plot_mae_by_components(
        df_mixture_metrics,
        title=f"MAE by component count — regime: {cfg['label']}",
        output_path=out_dir / f"eval_mae_by_k.{fmt}",
        dpi=dpi,
    )

    deconv_model.plot_residual_distribution(
        residuals,
        title=f"Residual distribution — regime: {cfg['label']}",
        output_path=out_dir / f"eval_residuals.{fmt}",
        dpi=dpi,
    )

    deconv_model.plot_per_tissue_mae(
        df_tissue_metrics,
        title=f"Per-tissue MAE — regime: {cfg['label']}",
        output_path=out_dir / f"eval_per_tissue_mae.{fmt}",
        dpi=dpi,
    )

    # barplot: select representative mixtures
    # for cfDNA regimes k_total = 1+k_nb ∈ [k_nb_min+1, k_nb_max+1]
    if cfg["type"] == "balanced":
        barplot_k_min = barplot_k_max = common_params.get("barplot_k", k_max)
    else:
        barplot_k_min = cfg["k_nb_min"] + 1
        barplot_k_max = cfg["k_nb_max"] + 1

    selected_ids = deconv_model.select_random_k_mixtures(
        df_proportions=df_proportions,
        k_min=barplot_k_min,
        k_max=barplot_k_max,
        n_plots=barplot_n,
        random_seed=barplot_seed,
    )

    deconv_model.plot_selected_mixture_barplots(
        df_proportions=df_proportions,
        df_estimated=df_estimated,
        mixture_ids=selected_ids,
        title=(
            f"True vs predicted proportions for {len(selected_ids)} mixtures "
            f"— regime: {cfg['label']}"
        ),
        output_path=out_dir / f"selected_barplots.{fmt}",
        dpi=dpi,
    )

    # ---- cfDNA-specific metrics (baseline gets None values) ----------------
    if cfg["type"] == "blood_dominant":
        cfdna = compute_cfdna_metrics(df_proportions, df_estimated)
    else:
        cfdna = {"blood_mae": None, "nonblood_recall": None}

    return {
        "regime_name":        cfg["label"],
        "n_mixtures":         n_mixtures,
        "median_mae":         df_mixture_metrics["mae"].median(),
        "median_mae_present": df_mixture_metrics["mae_present"].median(),
        "median_mae_absent":  df_mixture_metrics["mae_absent"].median(),
        "mean_pearson_r":     df_mixture_metrics["pearson_r"].mean(),
        "blood_mae":          cfdna["blood_mae"],
        "nonblood_recall":    cfdna["nonblood_recall"],
    }
