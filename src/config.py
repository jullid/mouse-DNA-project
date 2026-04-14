from pathlib import Path

# -------------------------------- Paths --------------------------------
DATA_DIR = Path.home() / "projects" / "data"
#DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path.home() / "projects" / "mouse-DNA-project" / "results"
#OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE  = DATA_DIR / "processed" / "GSE290585_SeSaMeBeta_MM285_BS_simplified_columns.csv"
ANNOT_FILE = DATA_DIR / "original" / "12_02_2026_MM285.csv"

OUTPUT_EXCEL = OUTPUT_DIR / "mouse_methylation_markers.xlsx"

FIGURES_DIR = Path.home() / "projects" / "mouse-DNA-project" / "figures"
#FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------- Figure Output --------------------------------
FIGURE_FORMAT = "png"              # "pdf" (vector, for papers) or "png" (raster, for quick checks)
FIGURE_DPI    = 150                # only applies to raster formats (png, etc.)

# -------------------------------- Pipeline Configuration --------------------------------
# use filtering + diff ranking (True) VS. Only diff ranking (False)
USE_FILTERING = True               # passed to build_tissue_diff_table
USE_VERBOSE   = True               # display the distribution of regions in cell types.
MAX_PER_TISSUE = 50

# region selection
TOP_N       = 200                  # number of top regions per tissue
REGION_MODE = "unique"             # "all" or "unique"

# correlation / clustering
CORR_METHOD    = "pearson"         # "pearson" or "spearman"
CLUSTER_METHOD = "average"         # linkage method

# -------------------------------- Plot Titles --------------------------------
HEATMAP_TITLE        = "Mouse methylation atlas heatmap"
CORR_TITLE           = "Tissue–tissue methylation correlation"
DENDRO_TITLE         = "Hierarchical clustering of tissues (methylation correlation)"
CLUSTERED_CORR_TITLE = "Clustered tissue–tissue methylation correlation"

# -------------------------------- Deconvolution Configuration --------------------------------
# reference / mixture pool split
DECONV_REFERENCE_FRACTION = 0.5    # fraction of replicates per tissue → reference set
DECONV_SPLIT_SEED         = 42     # random seed for reference/pool split reproducibility
DECONV_MIN_POOL_REPLICATES = 3     # tissues with fewer pool replicates excluded from mixtures

# synthetic mixture generation
DECONV_N_MIXTURES       = 1000     # number of synthetic mixtures to generate
DECONV_K_MIN            = 2        # minimum tissues per mixture
DECONV_K_MAX            = 4        # maximum tissues per mixture
DECONV_DIRICHLET_ALPHA  = 1.0      # Dirichlet concentration (1.0 = uniform)
DECONV_MIXTURE_SEED     = 123      # random seed for mixture generation

# full-atlas probe panel (produced by run_pipeline.py, used as default probe panel)
DECONV_ATLAS_PROBES_PATH = DATA_DIR / "processed" / "top_50_regions_df.csv"

# barplot selection (for plot_selected_mixture_barplots)
DECONV_BARPLOT_K       = 4     # exact component count to filter mixtures; must be ≤ DECONV_K_MAX
DECONV_BARPLOT_N_PLOTS = 10    # number of mixture panels in the figure
DECONV_BARPLOT_SEED    = 42    # random seed for reproducible mixture selection

# -------------------------------- cfDNA Blood-Dominant Benchmark --------------------------------
# The blood tissue key in MERGE_GROUPS
DECONV_CFDNA_BLOOD_TISSUE   = "Blood_Spleen_Thymus"

# Non-blood tissue count per mixture (uniform draw from [k_nb_min, k_nb_max])
DECONV_CFDNA_K_NB_MIN       = 2
DECONV_CFDNA_K_NB_MAX       = 5

# Dirichlet concentration for the non-blood proportion split
DECONV_CFDNA_NONBLOOD_ALPHA = 0.5

# Blood fraction bands per regime [lo, hi) — uniform draw
DECONV_CFDNA_EASY_RANGE    = (0.40, 0.60)
DECONV_CFDNA_MEDIUM_RANGE  = (0.60, 0.80)
DECONV_CFDNA_HARD_RANGE    = (0.80, 0.95)
DECONV_CFDNA_HEALTHY_RANGE = (0.90, 0.99)  # opt-in; near-healthy cfDNA

# output paths
DECONV_OUTPUT_DIR       = OUTPUT_DIR / "deconvolution"
DECONV_FIGURES_DIR      = FIGURES_DIR / "deconvolution"
DECONV_MIXTURES_FIG_DIR = FIGURES_DIR / "deconvolution" / "mixtures"
