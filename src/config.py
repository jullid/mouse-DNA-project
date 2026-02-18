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
