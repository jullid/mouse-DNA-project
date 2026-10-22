# Mouse DNA Methylation Atlas & Tissue Classification/Deconvolution

A modular Python pipeline for building a tissue-specific DNA methylation atlas from mouse MM285 array data, training a tissue classifier, and performing reference-based tissue deconvolution from methylation profiles.

See [USAGE.md](USAGE.md) for how to run the pipelines and [FIGURES.md](FIGURES.md) for a description of every figure.

---

## Project Overview

This project processes Illumina MM285 methylation array data (beta values) from 267 biological replicates across 29 mouse tissue types (GSE290585). The pipeline identifies tissue-specific differentially methylated CpG regions, builds a methylation atlas, and uses the selected markers for downstream tissue classification and deconvolution.

The work is inspired by the Human Methylation Atlas (Loyfer et al., Nature 2023), with the goal of enabling **controlled validation of methylation-based tissue-of-origin methods in mouse models**, particularly in cfDNA settings.

---

## Atlas Overview

The atlas is constructed by identifying **tissue-specific unmethylated regions** through comparison of each tissue against all others. Regions are ranked based on the difference between background methylation and target tissue methylation, and the top markers define a reference signature for each tissue.

### Heatmap of Top Regions

![Heatmap of Top Regions](figures/heatmap.png)

Rows represent tissues and columns represent selected tissue-specific genomic regions, analogous to the Human Methylation Atlas.

---

## Three Independent Pipelines

### **Pipeline 1 — Methylation Atlas** (`run_pipeline.py`)
Builds the full tissue methylation atlas from all 240 usable replicates (267 on file, minus the excluded tissues).

- Computes differential methylation scores  
- Selects tissue-specific marker probes  
- Generates heatmaps, correlation matrices, dendrograms  
- Exports annotated marker regions to Excel and the marker panel to `top_50_regions_df.csv`  

This pipeline uses all available data and produces the **reference atlas** used for downstream analysis.

---

### **Pipeline 2 — Tissue Classifier** (`run_classifier.py`)
Trains a multinomial logistic regression classifier on individual replicates using merged tissue labels (20 classes).

- Leakage-free design: probe selection performed inside each CV fold  
- Uses only fold-training data for feature selection and model fitting  
- Evaluated on a held-out test set (1 sample per original tissue type)  

**Note:**  
Classifier performance is near-perfect (~100% accuracy), but evaluation is limited to a very small test set (one replicate per tissue). This likely overestimates real-world performance and should be interpreted with caution.

---

### **Pipeline 3 — Tissue Deconvolution** (`run_deconvolution.py`)
Estimates tissue proportions from mixed methylation signals using non-negative least squares (NNLS).

- Splits replicates into:
  - **Reference set** (builds signature matrix and selects probes)  
  - **Mixture pool** (generates synthetic mixtures)  
- Evaluates performance on synthetic mixtures with known ground truth  
- Supports a staged **cfDNA blood-dominant benchmark** (see below)

**cfDNA benchmark (`--regimes`):**  
Four difficulty levels simulate mixtures where `Blood_Spleen` is the dominant component, approximating cfDNA from blood draws. Each regime produces the same evaluation figure set written to `figures/deconvolution/mixtures/<regime>/`.

| Regime | Blood fraction |
|--------|---------------|
| easy | 40–60 % |
| medium | 60–80 % |
| hard | 80–95 % |
| healthy *(opt-in)* | 90–99 % |

```bash
python run_deconvolution.py                   # balanced baseline
python run_deconvolution.py --regimes all     # all cfDNA regimes
python run_deconvolution.py --regimes suite   # baseline + all cfDNA
```

**Current status:**  
The balanced baseline and the easy, medium and hard cfDNA regimes are implemented; their figures are described in [FIGURES.md](FIGURES.md).

---

## Validation Notebooks

Two notebooks in `notebooks/` validate the deconvolution on data outside the atlas and produce the publication figures. They import the `src` modules; the atlas, classifier and benchmark outputs come from the three pipelines above.

- `publication_preprocessing.ipynb`: reshapes the independent MM285 cohort GSE196902 into the atlas file layout. Run once.
- `publication.ipynb`:
  - **Part I.** GSE196902 deconvolved against the full atlas: pure samples, synthetic mixtures (k = 1–4), held-out tissues, and the limit of detection in an 85 % blood background.
  - **Part II.** Mixtures with blood fixed at 85 % (Scenario A/B), a one-tissue recovery map with NNLS fit error, and cfDNA-hard mixtures on a log scale with per-probe residuals.
  - **Part III.** NNLS deconvolution of four external mm10 datasets after per-sample coverage QC.

Figure saving is off by default (`SAVE_FIGURES` in the first cell); every figure is shown inline.

---

## Data

### Prerequisites: input data

Input files are not included in the repository. They live in one data folder, `config.DATA_DIR`, which defaults to `~/projects/data`; set the environment variable `MOUSE_DNA_DATA` to use another location. Results and figures are written inside the repository, wherever it is cloned.

```
<DATA_DIR>/
  raw/         GSE290585_SeSaMeBeta_MM285_BS.csv, GSE196902_Matrix_processed.csv, GSE196902_family.soft
  original/    12_02_2026_MM285.csv
  processed/   GSE290585_SeSaMeBeta_MM285_BS_simplified_columns.csv, GSE196902_MM285_normal_processed.csv,
               top_50_regions_df.csv
  external/    Meth_mice/   GSE42836_CpG.BETA.mm10.txt, GSE106379_CpG.BETA.mm10.txt, GSE93957_CpG.BETA.mm10.txt,
                            GSE184410_CpG.BETA.mm10.txt, GEO_mouse_methylation_metadata.xlsx (may be a symlink)
```

**Studies and annotation:**
- **GSE290585** (`GSE290585_SeSaMeBeta_MM285_BS.csv`): MM285 beta values for 267 replicates of 29 mouse tissues; the atlas cohort, used by all three pipelines. `scripts/preprocessing.py` converts it once to `GSE290585_SeSaMeBeta_MM285_BS_simplified_columns.csv` with tissue-named columns.
- **MM285 probe annotation** (`12_02_2026_MM285.csv`): genomic coordinates, probe design and gene annotation per probe; used to drop probes without coordinates, to annotate the marker export, and to map probes to CpG positions for the external datasets.
- **GSE196902** (`GSE196902_Matrix_processed.csv`, `GSE196902_family.soft`): MM285 beta values for 67 samples (56 normal tissues, 11 cell lines) and their GEO metadata; an independent validation cohort, converted by `notebooks/publication_preprocessing.ipynb` and used in `notebooks/publication.ipynb` Parts I–II.
- **GSE42836** (`GSE42836_CpG.BETA.mm10.txt`): WGBS, 17 samples across mouse tissues; external dataset for `publication.ipynb` Part III.
- **GSE106379** (`GSE106379_CpG.BETA.mm10.txt`): WGBS, 24 liver samples; external dataset for Part III.
- **GSE93957** (`GSE93957_CpG.BETA.mm10.txt`): RRBS, 62 samples, strands reported separately; external dataset for Part III.
- **GSE184410** (`GSE184410_CpG.BETA.mm10.txt`): MM285 array, 1,239 samples; external dataset for Part III.
- **External metadata** (`GEO_mouse_methylation_metadata.xlsx`): per-sample GEO metadata for the four external datasets, one sheet per study; used to label Part III samples.

**Generated inputs:**
- `top_50_regions_df.csv`: the full-atlas marker panel, written by `run_pipeline.py`.
- `GSE196902_MM285_normal_processed.csv`: the 56 normal GSE196902 samples in atlas layout, written by `publication_preprocessing.ipynb`.

**Data dimensions (GSE290585):**
- 296,070 probes × 267 replicates in the simplified matrix; 288,655 probes after removing probes without genomic coordinates  
- 29 original tissue types → 25 after exclusions → 20 merged classes (240 replicates)  

---

## Tissue Merging and Exclusion

Based on correlation analysis of the atlas:

**Merged groups:**
- Eye + Retina → `Eye_Retina`  
- Brain_Cortex + Subcortical_Brain → `Brain_Cortex_Subcortical`  
- Blood + Spleen → `Blood_Spleen`  
- Skin + Ears + Tail → `Skin_Ears_Tail`  

**Excluded tissues:**
- Sciatic_Nerve  
- Optic_Nerve  
- Mammary_Glands  
- Thymus  

Definitions are stored in `data_loading.py` (`MERGE_GROUPS`, `EXCLUDED_TISSUES`) and used consistently across pipelines.

---

## Directory Structure

```text
mouse-DNA-project/
├── .gitignore
├── README.md
├── USAGE.md
├── FIGURES.md
├── CHANGELOG.md
│
├── notebooks/
│   ├── publication_preprocessing.ipynb
│   └── publication.ipynb
│
├── scripts/
│   └── preprocessing.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loading.py
│   ├── diff_analysis.py
│   ├── visualization.py
│   ├── annotation.py
│   ├── utils.py
│   ├── classifier_data.py
│   ├── classifier_model.py
│   ├── deconv_data.py
│   ├── deconv_model.py
│   ├── deconv_regimes.py
│   ├── run_pipeline.py
│   ├── run_classifier.py
│   └── run_deconvolution.py
│
├── results/
│   └── deconvolution/
│       └── split.json
│
└── figures/
    ├── heatmap.png
    ├── tissue_correlation.png
    ├── dendrogram.png
    ├── clustered_tissue_correlation.png
    ├── classifier/
    ├── deconvolution/
    └── publication/
```

Only PNG figures are tracked. Tables and the Excel marker export are written to `results/` locally and are not part of the repository.
---

## Pipeline Design Principles

### Leakage Prevention

All pipelines enforce strict data separation:

- **Atlas:** Uses all samples (reference only)  
- **Classifier:** Feature selection performed inside CV folds only  
- **Deconvolution:** Strict split between reference and mixture samples  

No sample is reused across incompatible steps.

---

### Modular Architecture

- Shared functionality in `utils.py`  
- No cross-pipeline dependencies  
- Clean separation between atlas, classification, and deconvolution  

---

## Known Limitations

- Small tissue classes (e.g. Uterus, Stomach, Adrenal) limit CV robustness  
- Hyperparameters were tuned during full-dataset exploration  
- Classifier performance likely overestimates real-world generalisation  
- Real data will require probe ID remapping (platform differences)

---

## Requirements

Python 3.9+

pandas  
numpy  
scikit-learn  
scipy  
matplotlib  
seaborn  
openpyxl  
jupyter (notebooks only)  

---

## References

Loyfer et al., *A DNA methylation atlas of normal human cell types*, Nature, 2023.