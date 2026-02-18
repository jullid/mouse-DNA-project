import pandas as pd
from pathlib import Path

import config


# -------------------------------- Raw data loading --------------------------------

def load_methylation_data(csv_file: Path) -> pd.DataFrame:
    """Load the methylation beta-value CSV into a DataFrame."""
    df = pd.read_csv(csv_file)
    # Note that the added sufixes (".1", ".2", ..., ".25") is added by pandas when read by doing df = pd.read_csv(csv_file).
    # The actual CSV does not contain those sufixes.
    return df


def load_annotation_data(annot_file: Path) -> pd.DataFrame:
    """
    Load the MM285 annotation CSV.

    Drops any rows where CpG_chrm is NaN — we do not want to extract regions
    that contain no genomic information in the annotation file.
    """
    #Import Excel annotation file with metadata, identical probe_IDs as dataset -> we can filter on this and also map extracted regions back to the excel file.
    df_annot = pd.read_csv(annot_file, low_memory=False)

    # drop any nan values from df_annot. We do not want to extract regions that contain no genomic information in the annotation file.
    df_annot = df_annot.dropna(subset=["CpG_chrm"]).copy()
    return df_annot


# -------------------------------- Filter df on NaN in the annotation csv file (originally from excel MM285) --------------------------------

def filter_by_annotation(df: pd.DataFrame, df_annot: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only probes that have a valid entry in the annotation file.

    Removes probes whose probe_ID is absent from df_annot — i.e. probes that
    carry no genomic coordinate information.  This removes ~7 415 rows
    (296 070 → 288 655).
    """
    # make sure probe_id columns are strings
    df = df.copy()
    df["probe_ID"]        = df["probe_ID"].astype(str)
    df_annot["Probe_ID"]  = df_annot["Probe_ID"].astype(str)

    # create set of valid ids, for efficiency
    valid_ids = set(df_annot["Probe_ID"])

    # create mask based on valid_ids, and filter df using this mask + make it as independent dataframe by using copy (instead of accidentally creating a .view)
    df = df[df["probe_ID"].isin(valid_ids)].copy() # Removing 7,415 rows, 296,070 rows -> 288,655 rows

    #print("Shape of df after NaN row filtering", df.shape)
    return df


# -------------------------------- df_cell_type --------------------------------

def build_celltype_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse cell type columns into one averaged column per cell type.

    Replicates are identified by the suffix pattern "<CellType>.<n>" added by
    pandas on read (e.g. "Brain_Cortex.1", "Brain_Cortex.2").  The prefix
    before the first "." is used as the group key.

    Returns a DataFrame with one averaged column per cell type (29 columns)
    plus the probe_ID column.
    """
    # Collapse cell type columns into one averaged column per cell type, resulting in 29 averaged cell types/columns
    probe_ids    = df["probe_ID"]
    df_drop_ID   = df.drop(columns="probe_ID")

    # remove suffix .1 .2 etc. added by pandas
    cell_type_labels = df_drop_ID.columns.str.split(".", n=1).str[0]

    # collapse replicates
    # Note: .T.groupby().mean().T is the pandas>=2.0 replacement for the deprecated groupby(axis=1)
    df_celltype = df_drop_ID.T.groupby(cell_type_labels).mean().T

    # Reattach probe_ID
    df_celltype.insert(0, "probe_ID", probe_ids)

    return df_celltype


# --------------------------------- df_merged ----------------------------------
"""
After analysis of (725 region, 29 tissue) correlation matrix, we will do the following:

MERGE:
- Retina & Eye
- Brain_Cortex & Subcortinal_Brain
- Thymus & Blood & Spleen
- Tail & Ear & Skin

REMOVE:
- Optic_Nerve
- Sciatic_Nerve
(These are removed because cancer rarely metastasizes here)

This is done by filtering celltype_df and then merging celltypes. Expected: (525 regions, 21 tissues). HOWEVER, we will now use more than top25 regions -> Maybe top 100, 250 or even 500.
"""

def build_merged_df(df_celltype: pd.DataFrame) -> pd.DataFrame:
    """
    Apply tissue-level merges and removals to the cell-type-averaged DataFrame.

    Drops Sciatic_Nerve, Optic_Nerve, and Mammary_Glands, then averages
    predefined tissue groups into single columns.

    Returns df_merged with probe_ID + merged/remaining tissue columns.
    """
    # Drop columns that we don't need (or too low signal)
    df_filtered_tissues = df_celltype.drop(columns=['Sciatic_Nerve', 'Optic_Nerve', 'Mammary_Glands'])

    # Define "merge groups", the KEYS are the new merged tissue names.
    merge_groups = {
        "Eye_Retina":                 ["Eye", "Retina"],
        "Brain_Cortex_Subcortical":   ["Brain_Cortex", "Subcortical_Brain"],
        "Blood_Spleen_Thymus":        ["Blood", "Spleen", "Thymus"],
        "Skin_Ears_Tail":             ["Skin", "Ears", "Tail"],
    }

    # Build new merged df
    df_merged = pd.DataFrame()
    df_merged["probe_ID"] = df_filtered_tissues["probe_ID"] # copy over IDs

    # add merged columns
    for new_name, cols in merge_groups.items():
        df_merged[new_name] = df_filtered_tissues[cols].mean(axis=1)

    # keep unmerged tissue as-is, flatten the merge_groups list
    merged_cols_flat = [c for cols in merge_groups.values() for c in cols]

    # keep non-merged columns
    remaining_cols = []
    for c in df_filtered_tissues.columns:
        if c not in merged_cols_flat and c != "probe_ID":
            remaining_cols.append(c)

    # copy remaining columns into the merged df
    df_merged[remaining_cols] = df_filtered_tissues[remaining_cols]

    return df_merged


# -------------------------------- Convenience loader --------------------------------

def load_all(
    csv_file:   Path = config.CSV_FILE,
    annot_file: Path = config.ANNOT_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full loading sequence.  Returns (df_raw, df_annot, df_celltype, df_merged).

    Intended as the single entry-point for run_pipeline.py and any future
    scripts (synthetic-data generation, modelling) that need the base data.
    """
    df_raw    = load_methylation_data(csv_file)
    df_annot  = load_annotation_data(annot_file)
    df_raw    = filter_by_annotation(df_raw, df_annot)
    df_celltype = build_celltype_df(df_raw)
    df_merged   = build_merged_df(df_celltype)
    return df_raw, df_annot, df_celltype, df_merged


# -------------------------------- Module entry-point (data verification) --------------------------------

if __name__ == "__main__":
    import sys
    print(sys.executable)
    print(f"CSV exists:   {config.CSV_FILE.exists()}")
    print(f"Annot exists: {config.ANNOT_FILE.exists()}")

    df_raw, df_annot, df_celltype, df_merged = load_all()

    print("Shape of df after NaN row filtering:", df_raw.shape)
    print("df_celltype shape:", df_celltype.shape)
    print("df_merged shape:  ", df_merged.shape)
    print("df_merged columns:", df_merged.columns.tolist())
