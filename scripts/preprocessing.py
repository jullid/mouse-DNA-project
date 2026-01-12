"""
Preprocessing script intended to be run once. It takes a CSV file, simplifies
column names, and groups cell types. The output is a processed CSV
used for all downstream modeling.
"""

from pathlib import Path
import pandas as pd
import warnings
import sys

DATA_DIR = Path.home() / "projects" / "data"
OUTPUT_DIR = DATA_DIR / "processed"
csv_file = DATA_DIR / "raw" /"GSE290585_SeSaMeBeta_MM285_BS.csv" # path to mouse csv file

print(sys.executable)
print(csv_file.exists())

OUTPUT_DIR.mkdir(exist_ok=True)

# Read CSV file:
df = pd.read_csv(csv_file)

def simplify_colname(name: str, n_underscores: int = 4) -> str:
    """
    Simplify column names by counting 'n_underscores' from the back of the string
    and removing everything to the right of that underscore (including it).

    Any duplicate suffixes of the form ".<digit>" (such as ".1" or ".2")
    are removed.

    Some columns end with the suffix "_ACEseq"; these are handled explicitly
    with conditional logic to avoid breaking tissue names that contain
    underscores.

    A warning is raised if a column contains fewer than 'n_underscores'
    underscores.
    """

    # remove trailing numeric suffix if present
    core = name.split(".", 1)[0]

    # SPECIAL CASE: ACE-seq columns
    if core.endswith("_ACEseq"):
        core = core.replace("_ACEseq", "")

    # count underscores
    underscore_count = core.count("_")
    if underscore_count < n_underscores:
        warnings.warn(
            f"Column '{name}' has only {underscore_count} underscores (< {n_underscores}); leaving unchanged."
        )
        return core

    # split from the right and keep left part
    simplified = core.rsplit("_", n_underscores)[0]

    return simplified


def reorder_columns_grouped(df):
    """
    Reorder columns by grouping base tissue/cell type names together.

    Safe with duplicate column names because reordering is done by column position.
    Keeps the first column unchanged.
    """

    def col_sort_key(col: str):
        return col.split(".", 1)[0]

    # indices for all columns except the first
    other_idx = list(range(1, df.shape[1]))

    # sort indices using base column name only
    other_idx_sorted = sorted(
        other_idx,
        key=lambda i: col_sort_key(df.columns[i])
    )

    new_idx_order = [0] + other_idx_sorted
    return df.iloc[:, new_idx_order]


# we do not want to change the first column, so here we separate them
first_col = df.columns[0]
other_cols = df.columns[1:]

# loop through the columns
simplified_other_cols = []

for col in other_cols:
    new_col = simplify_colname(col, n_underscores=4)
    simplified_other_cols.append(new_col)

# reconstruct the full column list with first column:
new_columns = [first_col] + simplified_other_cols

# assign new column names to the df
df.columns = new_columns

# call re-order function:
df_reordered = reorder_columns_grouped(df)

print(df_reordered.head())

output_path = OUTPUT_DIR / (csv_file.stem + "_simplified_columns.csv")
df_reordered.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")