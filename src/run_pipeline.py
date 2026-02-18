import config
import data_loading
import diff_analysis
import visualization
import annotation


def main():

    # ----------------------------------------- Data Loading ------------------------------------------
    _, df_annot, _, df_merged = data_loading.load_all(
        csv_file=config.CSV_FILE,
        annot_file=config.ANNOT_FILE,
    )

    # ---------------------------------- Pipeline Configuration ------------------------------------
    # input dataframe
    DATAFRAME = df_merged            # tissue-averaged beta matrix (probe_ID + tissues)

    # use filtering + diff ranking (True) VS. Only diff ranking (False)
    USE_FILTERING  = config.USE_FILTERING   # passed to build_tissue_diff_table
    USE_VERBOSE    = config.USE_VERBOSE     # display the distribution of regions in cell types.
    MAX_PER_TISSUE = config.MAX_PER_TISSUE

    # region selection
    TOP_N       = config.TOP_N              # number of top regions per tissue
    REGION_MODE = config.REGION_MODE        # "all" or "unique"

    # correlation / clustering
    CORR_METHOD    = config.CORR_METHOD     # "pearson" or "spearman"
    CLUSTER_METHOD = config.CLUSTER_METHOD  # linkage method

    # plotting
    HEATMAP_TITLE        = config.HEATMAP_TITLE
    CORR_TITLE           = config.CORR_TITLE
    DENDRO_TITLE         = config.DENDRO_TITLE
    CLUSTERED_CORR_TITLE = config.CLUSTERED_CORR_TITLE

    # figure output
    FIGURES_DIR = config.FIGURES_DIR
    FMT         = config.FIGURE_FORMAT
    DPI         = config.FIGURE_DPI
    # ---------------------------------------------------------------------------------------------


    # ----------------------------------------- PIPELINE ------------------------------------------
    """ FULL PIPELINE - 20 TISSUES """
    # Step 1: compute per-tissue differential tables
    all_tissues_results = diff_analysis.build_tissue_diff_table(
        df=DATAFRAME,
        use_filtering=USE_FILTERING
    )

    # Step 2: extract top N regions per tissue
    df_top_regions = diff_analysis.extract_top_regions(
        all_tissues_results,
        top_n=TOP_N
    )

    # Step 3: create heatmap matrix (tissues x regions)
    heatmap_matrix, regions_per_tissue_count = diff_analysis.create_heatmap_matrix(
        df=DATAFRAME,
        df_top_regions=df_top_regions,
        region_mode=REGION_MODE,
        max_per_tissue=MAX_PER_TISSUE,
        verbose=USE_VERBOSE
    )

    # Step 4: plot methylation heatmap
    visualization.plot_heatmap(
        heatmap_matrix,
        regions_per_tissue_count,
        top_n=TOP_N,
        title=f"{HEATMAP_TITLE} (top {TOP_N} regions per tissue, truncated at {MAX_PER_TISSUE} regions per tissue) - {REGION_MODE} regions",
        output_path=FIGURES_DIR / f"heatmap.{FMT}",
        dpi=DPI,
    )

    # Step 5: compute tissue-tissue correlation matrix
    tissue_corr = visualization.compute_tissue_correlation(
        heatmap_matrix,
        method=CORR_METHOD
    )

    # Step 6: plot tissue correlation heatmap
    visualization.plot_tissue_correlation(
        tissue_corr,
        title=CORR_TITLE,
        output_path=FIGURES_DIR / f"tissue_correlation.{FMT}",
        dpi=DPI,
    )

    # Step 7: plot tissue dendrogram (hierarchical clustering)
    Z = visualization.plot_tissue_dendrogram(
        tissue_corr,
        method=CLUSTER_METHOD,
        title=DENDRO_TITLE,
        output_path=FIGURES_DIR / f"dendrogram.{FMT}",
        dpi=DPI,
    )

    # Step 8: plot clustered tissue correlation heatmap
    visualization.plot_clustered_tissue_correlation(
        tissue_corr,
        title=CLUSTERED_CORR_TITLE,
        output_path=FIGURES_DIR / f"clustered_tissue_correlation.{FMT}",
        dpi=DPI,
    )

    # Step 9: join surviving probes (after unique filtering) with metadata annotations in single df
    df_annot_final = annotation.build_annotated_regions(
        heatmap_matrix=heatmap_matrix,
        df_top_regions=df_top_regions,
        df_annot=df_annot
    )

    # Step 10: export the joint probe and metadata df to an excel file, saved to output_path.
    output_path = config.OUTPUT_EXCEL
    #print(output_path)
    #print(output_path.exists())
    annotation.export_annotated_regions_excel(
        df_annotated=df_annot_final,
        output_path=output_path
    )
    # ---------------------------------------------------------------------------------------------

    # Some more sanity checks:
    list_probes     = heatmap_matrix.columns.to_list()
    list_probes_set = set(list_probes)

    mask = df_top_regions[df_top_regions["probe_ID"].isin(list_probes_set)]
    #print(mask.nunique())


if __name__ == "__main__":
    main()
