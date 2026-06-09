# %%
from pathlib import Path
import sys

root_dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir_path))


from src.eval_runs import (
    build_results_df,
    print_latex_rows,
    generate_latex_rows,
    create_statistics_df,
    populate_critical_cells_df,
    auto_build_result_rows,
    build_critical_cells_results_df,
    filter_result_rows_with_critical_cells,
    get_critical_cells_latex_table_core_splits_iou,
    get_critical_cells_latex_table_countries_iou,
    get_critical_cells_latex_table_core_splits_coverage,
    get_critical_cells_latex_table_countries_coverage,
)

# %%
# Specify the base directory
base_dir = Path("../ml_training")
extra_grids = ["elmod", "osf_france", "osf_gb", "osf_spain"]
# extra_grids = None

# %%
# Create statistics DataFrame
statistics_df = create_statistics_df(base_dir, extra_grids=extra_grids)

# ============================================================================
# CONFIG: Define models by their pattern in filenames
# ============================================================================

MODELS_CONFIG = {
    "TAG-VIT": {
        "ds20_pattern": r"^evaluation_results_run_ns20_ds20_TAG_vit$",
        "ds100_pattern": r"^evaluation_results_run_ns20_ds100_TAG_vit$",
    },
    "TAG-MLP-ns10": {
        "ds20_pattern": r"^evaluation_results_run_ns10_ds20_TAG(?!_cnn)$",
        "ds100_pattern": r"^evaluation_results_run_ns10_ds100_TAG(?!_cnn)$",
    },
    "TAG-MLP": {
        "ds20_pattern": r"^evaluation_results_run_ns20_ds20_TAG(?!_cnn|_vit|_intermediate)$",
        "ds100_pattern": r"^evaluation_results_run_ns20_ds100_TAG(?!_cnn|_vit|_intermediate)$",
    },
    "TAG-CNN": {
        "ds20_pattern": r"^evaluation_results_run_ns20_ds20_TAG_cnn$",
        "ds100_pattern": r"^evaluation_results_run_ns20_ds100_TAG_cnn$",
    },
    "DBGNN-MLP": {
        "ds20_pattern": r"^evaluation_results_run_ns20_ds20_DBGNN(?!_cnn|_vit|_intermediate)$",
        "ds100_pattern": r"^evaluation_results_run_ns20_ds100_DBGNN(?!_cnn|_vit|_intermediate)$",
    },
    "DBGNN-CNN": {
        "ds20_pattern": r"^evaluation_results_run_ns20_ds20_DBGNN_cnn$",
        "ds100_pattern": r"^evaluation_results_run_ns20_ds100_DBGNN_cnn$",
    },
    "DBGNN-VIT": {
        "ds20_pattern": r"^evaluation_results_run_ns20_ds20_DBGNN_vit$",
        "ds100_pattern": r"^evaluation_results_run_ns20_ds100_DBGNN_vit$",
    },
    "TAG-MLP (1000 grids)": {
        "ds20_pattern": r"^evaluation_results_run_ns20_ds20_TAG_reduced_training_size_1000$",
        "ds100_pattern": r"^evaluation_results_run_ns20_ds100_TAG_reduced_training_size_1000$",
    },
    "TAG-MLP (3000 grids)": {
        "ds20_pattern": r"^evaluation_results_run_ns20_ds20_TAG_reduced_training_size_3000$",
        "ds100_pattern": r"^evaluation_results_run_ns20_ds100_TAG_reduced_training_size_3000$",
    },
    "TAG-MLP (intermediate)": {
        "ds20_pattern": r"^evaluation_results_run_ns20_ds20_TAG_intermediate$",
        "ds100_pattern": r"^evaluation_results_run_ns20_ds100_TAG_intermediate$",
    },
    "DBGNN-MLP (intermediate)": {
        "ds20_pattern": r"^evaluation_results_run_ns20_ds20_DBGNN_intermediate$",
        "ds100_pattern": r"^evaluation_results_run_ns20_ds100_DBGNN_intermediate$",
    },
    "TAG-MLP (3L)": {
        "ds20_pattern": r"^evaluation_results_run_ns20_ds20_TAG_3L$",
        "ds100_pattern": r"^evaluation_results_run_ns20_ds100_TAG_3L$",
    },
    "TAG-MLP (8L)": {
        "ds20_pattern": r"^evaluation_results_run_ns20_ds20_TAG_8L$",
        "ds100_pattern": r"^evaluation_results_run_ns20_ds100_TAG_8L$",
    },
    # "DBGNN-MLP-lr68e5": {
    #     "ds20_pattern": r"^evaluation_results_run_ns20_ds100_DBGNN_lr68e5",
    #     "ds100_pattern": r"^evaluation_results_run_ns20_ds100_DBGNN_lr68e5",
    # },
    # "DBGNN-MLP-drop0015": {
    #     "ds20_pattern": r"^evaluation_results_run_ns20_ds100_DBGNN_drop0015",
    #     "ds100_pattern": r"^evaluation_results_run_ns20_ds100_DBGNN_drop0015",
    # },
    # "DBGNN-MLP-drop01": {
    #     "ds20_pattern": r"^evaluation_results_run_ns20_ds100_DBGNN_drop01",
    #     "ds100_pattern": r"^evaluation_results_run_ns20_ds100_DBGNN_drop01",
    # },
    # "DBGNN-MLP-drop026": {
    #     "ds20_pattern": r"^evaluation_results_run_ns20_ds100_DBGNN_drop026",
    #     "ds100_pattern": r"^evaluation_results_run_ns20_ds100_DBGNN_drop026",
    # },
}

SELECTED_MODELS = [
    # "TAG-VIT",
    "TAG-MLP-ns10",
    "TAG-MLP",
    "TAG-CNN",
    "DBGNN-CNN",
    "DBGNN-MLP",
    # "DBGNN-VIT",
    "TAG-MLP (1000 grids)",
    "TAG-MLP (3000 grids)",
    "TAG-MLP (intermediate)",
    "DBGNN-MLP (intermediate)",
    "TAG-MLP (3L)",
    "TAG-MLP (8L)",
    # "DBGNN-MLP-lr68e5",
    # "DBGNN-MLP-drop0015",
    # "DBGNN-MLP-drop01",
    # "DBGNN-MLP-drop026"
]


# Load critical-cell summary once, before building the wide table below.
critical_cells_df = populate_critical_cells_df(base_dir)

# Generate all results dataframes

# Generate the image results DataFrame
result_rows = auto_build_result_rows(
    MODELS_CONFIG, SELECTED_MODELS, statistics_df
)

image_results_df = build_results_df(
    result_rows,
    statistics_df,
    "r2",
    extra_grids,
    task="image",
)

image_results_df_ssim = build_results_df(
    result_rows,
    statistics_df,
    "ssim",
    extra_grids,
    task="image",
)

image_results_df_lpips = build_results_df(
    result_rows,
    statistics_df,
    "lpips",
    extra_grids,
    task="image",
)

# Generate the SNBS results DataFrame
snbs_results_df = build_results_df(
    result_rows,
    statistics_df,
    "r2",
    extra_grids,
    task="snbs",
)

# %%
print_latex_rows(
    "IMAGE ROWS with metric: SSIM:",
    generate_latex_rows(
        image_results_df_ssim, metric="ssim", task="image", include_header=True
    ),
)

# %%
print_latex_rows(
    "\nSNBS ROWS:",
    generate_latex_rows(snbs_results_df, include_header=True),
)

# %%
print_latex_rows(
    "IMAGE ROWS with metric: R2:",
    generate_latex_rows(image_results_df, metric="r2", task="image", include_header=True),
)

# %%
print_latex_rows(
    "IMAGE ROWS with metric: LPIPS:",
    generate_latex_rows(
        image_results_df_lpips, metric="lpips", task="image", include_header=True
    ),
)



# %%
print_latex_rows(
    "IMAGE ROWS with metric: SSIM:",
    generate_latex_rows(
        image_results_df_ssim,
        metric="ssim",
        task="image",
        real_grids=True,
        include_header=True,
    ),
)

# %%
print_latex_rows(
    "IMAGE ROWS with metric: LPIPS:",
    generate_latex_rows(
        image_results_df_lpips,
        metric="lpips",
        task="image",
        real_grids=True,
        include_header=True,
    ),
)



# %%
print_latex_rows(
    "IMAGE ROWS with metric: R2:",
    generate_latex_rows(
        image_results_df,
        metric="r2",
        task="image",
        real_grids=True,
        include_header=True,
    ),
)
# %%
print_latex_rows(
    "SNBS ROWS:",
    generate_latex_rows(
        snbs_results_df, include_header=True, task="snbs", real_grids=True
    ),
)



# Generate the critical cells results DataFrames
result_rows_with_cells = filter_result_rows_with_critical_cells(result_rows, critical_cells_df)
critical_cells_results_df_iou = build_critical_cells_results_df(
        critical_cells_df, result_rows_with_cells, metric_col="iou_mean"
)
critical_cells_results_df_coverage = build_critical_cells_results_df(
        critical_cells_df, result_rows_with_cells, metric_col="coverage_at_30_mean"
)


# %%
# Generate LaTeX tables using generate_latex_rows
print_latex_rows(
    "\nCRITICAL CELLS LATEX TABLE 1 (core splits) - IoU:",
    get_critical_cells_latex_table_core_splits_iou(
        critical_cells_results_df_iou, as_percent=False
    ),
)

# %%
print_latex_rows(
    "\nCRITICAL CELLS LATEX TABLE 2 (countries) - IoU:",
    get_critical_cells_latex_table_countries_iou(
        critical_cells_results_df_iou, as_percent=False
    ),
)

# %%
print_latex_rows(
    "\nCRITICAL CELLS LATEX TABLE 3 (core splits) - Coverage@30:",
    get_critical_cells_latex_table_core_splits_coverage(critical_cells_results_df_coverage),
)

# %%
print_latex_rows(
    "\nCRITICAL CELLS LATEX TABLE 4 (countries) - Coverage@30:",
    get_critical_cells_latex_table_countries_coverage(critical_cells_results_df_coverage),
)



# %%
# Optionally save to CSV
image_results_df.to_csv(f"{root_dir_path}/csv_results/image_results_df.csv")
image_results_df_ssim.to_csv(f"{root_dir_path}/csv_results/image_results_df_ssim.csv")
image_results_df_lpips.to_csv(f"{root_dir_path}/csv_results/image_results_df_lpips.csv")
snbs_results_df.to_csv(f"{root_dir_path}/csv_results/snbs_results_df.csv")
if not critical_cells_results_df_iou.empty:
    critical_cells_results_df_iou.to_csv(
        f"{root_dir_path}/csv_results/critical_cells_results_df_iou.csv"
    )
if not critical_cells_results_df_coverage.empty:
    critical_cells_results_df_coverage.to_csv(
        f"{root_dir_path}/csv_results/critical_cells_results_df_coverage.csv"
    )
# %%
