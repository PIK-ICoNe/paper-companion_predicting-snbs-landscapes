# %%
import pandas as pd
from pathlib import Path
import sys

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_path))


from eval_runs import (
    populate_results_df,
    generate_latex_rows,
    create_statistics_df,
    get_column_mappings,
)

# Specify the base directory
base_dir = Path("../ml_training")

# Create statistics DataFrame
statistics_df = create_statistics_df(base_dir)

# Define column mappings
image_column_mappings, snbs_column_mappings = get_column_mappings()

# Define a dictionary containing multiple rows
result_rows = {
    "row1": {
        "tr20": "evaluation_results_run_bo_opt010",
        "tr100": "evaluation_results_run_bo_opt017_1",
        "model": "TAG-MLP",
    },
    "row2": {
        "tr20": "evaluation_results_run_bo_opt016",
        "tr100": "evaluation_results_run_bo_opt015",
        "model": "DBGNN-MLP",
    },
}

# Generate the image results DataFrame
image_results_df = populate_results_df(
    result_rows, statistics_df, image_column_mappings
)

# Generate the SNBS results DataFrame
snbs_results_df = populate_results_df(result_rows, statistics_df, snbs_column_mappings)

# %%
print("IMAGE ROWS:")
# Generate and print LaTeX rows for image results
image_latex_rows = generate_latex_rows(image_results_df)
for row in image_latex_rows:
    print(row)

print("SNBS ROWS:")
# Generate and print LaTeX rows for SNBS results
snbs_latex_rows = generate_latex_rows(snbs_results_df)
for row in snbs_latex_rows:
    print(row)

# %%
