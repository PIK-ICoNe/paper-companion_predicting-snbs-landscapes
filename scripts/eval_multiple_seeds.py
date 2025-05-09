# %%
from pathlib import Path
import argparse


import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_path))

from eval_runs import evaluate_seeds


# %%
if (
    hasattr(sys, "ps1") or "ipykernel_launcher" in sys.argv[0]
):  # to check if interactive

    # Interactive mode: Set default values
    training_dir = Path(
        "/home/nauck/joined_work/landscape_generation/ml_training/run_bo_opt13_5"
    )
    study_name = None
else:
    # Command-line mode: Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate multiple seeds")
    parser.add_argument(
        "--training_dir",
        type=str,
        required=True,
        help="Path to the training directory",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        required=False,
        help="Name of the Optuna study (optional)",
    )
    args = parser.parse_args()

    # Convert training_dir to a Path object
    training_dir = Path(args.training_dir)
    study_name = args.study_name

print(f"Using training_dir: {training_dir}")
if study_name:
    print(f"Using study_name: {study_name}")
else:
    print("No study_name provided.")


# %%
# Evaluate seeds
results_df = evaluate_seeds(training_dir, ood_eval=True, study_name=study_name)
# %%

results_df.sort_values(by="val_loss", ascending=True, inplace=True)

# %%
# Print the mean and standard deviation for the top 3 seeds
print(f"Top 3 Image Test R2 Mean: {100 * results_df['image_test_r2'][0:3].mean():.2f}%")
print(f"Top 3 Image Test R2 Std: {100 * results_df['image_test_r2'][0:3].std():.2f}%")

print(f"Top 3 SNBS Test R2 Mean: {100 * results_df['snbs_test_r2'][0:3].mean():.2f}%")
print(f"Top 3 SNBS Test R2 Std: {100 * results_df['snbs_test_r2'][0:3].std():.2f}%")

print(
    f"Top 3 Image Test R2 Mean: {100 * results_df['image_ood_test_r2'][0:3].mean():.2f}%"
)
print(
    f"Top 3 Image Test R2 Std: {100 * results_df['image_ood_test_r2'][0:3].std():.2f}%"
)

print(
    f"Top 3 OOD SNBS Test R2 Mean: {100 * results_df['snbs_ood_test_r2'][0:3].mean():.2f}%"
)
print(
    f"Top 3 OOD SNBS Test R2 Std: {100 * results_df['snbs_ood_test_r2'][0:3].std():.2f}%"
)
# %%
# Save DataFrame to CSV
results_df.to_csv(
    Path(training_dir) / f"evaluation_results_{training_dir.name}.csv", index=False
)
