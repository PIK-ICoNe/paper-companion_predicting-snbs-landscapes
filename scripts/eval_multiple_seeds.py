# %%
from pathlib import Path
import argparse
import sys
import pandas as pd

root_dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir_path))

from src.eval_runs import evaluate_seeds


# %%
def parse_args():
    if (
        hasattr(sys, "ps1")
        or "ipykernel_launcher" in sys.argv[0]
        or "pydevd" in sys.modules
    ):
        # Interactive mode / debug mode
        return (
            Path(
                "/home/nauck/joined_work/landscape_generation/ml_training/ns10_ds20_run_bo_op01"
            ),
            None,
            [
                "elmod",
                "osf_france",
                "osf_gb",
                "osf_spain",
            ],  # <-- set default extra_grids
        )
    parser = argparse.ArgumentParser(description="Evaluate multiple seeds")
    # parser.add_argument(
    #     "--training_dir", type=str, required=True, help="Path to the training directory"
    # )
    parser.add_argument(
        "--training_dir", type=str, default="../ml_training/tmp_try_uv_DBGNN_cnn_ds20", help="Path to the training directory"
    )
    parser.add_argument(
        "--study_name",
        type=str,
        nargs="?",
        default=None,
        help="Name of the Optuna study (optional)",
    )
    parser.add_argument(
        "--extra_grids",
        type=str,
        nargs="*",
        default=[],
        help="List of extra grid names to evaluate (e.g. osf_france osf_gb ...)",
    )
    parser.add_argument(
        "--eval_lpips",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Toggle LPIPS metric computation (disable via --no-eval-lpips).",
    )
    args = parser.parse_args()
    return Path(args.training_dir), args.study_name, args.extra_grids, args.eval_lpips


def has_configs(d):
    return (d / "model_config.yaml").exists() and (d / "training_config.yaml").exists()


def find_config_dirs(base_dir):
    # Level 1: subdirs with configs
    dirs = [d for d in base_dir.iterdir() if d.is_dir() and has_configs(d)]
    if dirs:
        return dirs
    # Level 2: sub-subdirs with configs
    dirs = []
    for d in base_dir.iterdir():
        if d.is_dir():
            dirs.extend([sd for sd in d.iterdir() if sd.is_dir() and has_configs(sd)])
    return dirs


# %%
training_dir, study_name, extra_grids, eval_lpips = parse_args()
print(f"Using training_dir: {training_dir}")
print(f"Using study_name: {study_name}" if study_name else "No study_name provided.")
if extra_grids:
    print(f"Evaluating extra grids: {extra_grids}")

# %%
config_dirs = find_config_dirs(training_dir)

if config_dirs:
    all_results = []
    for subdir in sorted(config_dirs):
        print(f"Evaluating ablation subdir: {subdir}")
        df = evaluate_seeds(
            subdir,
            ood_eval=True,
            study_name=study_name,
            extra_grids=extra_grids,
            eval_lpips=eval_lpips,
        )
        df["ablation_subdir"] = subdir.name
        # Add idx column if subdir name matches idx_N
        if subdir.name.startswith("idx_"):
            try:
                df["idx"] = int(subdir.name.split("_")[1])
            except Exception:
                df["idx"] = None
        else:
            df["idx"] = None
        all_results.append(df)
    results_df = pd.concat(all_results, ignore_index=True)
elif has_configs(training_dir): #run here
    results_df = evaluate_seeds(
        training_dir,
        ood_eval=True,
        study_name=study_name,
        extra_grids=extra_grids,
        eval_lpips=eval_lpips,
    )
else:
    print("No valid config directories found for evaluation.")
    sys.exit(1)

results_df.sort_values(by="val_loss", ascending=True, inplace=True)

# %%
# Print the mean and std for the top 3 seeds
for key in ["image_test_r2", "snbs_test_r2", "image_ood_test_r2", "snbs_ood_test_r2"]:
    mean = 100 * results_df[key][0:3].mean()
    std = 100 * results_df[key][0:3].std()
    print(f"Top 3 {key.replace('_', ' ').title()} Mean: {mean:.2f}%")
    print(f"Top 3 {key.replace('_', ' ').title()} Std: {std:.2f}%")

# Print perceptual metrics if available (no percentage scaling)
for key in ["ssim_test", "ssim_ood_test"]:
    if key in results_df.columns:
        mean = results_df[key][0:3].mean()
        std = results_df[key][0:3].std()
        print(f"Top 3 {key.replace('_', ' ').title()} Mean: {mean:.4f}")
        print(f"Top 3 {key.replace('_', ' ').title()} Std: {std:.4f}")

for key in ["lpips_test", "lpips_ood_test"]:
    if key in results_df.columns:
        mean = results_df[key][0:3].mean()
        std = results_df[key][0:3].std()
        print(f"Top 3 {key.replace('_', ' ').title()} Mean: {mean:.4f}")
        print(f"Top 3 {key.replace('_', ' ').title()} Std: {std:.4f}")

# Print extra grid R2 means/stds if present
if extra_grids:
    for grid in extra_grids:
        image_col = f"image_{grid}_r2"
        snbs_col = f"snbs_{grid}_r2"
        ssim_col = f"ssim_{grid}"
        lpips_col = f"lpips_{grid}"

        if image_col in results_df.columns:
            mean = 100 * results_df[image_col][0:3].mean()
            std = 100 * results_df[image_col][0:3].std()
            print(f"Top 3 {grid} Image R2 Mean: {mean:.2f}%")
            print(f"Top 3 {grid} Image R2 Std: {std:.2f}%")

        if snbs_col in results_df.columns:
            mean = 100 * results_df[snbs_col][0:3].mean()
            std = 100 * results_df[snbs_col][0:3].std()
            print(f"Top 3 {grid} SNBS R2 Mean: {mean:.2f}%")
            print(f"Top 3 {grid} SNBS R2 Std: {std:.2f}%")

        if ssim_col in results_df.columns:
            mean = results_df[ssim_col][0:3].mean()
            std = results_df[ssim_col][0:3].std()
            print(f"Top 3 {grid} SSIM Mean: {mean:.4f}")
            print(f"Top 3 {grid} SSIM Std: {std:.4f}")

        if lpips_col in results_df.columns:
            mean = results_df[lpips_col][0:3].mean()
            std = results_df[lpips_col][0:3].std()
            print(f"Top 3 {grid} LPIPS Mean: {mean:.4f}")
            print(f"Top 3 {grid} LPIPS Std: {std:.4f}")

# Ensure extra grid columns are present in the CSV, even if all values are None
if extra_grids:
    for grid in extra_grids:
        for col in [
            f"image_{grid}_r2",
            f"snbs_{grid}_r2",
            f"ssim_{grid}",
            f"lpips_{grid}",
        ]:
            if col not in results_df.columns:
                results_df[col] = None

# Save DataFrame to CSV
results_df.to_csv(
    Path(training_dir) / f"evaluation_results_{training_dir.name}.csv", index=False
)
