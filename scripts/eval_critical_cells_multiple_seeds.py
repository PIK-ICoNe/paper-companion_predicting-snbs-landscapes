# %%
from pathlib import Path
import argparse
import sys
import pandas as pd
import os
import yaml
import optuna

root_dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir_path))

from src.training import load_datasets
from src.eval_runs import load_model_with_best_seed
from src.eval import evaluate_gridwise_top20


def get_seeds_from_study(training_dir, study_name=None):
    study_db_path = training_dir / "seeds_study.db"
    if not study_db_path.exists():
        return []
    if study_name is None:
        study_name = training_dir.name
    try:
        study = optuna.load_study(
            study_name=study_name, storage=f"sqlite:///{study_db_path}"
        )
    except Exception as e:
        print(f"Could not load Optuna study: {e}")
        return []
    seeds = [
        trial.system_attrs["fixed_params"]["seed"]
        for trial in study.trials
        if "fixed_params" in trial.system_attrs
        and "seed" in trial.system_attrs["fixed_params"]
    ]
    return sorted(set(seeds))


def has_configs(d):
    return (d / "model_config.yaml").exists() and (d / "training_config.yaml").exists()


def find_config_dirs(base_dir):
    dirs = [d for d in base_dir.iterdir() if d.is_dir() and has_configs(d)]
    if dirs:
        return dirs
    dirs = []
    for d in base_dir.iterdir():
        if d.is_dir():
            dirs.extend([sd for sd in d.iterdir() if sd.is_dir() and has_configs(sd)])
    return dirs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate critical cells IoU over all seeds"
    )
    parser.add_argument(
        "--training_dir", type=str, required=True, help="Path to the training directory"
    )
    parser.add_argument(
        "--study_name", type=str, default=None, help="Optuna study name (optional)"
    )
    args = parser.parse_args()
    return Path(args.training_dir), args.study_name


# Datasets to evaluate
all_datasets = [
    {"name": "ds20", "dataset_name": "ds20", "test_slice_index": [8501, 10000]},
    {"name": "ds100", "dataset_name": "ds100", "test_slice_index": [8501, 10000]},
    {"name": "France", "dataset_name": "osf_france", "test_slice_index": [1, 1]},
    {"name": "Germany", "dataset_name": "elmod", "test_slice_index": [1, 1]},
    {"name": "Great Britain", "dataset_name": "osf_gb", "test_slice_index": [1, 1]},
    {"name": "Spain", "dataset_name": "osf_spain", "test_slice_index": [1, 1]},
]


def main():
    training_dir, study_name = parse_args()
    config_dirs = find_config_dirs(training_dir)
    if config_dirs:
        model_dirs = config_dirs
    elif has_configs(training_dir):
        model_dirs = [training_dir]
    else:
        print("No valid config directories found for evaluation.")
        sys.exit(1)

    results_seeds = []
    for model_dir in model_dirs:
        model_name = model_dir.name
        seeds = get_seeds_from_study(model_dir, study_name)
        if not seeds:
            print(f"No seeds found for {model_name} using Optuna study. Skipping.")
            continue
        for dataset in all_datasets:
            metric_values: dict[str, list[float]] = {}
            for seed in seeds:
                try:
                    model = load_model_with_best_seed(model_dir, seed)
                except Exception as e:
                    print(f"Could not load model for seed {seed}: {e}")
                    continue
                training_config_path = model_dir / "training_config.yaml"
                with open(training_config_path, "r") as f:
                    dataset_config = yaml.safe_load(f)
                dataset_config = dataset_config.copy()
                dataset_config["dataset_name"] = dataset["dataset_name"]
                dataset_config["test_slice_index"] = dataset["test_slice_index"]
                loaders = load_datasets(
                    dataset_config, train=False, val=False, test=True
                )
                test_loader = loaders.get("test_loader")
                if test_loader is None:
                    continue
                all_metrics, mean_metrics = evaluate_gridwise_top20(
                    test_loader, model, threshold=0.7
                )
                for metric_name, metric_value in mean_metrics.items():
                    metric_values.setdefault(metric_name, []).append(metric_value)
            if metric_values:
                row = {
                    "model": model_name,
                    "dataset": dataset["name"],
                    "num_seeds": len(seeds),
                }
                for metric_name, values in metric_values.items():
                    series = pd.Series(values, dtype="float64")
                    row[f"{metric_name}_mean"] = float(series.mean())
                    row[f"{metric_name}_std"] = float(series.std())
                results_seeds.append(row)
    results_seeds_df = pd.DataFrame(results_seeds)
    results_seeds_df.to_csv(
        training_dir / f"metrics_critical_cells_all_seeds_summary.csv", index=False
    )
    print("Saved:", training_dir / f"metrics_critical_cells_all_seeds_summary.csv")


if __name__ == "__main__":
    main()
