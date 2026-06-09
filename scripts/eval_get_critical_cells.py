from pathlib import Path
import argparse
import sys
from typing import Optional

import optuna
import pandas as pd
import yaml

root_dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir_path))

from src.training import load_datasets
from src.eval_runs import load_model_with_best_seed
from src.eval import evaluate_gridwise_top20


def get_seeds_from_study(
    training_dir: Path, study_name: Optional[str] = None
) -> list[int]:
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
        print(f"Could not load Optuna study for {training_dir.name}: {e}")
        return []

    seeds = [
        trial.system_attrs["fixed_params"]["seed"]
        for trial in study.trials
        if "fixed_params" in trial.system_attrs
        and "seed" in trial.system_attrs["fixed_params"]
    ]
    return sorted(set(seeds))


def has_configs(d: Path) -> bool:
    return (d / "model_config.yaml").exists() and (d / "training_config.yaml").exists()


def find_config_dirs(base_dir: Path) -> list[Path]:
    dirs = [d for d in base_dir.iterdir() if d.is_dir() and has_configs(d)]
    if dirs:
        return sorted(dirs)

    nested = []
    for d in base_dir.iterdir():
        if d.is_dir():
            nested.extend([sd for sd in d.iterdir() if sd.is_dir() and has_configs(sd)])
    return sorted(nested)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate critical cells IoU over all seeds"
    )
    parser.add_argument(
        "--training_dir",
        type=str,
        required=True,
        help="Path to training dir or parent dir containing multiple training dirs",
    )
    parser.add_argument(
        "--study_name", type=str, default=None, help="Optuna study name (optional)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="Critical-cell threshold"
    )
    args = parser.parse_args()
    return Path(args.training_dir), args.study_name, args.threshold


ALL_DATASETS = [
    {"name": "ds20", "dataset_name": "ds20", "test_slice_index": [8501, 10000]},
    {"name": "ds100", "dataset_name": "ds100", "test_slice_index": [8501, 10000]},
    {"name": "France", "dataset_name": "osf_france", "test_slice_index": [1, 1]},
    {"name": "Germany", "dataset_name": "elmod", "test_slice_index": [1, 1]},
    {"name": "Great Britain", "dataset_name": "osf_gb", "test_slice_index": [1, 1]},
    {"name": "Spain", "dataset_name": "osf_spain", "test_slice_index": [1, 1]},
]


def main():
    training_dir, study_name, threshold = parse_args()

    if has_configs(training_dir):
        model_dirs = [training_dir]
        output_dir = training_dir
    else:
        model_dirs = find_config_dirs(training_dir)
        output_dir = training_dir

    if not model_dirs:
        print("No valid config directories found for evaluation.")
        sys.exit(1)

    results = []
    for model_dir in model_dirs:
        model_name = model_dir.name
        seeds = get_seeds_from_study(model_dir, study_name)
        if not seeds:
            print(f"No seeds found for {model_name}. Skipping.")
            continue

        training_config_path = model_dir / "training_config.yaml"
        with open(training_config_path, "r") as f:
            base_dataset_config = yaml.safe_load(f)

        for dataset in ALL_DATASETS:
            metric_values: dict[str, list[float]] = {}
            for seed in seeds:
                try:
                    model = load_model_with_best_seed(model_dir, seed)
                except Exception as e:
                    print(f"Could not load model for {model_name}, seed={seed}: {e}")
                    continue

                dataset_cfg = base_dataset_config.copy()
                dataset_cfg["dataset_name"] = dataset["dataset_name"]
                dataset_cfg["test_slice_index"] = dataset["test_slice_index"]

                loaders = load_datasets(dataset_cfg, train=False, val=False, test=True)
                test_loader = loaders.get("test_loader")
                if test_loader is None:
                    continue

                _, mean_metrics = evaluate_gridwise_top20(
                    test_loader, model, threshold=threshold
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
                results.append(row)

    results_df = pd.DataFrame(results)
    output_path = output_dir / "metrics_critical_cells_all_seeds_summary.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
