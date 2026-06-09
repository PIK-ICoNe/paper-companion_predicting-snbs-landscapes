from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import yaml

root_dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir_path))

from src.eval_runs import evaluate_loader
from src.gnn import init_model
from src.training import load_datasets


CHECKPOINT_RE = re.compile(r"seed_(?P<seed>\d+)_epoch_(?P<epoch>\d+)\.pt$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all saved epoch checkpoints and plot performance over epochs."
    )
    parser.add_argument("training_dir", type=str, help="Path to the training directory")
    parser.add_argument(
        "--eval_lpips",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable LPIPS evaluation.",
    )
    return parser.parse_args()


def load_configs(training_dir: Path):
    with open(training_dir / "model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    with open(training_dir / "training_config.yaml", "r") as f:
        training_config = yaml.safe_load(f)
    training_config.setdefault("vae_beta", 0.0)
    return model_config, training_config


def find_checkpoints(training_dir: Path):
    checkpoints = []
    for path in sorted(training_dir.glob("seed_*_epoch_*.pt")):
        match = CHECKPOINT_RE.match(path.name)
        if match:
            checkpoints.append(
                {
                    "seed": int(match.group("seed")),
                    "epoch": int(match.group("epoch")),
                    "path": path,
                }
            )
    return checkpoints


def dataset_test_slice(dataset_name: str):
    if dataset_name in {"ds20", "ds100"}:
        return [8501, 10000]
    return [1, 1]


def dataset_label(dataset_name: str):
    return {
        "ds20": "ds20",
        "ds100": "Texas",
        "elmod": "Germany",
        "osf_france": "France",
        "osf_gb": "Great Britain",
        "osf_spain": "Spain",
    }.get(dataset_name, dataset_name)


def build_loaders(training_config: dict):
    # Keep the logic simple and close to eval_runs: copy config and only override dataset_name/test_slice_index.
    dataset_names = [
        training_config["dataset_name"],
        "ds20",
        "ds100",
        "elmod",
        "osf_france",
        "osf_gb",
        "osf_spain",
    ]

    loaders = {}
    for dataset_name in dict.fromkeys(
        dataset_names
    ):  # preserve order, remove duplicates
        cfg = copy.deepcopy(training_config)
        cfg["dataset_name"] = dataset_name
        cfg["test_slice_index"] = dataset_test_slice(dataset_name)
        loaded = load_datasets(cfg, train=False, val=False, test=True)
        loaders[dataset_name] = {
            "label": dataset_label(dataset_name),
            "loader": loaded.get("test_loader"),
        }
    return loaders


def load_model_from_checkpoint(
    model_config: dict, checkpoint_path: Path, device: torch.device
):
    model = init_model(model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("mlp.") for k in state_dict):
        state_dict = {
            (k.replace("mlp.", "decoder.", 1) if k.startswith("mlp.") else k): v
            for k, v in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=False)
    return model


def evaluate_checkpoints(
    training_dir: Path, model_config: dict, training_config: dict, eval_lpips: bool
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    loaders = build_loaders(training_config)
    checkpoints = find_checkpoints(training_dir)

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint files matching seed_*_epoch_*.pt found in {training_dir}"
        )

    rows = []
    for ckpt in checkpoints:
        print(f"Evaluating {ckpt['path'].name}")
        model = load_model_from_checkpoint(model_config, ckpt["path"], device)

        for dataset_name, info in loaders.items():
            loader = info["loader"]
            if loader is None or len(loader.dataset) == 0:
                continue

            loss, image_r2, snbs_r2, quality = evaluate_loader(
                model,
                loader,
                criterion,
                device,
                enable_lpips=eval_lpips,
                vae_beta=training_config["vae_beta"],
            )
            rows.append(
                {
                    "seed": ckpt["seed"],
                    "epoch": ckpt["epoch"],
                    "checkpoint_file": ckpt["path"].name,
                    "dataset_name": dataset_name,
                    "dataset_label": info["label"],
                    "loss": loss,
                    "image_r2": image_r2,
                    "snbs_r2": snbs_r2,
                    "ssim": quality["ssim"],
                    "lpips": quality["lpips"],
                }
            )

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df.sort_values(["seed", "epoch", "dataset_name"], inplace=True)
    return results_df


def summarize_over_epochs(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df.copy()

    return (
        results_df.groupby(["dataset_name", "dataset_label", "epoch"], dropna=False)
        .agg(
            n=("seed", "count"),
            loss_mean=("loss", "mean"),
            loss_std=("loss", "std"),
            image_r2_mean=("image_r2", "mean"),
            image_r2_std=("image_r2", "std"),
            snbs_r2_mean=("snbs_r2", "mean"),
            snbs_r2_std=("snbs_r2", "std"),
            ssim_mean=("ssim", "mean"),
            ssim_std=("ssim", "std"),
            lpips_mean=("lpips", "mean"),
            lpips_std=("lpips", "std"),
        )
        .reset_index()
        .sort_values(["dataset_name", "epoch"])
    )


def save_plot(summary_df: pd.DataFrame, metric: str, out_path: Path):
    col = f"{metric}_mean"
    if summary_df.empty or col not in summary_df.columns:
        return

    plot_df = summary_df.dropna(subset=["epoch", col])
    if plot_df.empty:
        return

    plt.figure(figsize=(10, 6))
    for label, group in plot_df.groupby("dataset_label"):
        group = group.sort_values("epoch")
        plt.plot(group["epoch"], group[col], marker="o", label=label)
    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").upper())
    plt.title(f"{metric.replace('_', ' ').upper()} over saved epochs")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    training_dir = Path(args.training_dir).resolve()

    model_config, training_config = load_configs(training_dir)
    results_df = evaluate_checkpoints(
        training_dir=training_dir,
        model_config=model_config,
        training_config=training_config,
        eval_lpips=args.eval_lpips,
    )
    summary_df = summarize_over_epochs(results_df)

    detailed_csv = training_dir / "performance_over_epoch_all_checkpoints.csv"
    summary_csv = training_dir / "performance_over_epoch_summary.csv"
    results_df.to_csv(detailed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    plots_dir = training_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["loss", "image_r2", "snbs_r2", "ssim", "lpips"]:
        save_plot(summary_df, metric, plots_dir / f"{metric}_over_epoch.png")

    print(f"Saved detailed CSV: {detailed_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved plots in: {plots_dir}")


if __name__ == "__main__":
    main()
