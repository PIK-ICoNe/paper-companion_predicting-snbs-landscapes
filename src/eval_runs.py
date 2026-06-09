import torch
import numpy as np
import yaml
import optuna
import copy
import matplotlib.pyplot as plt

from pathlib import Path
import torch
import pandas as pd
from sklearn.metrics import r2_score
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.training import load_datasets, compute_snbs_pred_labels, _unwrap_model_output
from src.gnn import init_model


def _plot_with_error(ax, epochs, mean, std, label=None):
    ax.plot(epochs, mean, marker="o", label=label)
    if std is not None:
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2)


def plot_metric_over_epoch_comparison(
    run_dirs,
    metric,
    out_path=None,
    ylabel=None,
    y_lim=None,
):
    """Plot `<metric>_mean` (± `<metric>_std`) over epochs for multiple run dirs.

    Expects each run directory to contain `performance_over_epoch_summary.csv`.
    Each run is one subplot and each available dataset in that run is a curve.
    """
    run_dirs = [Path(p) for p in run_dirs]
    dfs = []
    titles = []
    for p in run_dirs:
        csv = p / "performance_over_epoch_summary.csv"
        if not csv.exists():
            raise FileNotFoundError(f"Summary CSV not found: {csv}")
        df = pd.read_csv(csv)
        dfs.append(df)
        titles.append(p.name)

    col_mean = f"{metric}_mean"
    col_std = f"{metric}_std"

    n = len(dfs)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    preferred_order = [
        "ds20",
        "ds100",
        "Texas",
        "France",
        "Germany",
        "Great Britain",
        "Spain",
    ]

    for i, (df, title) in enumerate(zip(dfs, titles)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]

        groups = {
            label: grp.sort_values("epoch")
            for label, grp in df.groupby("dataset_label")
        }
        if not groups:
            ax.text(0.5, 0.5, "No dataset groups found", ha="center", va="center")
            ax.set_title(title)
            continue

        ordered_labels = [l for l in preferred_order if l in groups]
        ordered_labels += [l for l in groups.keys() if l not in ordered_labels]

        for lbl in ordered_labels:
            sub = groups[lbl]
            if col_mean not in sub.columns:
                continue
            epochs = sub["epoch"].to_numpy()
            mean = sub[col_mean].to_numpy()
            std = sub[col_std].to_numpy() if col_std in sub.columns else None
            _plot_with_error(ax, epochs, mean, std, label=lbl)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel or metric.upper())
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        if y_lim is not None:
            ax.set_ylim(y_lim)

    total_axes = rows * cols
    for j in range(n, total_axes):
        r = j // cols
        c = j % cols
        axes[r][c].axis("off")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=180)
        plt.close()
    else:
        plt.show()


def plot_stacked_metrics_over_epoch_for_runs(
    run_dirs,
    metrics=("image_r2", "ssim", "lpips"),
    out_path=None,
    dataset_labels=("Texas", "ds100"),
    y_lims=None,
):
    """Plot multiple metrics in stacked rows for selected runs.

    Args:
        run_dirs: Iterable of run directories. Each must contain
            `performance_over_epoch_summary.csv`.
        metrics: Metric names (without suffix), e.g. ('image_r2', 'ssim', 'lpips').
        out_path: Optional save path. If None, the figure is shown.
        dataset_labels: Preferred dataset labels to plot (first one present is used
            per run). Useful when `ds100` is labeled as `Texas`.
        y_lims: Optional dict mapping metric -> (ymin, ymax), e.g.
            {'image_r2': (-1.0, 1.0)}.
    """
    run_dirs = [Path(p) for p in run_dirs]
    metrics = list(metrics)
    y_lims = y_lims or {}

    per_run = []
    for p in run_dirs:
        csv = p / "performance_over_epoch_summary.csv"
        if not csv.exists():
            raise FileNotFoundError(f"Summary CSV not found: {csv}")
        df = pd.read_csv(csv)

        if "dataset_label" not in df.columns:
            raise KeyError(f"Missing 'dataset_label' in {csv}")

        chosen_label = None
        for lbl in dataset_labels:
            if lbl in set(df["dataset_label"].dropna().tolist()):
                chosen_label = lbl
                break
        if chosen_label is None:
            chosen_label = df["dataset_label"].dropna().iat[0]

        sub = df[df["dataset_label"] == chosen_label].sort_values("epoch")
        per_run.append({"title": p.name, "label": chosen_label, "df": sub})

    rows = len(metrics)
    fig, axes = plt.subplots(rows, 1, figsize=(8, 3.6 * rows), squeeze=False)

    for i, metric in enumerate(metrics):
        ax = axes[i][0]
        col_mean = f"{metric}_mean"
        col_std = f"{metric}_std"

        for item in per_run:
            sub = item["df"]
            if col_mean not in sub.columns:
                continue
            epochs = sub["epoch"].to_numpy()
            mean = sub[col_mean].to_numpy()
            std = sub[col_std].to_numpy() if col_std in sub.columns else None
            legend_label = f"{item['title']} ({item['label']})"
            _plot_with_error(ax, epochs, mean, std, label=legend_label)

        if metric == "image_r2":
            y_label = "Image R²"
        elif metric == "ssim":
            y_label = "SSIM"
        elif metric == "lpips":
            y_label = "LPIPS"
        else:
            y_label = metric.upper()

        ax.set_ylabel(y_label)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        if metric in y_lims:
            ax.set_ylim(y_lims[metric])

    axes[-1][0].set_xlabel("Epoch")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=180)
        plt.close()
    else:
        plt.show()


def plot_metrics_grid_over_epoch_comparison(
    run_dirs,
    metrics=("image_r2", "ssim", "lpips"),
    out_path=None,
    y_lims=None,
):
    """Plot metrics x runs grid with all dataset curves in each subplot.

    Layout:
        - rows: metrics
        - cols: run directories

    Each subplot contains all available dataset/task curves for the corresponding
    run and metric, analogous to `plot_metric_over_epoch_comparison`.
    """
    run_dirs = [Path(p) for p in run_dirs]
    metrics = list(metrics)
    y_lims = y_lims or {}

    dfs = []
    titles = []
    for p in run_dirs:
        csv = p / "performance_over_epoch_summary.csv"
        if not csv.exists():
            raise FileNotFoundError(f"Summary CSV not found: {csv}")
        df = pd.read_csv(csv)
        dfs.append(df)
        titles.append(p.name)

    preferred_order = [
        "ds20",
        "ds100",
        "Texas",
        "France",
        "Germany",
        "Great Britain",
        "Spain",
    ]

    n_rows = len(metrics)
    n_cols = len(dfs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )

    for r, metric in enumerate(metrics):
        col_mean = f"{metric}_mean"
        col_std = f"{metric}_std"

        if metric == "image_r2":
            y_label = "Image R²"
        elif metric == "ssim":
            y_label = "SSIM"
        elif metric == "lpips":
            y_label = "LPIPS"
        else:
            y_label = metric.upper()

        for c, (df, title) in enumerate(zip(dfs, titles)):
            ax = axes[r][c]

            groups = {
                label: grp.sort_values("epoch")
                for label, grp in df.groupby("dataset_label")
            }
            if not groups:
                ax.text(0.5, 0.5, "No dataset groups found", ha="center", va="center")
                ax.set_title(title)
                continue

            ordered_labels = [l for l in preferred_order if l in groups]
            ordered_labels += [l for l in groups.keys() if l not in ordered_labels]

            any_curve = False
            for lbl in ordered_labels:
                sub = groups[lbl]
                if col_mean not in sub.columns:
                    continue
                epochs = sub["epoch"].to_numpy()
                mean = sub[col_mean].to_numpy()
                std = sub[col_std].to_numpy() if col_std in sub.columns else None
                _plot_with_error(ax, epochs, mean, std, label=lbl)
                any_curve = True

            if not any_curve:
                ax.text(0.5, 0.5, f"Missing {col_mean}", ha="center", va="center")

            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(y_label)
            ax.grid(alpha=0.3)
            if metric in y_lims:
                ax.set_ylim(y_lims[metric])
            ax.legend(loc="best")
            if r == n_rows - 1:
                ax.set_xlabel("Epoch")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=180)
        plt.close()
    else:
        plt.show()


def evaluate_model(
    model, loader, criterion, device, quality_tracker=None, vae_beta: float = 0.0
):
    """
    Evaluate the model on a given data loader.
    Returns predictions, targets, SNBS predictions, and SNBS labels.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_snbs_preds = []
    all_snbs_targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            predictions, mu, logvar = _unwrap_model_output(output)
            kl_loss = (
                (
                    -0.5
                    * torch.sum(
                        1 + logvar - mu.pow(2) - logvar.exp(), dtype=predictions.dtype
                    )
                )
                / predictions.shape[0]
                if mu is not None
                else 0.0
            )
            loss = criterion(predictions, data.y) + vae_beta * kl_loss
            total_loss += loss.item()

            preds_cpu = predictions.detach().cpu()
            targets_cpu = data.y.detach().cpu()
            all_preds.append(preds_cpu.numpy())
            all_targets.append(targets_cpu.numpy())

            snbs_preds, snbs_labels = compute_snbs_pred_labels(
                predictions, data.y, data.sample_heatmaps
            )
            all_snbs_preds.append(snbs_preds.cpu().numpy())
            all_snbs_targets.append(snbs_labels.cpu().numpy())

            if quality_tracker is not None:
                quality_tracker.update(preds_cpu, targets_cpu)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_snbs_preds = np.concatenate(all_snbs_preds)
    all_snbs_targets = np.concatenate(all_snbs_targets)

    return (
        total_loss / len(loader),
        all_preds,
        all_targets,
        all_snbs_preds,
        all_snbs_targets,
    )


def _prepare_heatmap_tensor(array) -> torch.Tensor:
    """
    Convert inputs of shape (N, H, W) or (N, C, H, W) into a normalized tensor
    suitable for image-quality metrics.
    """
    if isinstance(array, torch.Tensor):
        tensor = array.detach().float()
    else:
        tensor = torch.as_tensor(array).float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim != 4:
        raise ValueError(f"Unexpected heatmap shape: {tensor.shape}")
    return tensor.clamp(0.0, 1.0)


class ImageQualityMetricAccumulator:
    """Accumulate SSIM/LPIPS batch by batch to limit memory usage."""

    def __init__(self, enable_lpips: bool = True, lpips_resize: int = 64):
        self.enable_lpips = enable_lpips
        self.metric_device = torch.device("cpu")
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(
            self.metric_device
        )
        self.lpips_metric = (
            LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(self.metric_device)
            if enable_lpips
            else None
        )
        self.lpips_resize = lpips_resize

    def _prepare_for_lpips(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = F.interpolate(
            tensor,
            size=(self.lpips_resize, self.lpips_resize),
            mode="bilinear",
            align_corners=False,
        ).repeat(1, 3, 1, 1)
        return tensor * 2.0 - 1.0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds_tensor = _prepare_heatmap_tensor(preds).to(self.metric_device)
        targets_tensor = _prepare_heatmap_tensor(targets).to(self.metric_device)
        self.ssim_metric.update(preds_tensor, targets_tensor)
        if self.lpips_metric is not None:
            preds_lpips = self._prepare_for_lpips(preds_tensor)
            targets_lpips = self._prepare_for_lpips(targets_tensor)
            self.lpips_metric.update(preds_lpips, targets_lpips)

    def compute(self) -> dict:
        result = {"ssim": self.ssim_metric.compute().item()}
        if self.lpips_metric is not None:
            result["lpips"] = self.lpips_metric.compute().item()
        else:
            result["lpips"] = None
        return result


def evaluate_loader(
    model, loader, criterion, device, enable_lpips=True, vae_beta: float = 0.0
):
    """
    Helper function to evaluate a single data loader.
    Returns loss, predictions, targets, SNBS predictions, and SNBS labels.
    """
    quality_tracker = ImageQualityMetricAccumulator(enable_lpips=enable_lpips)
    loss, preds, targets, snbs_preds, snbs_targets = evaluate_model(
        model,
        loader,
        criterion,
        device,
        quality_tracker=quality_tracker,
        vae_beta=vae_beta,
    )
    r2 = r2_score(targets.flatten(), preds.flatten())
    snbs_r2 = r2_score(snbs_targets, snbs_preds)
    quality_metrics = quality_tracker.compute()
    return loss, r2, snbs_r2, quality_metrics


def evaluate_seeds(
    training_dir,
    ood_eval=False,
    study_name=None,
    extra_grids=None,
    eval_lpips=True,
):
    """
    Evaluate models for each seed and return a DataFrame with results.
    Optionally evaluate on extra grids (e.g., osf_france, osf_gb, ...).
    """

    model_config_path = training_dir / "model_config.yaml"
    training_config_path = training_dir / "training_config.yaml"

    with open(model_config_path, "r") as file:
        model_config = yaml.safe_load(file)

    with open(training_config_path, "r") as file:
        training_config = yaml.safe_load(file)
    training_config.setdefault("vae_beta", 0.0)

    # Load the study
    study_db_path = training_dir / "seeds_study.db"
    print("Loading study from :", study_db_path)
    if study_name == None:
        study_name = training_dir.name
    print("loading study with study_name: ", study_name)
    study = optuna.load_study(
        study_name=study_name, storage=f"sqlite:///{study_db_path}"
    )

    # Extract seeds from the study trials
    seeds = [
        trial.system_attrs["fixed_params"]["seed"]
        for trial in study.trials
        if "fixed_params" in trial.system_attrs
        and "seed" in trial.system_attrs["fixed_params"]
    ]

    # Load datasets
    loaders = load_datasets(training_config, train=False, val=True, test=True)
    test_loader = loaders.get("test_loader")
    val_loader = loaders.get("val_loader")

    if ood_eval:  # here, we assume ds100 as OOD dataset
        ood_config = copy.deepcopy(training_config)
        ood_config["dataset_name"] = "ds100"
        ood_config["num_sections"] = training_config["num_sections"]

        ood_config["test_slice_index"] = [8501, 10000]
        ood_loaders = load_datasets(ood_config, train=False, val=True, test=True)
        ood_val_loader = ood_loaders.get("val_loader")
        ood_test_loader = ood_loaders.get("test_loader")

    # Prepare extra grid loaders if requested
    extra_grid_loaders = {}
    if extra_grids is not None:
        for grid_name in extra_grids:
            grid_config = copy.deepcopy(training_config)
            grid_config["dataset_name"] = grid_name
            grid_config["num_sections"] = training_config["num_sections"]
            grid_config["test_slice_index"] = [1, 1]  # Always use index 1
            loaders_grid = load_datasets(grid_config, train=False, val=False, test=True)
            extra_grid_loaders[grid_name] = loaders_grid.get("test_loader")

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize loss function
    criterion = nn.MSELoss()

    def _fmt_metric(value):
        return "N/A" if value is None else f"{value:.4f}"

    results = []
    for seed in seeds:
        print(f"Evaluating model with seed: {seed}")
        model_path = Path(training_dir) / f"seed_{seed}_best_model.pt"
        if not model_path.exists():
            print(f"Model for seed {seed} not found. Skipping.")
            continue

        # Load model
        model = init_model(model_config).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("mlp.") for k in state_dict):
            state_dict = {
                (k.replace("mlp.", "decoder.", 1) if k.startswith("mlp.") else k): v
                for k, v in state_dict.items()
            }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(
                f"state_dict compatibility: missing={missing}, unexpected={unexpected}"
            )

        # Evaluate on test and validation sets
        test_loss, image_test_r2, snbs_test_r2, test_quality = evaluate_loader(
            model,
            test_loader,
            criterion,
            device,
            enable_lpips=eval_lpips,
            vae_beta=training_config["vae_beta"],
        )
        val_loss, image_val_r2, snbs_val_r2, val_quality = evaluate_loader(
            model,
            val_loader,
            criterion,
            device,
            enable_lpips=eval_lpips,
            vae_beta=training_config["vae_beta"],
        )

        # Evaluate on OOD datasets if applicable
        ood_results = {}
        if ood_eval:
            (
                ood_val_loss,
                ood_val_r2,
                ood_snbs_val_r2,
                ood_val_quality,
            ) = evaluate_loader(
                model,
                ood_val_loader,
                criterion,
                device,
                enable_lpips=eval_lpips,
                vae_beta=training_config["vae_beta"],
            )
            (
                ood_test_loss,
                ood_test_r2,
                ood_snbs_test_r2,
                ood_test_quality,
            ) = evaluate_loader(
                model,
                ood_test_loader,
                criterion,
                device,
                enable_lpips=eval_lpips,
                vae_beta=training_config["vae_beta"],
            )
            ood_results = {
                "ood_val_loss": ood_val_loss,
                "ood_val_r2": ood_val_r2,
                "snbs_ood_val_r2": ood_snbs_val_r2,
                "ood_test_loss": ood_test_loss,
                "image_ood_test_r2": ood_test_r2,
                "snbs_ood_test_r2": ood_snbs_test_r2,
                "ssim_ood_val": ood_val_quality["ssim"],
                "lpips_ood_val": ood_val_quality["lpips"],
                "ssim_ood_test": ood_test_quality["ssim"],
                "lpips_ood_test": ood_test_quality["lpips"],
            }

        # Evaluate on extra grids if requested
        extra_grid_results = {}
        if extra_grids is not None:
            for grid_name, grid_loader in extra_grid_loaders.items():
                if grid_loader is not None and len(grid_loader.dataset) > 0:
                    data = grid_loader.dataset[0]  # Only one item due to [1, 1] slice
                    data = data.to(device)
                    model.eval()
                    with torch.no_grad():
                        output = model(data)
                        predictions, mu, logvar = _unwrap_model_output(output)
                        # Image R2
                        image_r2 = r2_score(
                            data.y.cpu().numpy().flatten(),
                            predictions.cpu().numpy().flatten(),
                        )
                        # SNBS R2
                        snbs_pred, snbs_label = compute_snbs_pred_labels(
                            predictions, data.y, data.sample_heatmaps
                        )
                        snbs_r2 = r2_score(
                            snbs_label.cpu().numpy().flatten(),
                            snbs_pred.cpu().numpy().flatten(),
                        )
                        grid_quality_tracker = ImageQualityMetricAccumulator(
                            enable_lpips=eval_lpips
                        )
                        grid_quality_tracker.update(
                            predictions.detach().cpu(), data.y.detach().cpu()
                        )
                        grid_quality = grid_quality_tracker.compute()
                    extra_grid_results[f"image_{grid_name}_r2"] = image_r2
                    extra_grid_results[f"snbs_{grid_name}_r2"] = snbs_r2
                    extra_grid_results[f"ssim_{grid_name}"] = grid_quality["ssim"]
                    extra_grid_results[f"lpips_{grid_name}"] = grid_quality["lpips"]
                else:
                    extra_grid_results[f"image_{grid_name}_r2"] = None
                    extra_grid_results[f"snbs_{grid_name}_r2"] = None
                    extra_grid_results[f"ssim_{grid_name}"] = None
                    extra_grid_results[f"lpips_{grid_name}"] = None

        result = {
            "seed": seed,
            "test_loss": test_loss,
            "val_loss": val_loss,
            "image_test_r2": image_test_r2,
            "image_val_r2": image_val_r2,
            "snbs_test_r2": snbs_test_r2,
            "snbs_val_r2": snbs_val_r2,
            "ssim_test": test_quality["ssim"],
            "lpips_test": test_quality["lpips"],
            "ssim_val": val_quality["ssim"],
            "lpips_val": val_quality["lpips"],
        }

        # Add OOD results if available
        result.update(ood_results)
        # Add extra grid results if available
        result.update(extra_grid_results)

        results.append(result)
        print(
            f"Seed {seed}: Test Loss = {test_loss:.4f}, Image Test R2 = {image_test_r2:.4f}, SNBS Test R2 = {snbs_test_r2:.4f}"
        )
        print(
            f"Seed {seed}: Val Loss = {val_loss:.4f}, Image Val R2 = {image_val_r2:.4f}, SNBS Val R2 = {snbs_val_r2:.4f}"
        )
        print(
            f"Seed {seed}: SSIM Test = {test_quality['ssim']:.4f}, LPIPS Test = {_fmt_metric(test_quality['lpips'])}"
        )
        print(
            f"Seed {seed}: SSIM Val = {val_quality['ssim']:.4f}, LPIPS Val = {_fmt_metric(val_quality['lpips'])}"
        )
        if ood_eval:
            print(
                f"Seed {seed}: SSIM OOD Test = {ood_test_quality['ssim']:.4f}, LPIPS OOD Test = {_fmt_metric(ood_test_quality['lpips'])}"
            )
            print(
                f"Seed {seed}: SSIM OOD Val = {ood_val_quality['ssim']:.4f}, LPIPS OOD Val = {_fmt_metric(ood_val_quality['lpips'])}"
            )
        if extra_grids is not None:
            for grid_name in extra_grids:
                img_key = f"image_{grid_name}_r2"
                snbs_key = f"snbs_{grid_name}_r2"
                ssim_key = f"ssim_{grid_name}"
                lpips_key = f"lpips_{grid_name}"
                print(
                    f"Seed {seed}: {grid_name} Image R2 = {_fmt_metric(result.get(img_key, None))}"
                )
                print(
                    f"Seed {seed}: {grid_name} SNBS R2 = {_fmt_metric(result.get(snbs_key, None))}"
                )
                print(
                    f"Seed {seed}: {grid_name} SSIM = {_fmt_metric(result.get(ssim_key, None))}, LPIPS = {_fmt_metric(result.get(lpips_key, None))}"
                )

    # Create DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def populate_results_df(
    result_rows,
    statistics_df,
    column_mappings=None,
    extra_grids=None,
    task="image",
    metric="r2",
):
    """
    Populate a DataFrame with results based on mappings and input rows.
    Optionally include extra grid results (e.g., osf_france, osf_gb, ...).
    task controls which extra-grid columns are added:
    - task="image" -> image_<grid>_mean/std
    - task="snbs"  -> snbs_<grid>_mean/std

    If column_mappings is omitted, it is derived from `metric` via
    `get_column_mappings(metric)`.
    """
    task = task.lower()
    if task not in {"image", "snbs"}:
        raise ValueError(f"Unknown task: {task}. Use 'image' or 'snbs'.")

    if column_mappings is None:
        image_column_mappings, snbs_column_mappings = get_column_mappings(metric=metric)
        column_mappings = image_column_mappings if task == "image" else snbs_column_mappings

    rows = []  # Use a list to collect rows
    for key, result_one_row in result_rows.items():
        row = {"model": result_one_row["model"]}
        # Handle tr20ev20, tr100ev100, tr20ev100
        for stat_key in [
            ("tr20ev20_mean", "tr20"),
            ("tr20ev20_std", "tr20"),
            ("tr100ev100_mean", "tr100"),
            ("tr100ev100_std", "tr100"),
            ("tr20ev100_mean", "tr20"),
            ("tr20ev100_std", "tr20"),
        ]:
            colname, which = stat_key
            filename = result_one_row.get(which, None)  # Use .get() to avoid KeyError
            if filename is None:
                row[colname] = None
                continue
            mask = statistics_df["filename"] == filename
            if mask.any():
                row[colname] = statistics_df.loc[mask, column_mappings[colname]].values[
                    0
                ]
            else:
                row[colname] = None
        # Optionally add extra grid results
        if extra_grids is not None:
            for grid in extra_grids:
                mean_col = f"{task}_{grid}_mean"
                std_col = f"{task}_{grid}_std"
                filename = result_one_row.get(grid, "")
                mask = statistics_df["filename"] == filename
                if metric == "r2":
                    source_mean_col = mean_col
                    source_std_col = std_col
                else:
                    source_mean_col = f"{task}_{grid}_{metric}_mean"
                    source_std_col = f"{task}_{grid}_{metric}_std"

                if mask.any() and source_mean_col in statistics_df.columns:
                    row[mean_col] = statistics_df.loc[mask, source_mean_col].values[0]
                else:
                    row[mean_col] = None
                if mask.any() and source_std_col in statistics_df.columns:
                    row[std_col] = statistics_df.loc[mask, source_std_col].values[0]
                else:
                    row[std_col] = None
        rows.append(row)  # Add the row to the list
    return pd.DataFrame(rows)


def build_results_df(result_rows, statistics_df, metric, extra_grids, task):
    """Convenience wrapper around `populate_results_df()` for a given task."""
    return populate_results_df(
        result_rows,
        statistics_df,
        metric=metric,
        extra_grids=extra_grids,
        task=task,
    )


def print_latex_rows(title, latex_rows):
    """Print a section title and its LaTeX rows."""
    print(title)
    for row in latex_rows:
        print(row)


def generate_latex_rows(
    results_df,
    metric="r2",
    real_grids=False,
    task="image",
    include_header=False,
    column_mapping=None,
    as_percent=None,
):
    """
    Generate LaTeX rows for a given results DataFrame.

    By default, percent formatting is applied for:
    - r2, ssim (image metrics)
    - iou, coverage* (critical cell metrics)
    Raw formatting is used for:
    - lpips

    If as_percent is explicitly set (True/False), it overrides the default behavior.

    If real_grids=True, output columns are:
    - Germany
    - France
    - Great Britain
    - Spain

    In that case, task controls which columns are used:
    - task="image" -> image_<grid>_mean/std
    - task="snbs"  -> snbs_<grid>_mean/std

    If include_header=True, prepend a LaTeX header row.
    
    If column_mapping is provided, it maps column names to (mean_col, std_col) tuples.
    Example: {"tr20ev20": ("tr20", "tr20_std"), ...}
    Used for custom DataFrames (e.g., critical cells) with non-standard column names.
    """
    metric = metric.lower()
    task = task.lower()
    if task not in {"image", "snbs"}:
        raise ValueError(f"Unknown task: {task}. Use 'image' or 'snbs'.")

    # Determine if we should format as percent
    if as_percent is None:
        # Auto-detect: r2, ssim, iou, and coverage metrics are typically 0-1 or 0-100
        as_percent = metric in {"r2", "ssim", "iou", "coverage"} or "coverage" in metric

    latex_rows = []
    if include_header:
        if column_mapping:
            # Custom header based on mapping keys
            header_cols = list(column_mapping.keys())
            latex_rows.append(" & ".join(["Model"] + header_cols) + " \\\\")
        elif real_grids:
            latex_rows.append(
                "Model & tr100evGermany & tr100evFrance & tr100evGB & tr100evSpain \\\\"
            )
        else:
            latex_rows.append("Model & tr20ev20 & tr100ev100 & tr20ev100 \\\\")

    for _, row in results_df.iterrows():
        model = row.get("model", row.get("Model", ""))

        def fmt(val, std):
            if pd.isna(val) or pd.isna(std):
                return "--"
            if as_percent:
                return f"{100*val:.2f}\\tiny{{$\\pm {100*std:.2f}$}}"
            return f"{val:.2f}\\tiny{{$\\pm {std:.2f}$}}"

        if column_mapping:
            # Use custom column mapping
            values = [model]
            for col_name, (mean_col, std_col) in column_mapping.items():
                val = fmt(row.get(mean_col), row.get(std_col))
                values.append(val)
            latex_rows.append(" & ".join(values) + " \\\\")
        elif real_grids:
            germany = fmt(
                row.get(f"{task}_elmod_mean", None),
                row.get(f"{task}_elmod_std", None),
            )
            france = fmt(
                row.get(f"{task}_osf_france_mean", None),
                row.get(f"{task}_osf_france_std", None),
            )
            gb = fmt(
                row.get(f"{task}_osf_gb_mean", None),
                row.get(f"{task}_osf_gb_std", None),
            )
            spain = fmt(
                row.get(f"{task}_osf_spain_mean", None),
                row.get(f"{task}_osf_spain_std", None),
            )
            latex_rows.append(f"{model} & {germany} & {france} & {gb} & {spain} \\\\")
        else:
            tr20ev20 = fmt(row["tr20ev20_mean"], row["tr20ev20_std"])
            tr100ev100 = fmt(row["tr100ev100_mean"], row["tr100ev100_std"])
            tr20ev100 = fmt(row["tr20ev100_mean"], row["tr20ev100_std"])
            latex_rows.append(f"{model} & {tr20ev20} & {tr100ev100} & {tr20ev100} \\\\")
    return latex_rows


def find_evaluation_result(statistics_df, pattern):
    """Find an evaluation result filename matching the given regex pattern.

    The pattern is matched against the full filename stem, so callers should
    provide exact regexes anchored with `^...$` when they want a single run.
    """
    # Try a few strict fullmatch variants to avoid accidental substring collisions.
    # If the caller provided an anchored pattern (^...$), respect it.
    candidates = []
    anchored = pattern.startswith("^") or pattern.endswith("$")
    if anchored:
        candidates.append(pattern)
    else:
        # try the pattern as-is (useful when callers already include regex anchors),
        # then try forcing an exact-match, and then try common filename prefixes.
        candidates.extend([
            pattern,
            f"^{pattern}$",
            f"^evaluation_results_{pattern}$",
            f"^evaluation_results_run_{pattern}$",
        ])

    resolved = []
    for cand in candidates:
        try:
            matches = statistics_df[statistics_df["filename"].str.fullmatch(cand, na=False)].copy()
        except Exception:
            # If the candidate is an invalid regex, skip it.
            matches = pd.DataFrame()
        if not matches.empty:
            if len(matches) > 1:
                raise ValueError(
                    f"Pattern {pattern!r} matched multiple evaluation results via {cand!r}: "
                    f"{matches['filename'].tolist()}"
                )
            resolved.append(matches.iloc[0]["filename"])

    if not resolved:
        raise ValueError(
            f"No evaluation result matched pattern {pattern!r}. "
            f"Tried: {candidates}"
        )

    unique_resolved = list(dict.fromkeys(resolved))
    if len(unique_resolved) > 1:
        raise ValueError(
            f"Pattern {pattern!r} is ambiguous and matched multiple evaluation results: "
            f"{unique_resolved}"
        )

    return unique_resolved[0]


def auto_build_result_rows(models_config, selected_models, statistics_df):
    """
    Auto-generate result_rows dict from a model config.

    Each selected model needs a ds20 and ds100 pattern. The same ds100 run is reused
    for the real-grid columns.
    """
    result_rows = {}

    for model_name in selected_models:
        if model_name not in models_config:
            continue

        config = models_config[model_name]
        ds20_run = find_evaluation_result(statistics_df, config["ds20_pattern"])
        ds100_run = find_evaluation_result(statistics_df, config["ds100_pattern"])

        result_rows[model_name] = {
            "model": model_name,
            "tr20": ds20_run,
            "tr100": ds100_run,
            "elmod": ds100_run,
            "osf_france": ds100_run,
            "osf_gb": ds100_run,
            "osf_spain": ds100_run,
        }

    return result_rows


def build_critical_cells_results_df(
    critical_cells_df,
    result_rows,
    metric_col="iou_mean",
    std_col=None,
):
    """Build a wide critical-cells dataframe from the long-form critical cells CSV."""
    if std_col is None:
        std_col = metric_col.replace("_mean", "_std") if metric_col.endswith("_mean") else f"{metric_col}_std"

    critical_keys = [
        "tr20",
        "tr100",
        "tr20ev100",
        "elmod",
        "osf_france",
        "osf_gb",
        "osf_spain",
    ]

    rows = []
    for model_name, row_info in result_rows.items():
        ds20_model = row_info["tr20"].replace("evaluation_results_", "")
        ds100_model = row_info["tr100"].replace("evaluation_results_", "")

        lookup_map = {
            "tr20": ("ds20", ds20_model),
            "tr100": ("ds100", ds100_model),
            "tr20ev100": ("ds100", ds20_model),
            "elmod": ("Germany", ds100_model),
            "osf_france": ("France", ds100_model),
            "osf_gb": ("Great Britain", ds100_model),
            "osf_spain": ("Spain", ds100_model),
        }

        row_dict = {"Model": model_name}
        for key in critical_keys:
            dataset_name, model_name_query = lookup_map[key]
            match = critical_cells_df[
                (critical_cells_df["model"] == model_name_query)
                & (critical_cells_df["dataset"] == dataset_name)
            ]
            if match.empty:
                row_dict[key] = None
                row_dict[f"{key}_std"] = None
            else:
                row_dict[key] = match.iloc[0][metric_col]
                row_dict[f"{key}_std"] = match.iloc[0][std_col]
        rows.append(row_dict)

    return pd.DataFrame(rows)


def filter_result_rows_with_critical_cells(result_rows, critical_cells_df):
    """
    Filter result_rows to only include models that have critical cells data.
    
    Args:
        result_rows: Dict of model results from auto_build_result_rows()
        critical_cells_df: DataFrame from populate_critical_cells_df()
    
    Returns:
        Filtered dict containing only rows with critical cells data
    """
    return {
        k: v for k, v in result_rows.items()
        if v.get("tr20", "").replace("evaluation_results_", "") in critical_cells_df["model"].values
    }


def print_critical_cells_latex_tables(critical_cells_results_df):
    """
    Generate and print all 4 critical cells LaTeX tables.
    
    Tables:
    1. Core splits - IoU
    2. Countries - IoU
    3. Core splits - Coverage@30
    4. Countries - Coverage@30
    """
    if critical_cells_results_df.empty:
        print("WARNING: No critical cells data available for LaTeX tables")
        return
    
    # Table 1: Core splits - IoU
    print("\nCRITICAL CELLS LATEX TABLE 1 (core splits) - IoU:")
    table1 = get_critical_cells_latex_table_core_splits_iou(critical_cells_results_df)
    for row in table1:
        print(row)
    
    # Table 2: Countries - IoU
    print("\nCRITICAL CELLS LATEX TABLE 2 (countries) - IoU:")
    table2 = get_critical_cells_latex_table_countries_iou(critical_cells_results_df)
    for row in table2:
        print(row)
    
    # Table 3: Core splits - Coverage@30
    print("\nCRITICAL CELLS LATEX TABLE 3 (core splits) - Coverage@30:")
    table3 = get_critical_cells_latex_table_core_splits_coverage(critical_cells_results_df)
    for row in table3:
        print(row)
    
    # Table 4: Countries - Coverage@30
    print("\nCRITICAL CELLS LATEX TABLE 4 (countries) - Coverage@30:")
    table4 = get_critical_cells_latex_table_countries_coverage(critical_cells_results_df)
    for row in table4:
        print(row)


def get_critical_cells_latex_table_core_splits_iou(critical_cells_results_df, as_percent=False):
    """Generate LaTeX rows for critical cells core splits (tr20, tr100, tr20ev100) with IoU metric."""
    if critical_cells_results_df.empty:
        return []
    return generate_latex_rows(
        critical_cells_results_df,
        metric="iou",
        column_mapping={
            "tr20ev20": ("tr20", "tr20_std"),
            "tr100ev100": ("tr100", "tr100_std"),
            "tr20ev100": ("tr20ev100", "tr20ev100_std"),
        },
        include_header=True,
        as_percent=as_percent
    )


def get_critical_cells_latex_table_countries_iou(critical_cells_results_df, as_percent=False):
    """Generate LaTeX rows for critical cells countries (Germany, France, GB, Spain) with IoU metric."""
    if critical_cells_results_df.empty:
        return []
    return generate_latex_rows(
        critical_cells_results_df,
        metric="iou",
        column_mapping={
            "tr100evGermany": ("elmod", "elmod_std"),
            "tr100evFrance": ("osf_france", "osf_france_std"),
            "tr100evGB": ("osf_gb", "osf_gb_std"),
            "tr100evSpain": ("osf_spain", "osf_spain_std"),
        },
        include_header=True,
        as_percent=as_percent
    )


def get_critical_cells_latex_table_core_splits_coverage(critical_cells_results_df, as_percent=False):
    """Generate LaTeX rows for critical cells core splits with Coverage@30 metric."""
    if critical_cells_results_df.empty:
        return []
    return generate_latex_rows(
        critical_cells_results_df,
        metric="coverage_at_30_mean",
        column_mapping={
            "tr20ev20": ("tr20", "tr20_std"),
            "tr100ev100": ("tr100", "tr100_std"),
            "tr20ev100": ("tr20ev100", "tr20ev100_std"),
        },
        include_header=True,
        as_percent=as_percent
    )


def get_critical_cells_latex_table_countries_coverage(critical_cells_results_df, as_percent=False):
    """Generate LaTeX rows for critical cells countries with Coverage@30 metric."""
    if critical_cells_results_df.empty:
        return []
    return generate_latex_rows(
        critical_cells_results_df,
        metric="coverage_at_30_mean",
        column_mapping={
            "tr100evGermany": ("elmod", "elmod_std"),
            "tr100evFrance": ("osf_france", "osf_france_std"),
            "tr100evGB": ("osf_gb", "osf_gb_std"),
            "tr100evSpain": ("osf_spain", "osf_spain_std"),
        },
        include_header=True,
        as_percent=as_percent
    )


def create_statistics_df(base_dir, extra_grids=None):
    """
    Create a DataFrame containing statistics from CSV files in the given base directory.
    Optionally include extra grid results (e.g., osf_france, osf_gb, ...).
    """
    import pandas as pd

    # Dictionary to store DataFrames
    csv_data = {}

    # Loop over all directories and find .csv files
    skip_names = {
        "metrics_critical_cells_all_seeds_summary.csv",
        "performance_over_epoch_all_checkpoints.csv",
        "performance_over_epoch_summary.csv",
    }
    for csv_file in base_dir.rglob("*.csv"):  # Recursively find all .csv files
        if csv_file.name in skip_names:
            continue  # Skip known non-stat CSVs
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        # Only sort by val_loss if the column exists (some new CSVs don't have it)
        if "val_loss" in df.columns:
            try:
                df.sort_values(by="val_loss", ascending=False, inplace=True)
            except Exception:
                # If sorting fails for any reason, skip sorting and continue
                pass
        # Use the filename (without extension) as the key
        csv_data[csv_file.stem] = df

    # Initialize a list to store statistics for each file
    statistics = []

    # Loop over all keys in csv_data and calculate the requested statistics
    for filename, results_df in csv_data.items():
        try:
            stats = {
                "filename": filename,
                "image_test_r2_mean": results_df["image_test_r2"][0:3].mean(),
                "image_test_r2_std": results_df["image_test_r2"][0:3].std(),
                "snbs_test_r2_mean": results_df["snbs_test_r2"][0:3].mean(),
                "snbs_test_r2_std": results_df["snbs_test_r2"][0:3].std(),
                "image_ood_test_r2_mean": results_df["image_ood_test_r2"][0:3].mean(),
                "image_ood_test_r2_std": results_df["image_ood_test_r2"][0:3].std(),
                "snbs_ood_test_r2_mean": results_df["snbs_ood_test_r2"][0:3].mean(),
                "snbs_ood_test_r2_std": results_df["snbs_ood_test_r2"][0:3].std(),
            }
            if "ssim_test" in results_df:
                stats["ssim_test_mean"] = results_df["ssim_test"][0:3].mean()
                stats["ssim_test_std"] = results_df["ssim_test"][0:3].std()
            else:
                stats["ssim_test_mean"] = None
                stats["ssim_test_std"] = None

            if "lpips_test" in results_df:
                stats["lpips_test_mean"] = results_df["lpips_test"][0:3].mean()
                stats["lpips_test_std"] = results_df["lpips_test"][0:3].std()
            else:
                stats["lpips_test_mean"] = None
                stats["lpips_test_std"] = None

            if "ssim_ood_test" in results_df:
                stats["ssim_ood_test_mean"] = results_df["ssim_ood_test"][0:3].mean()
                stats["ssim_ood_test_std"] = results_df["ssim_ood_test"][0:3].std()
            else:
                stats["ssim_ood_test_mean"] = None
                stats["ssim_ood_test_std"] = None

            if "lpips_ood_test" in results_df:
                stats["lpips_ood_test_mean"] = results_df["lpips_ood_test"][0:3].mean()
                stats["lpips_ood_test_std"] = results_df["lpips_ood_test"][0:3].std()
            else:
                stats["lpips_ood_test_mean"] = None
                stats["lpips_ood_test_std"] = None
            # Optionally add extra grid results
            if extra_grids is not None:
                for grid in extra_grids:
                    image_mean_col = f"image_{grid}_mean"
                    image_std_col = f"image_{grid}_std"
                    image_ssim_mean_col = f"image_{grid}_ssim_mean"
                    image_ssim_std_col = f"image_{grid}_ssim_std"
                    image_lpips_mean_col = f"image_{grid}_lpips_mean"
                    image_lpips_std_col = f"image_{grid}_lpips_std"
                    snbs_mean_col = f"snbs_{grid}_mean"
                    snbs_std_col = f"snbs_{grid}_std"
                    image_r2_col = f"image_{grid}_r2"
                    snbs_r2_col = f"snbs_{grid}_r2"
                    if image_r2_col in results_df.columns:
                        stats[image_mean_col] = results_df[image_r2_col][0:3].mean()
                        stats[image_std_col] = results_df[image_r2_col][0:3].std()
                    else:
                        stats[image_mean_col] = None
                        stats[image_std_col] = None

                    if f"ssim_{grid}" in results_df.columns:
                        stats[image_ssim_mean_col] = results_df[f"ssim_{grid}"][0:3].mean()
                        stats[image_ssim_std_col] = results_df[f"ssim_{grid}"][0:3].std()
                    else:
                        stats[image_ssim_mean_col] = None
                        stats[image_ssim_std_col] = None

                    if f"lpips_{grid}" in results_df.columns:
                        stats[image_lpips_mean_col] = results_df[f"lpips_{grid}"][0:3].mean()
                        stats[image_lpips_std_col] = results_df[f"lpips_{grid}"][0:3].std()
                    else:
                        stats[image_lpips_mean_col] = None
                        stats[image_lpips_std_col] = None

                    if snbs_r2_col in results_df.columns:
                        stats[snbs_mean_col] = results_df[snbs_r2_col][0:3].mean()
                        stats[snbs_std_col] = results_df[snbs_r2_col][0:3].std()
                    else:
                        stats[snbs_mean_col] = None
                        stats[snbs_std_col] = None
            statistics.append(stats)
        except KeyError as e:
            print(f"Missing column in {filename}: {e}")

    # Convert the list of statistics into a DataFrame
    return pd.DataFrame(statistics)


def get_column_mappings(metric="r2"):
    """
    Return column mappings for image and SNBS results.
    metric: "r2", "ssim", or "lpips"

    Note: SNBS is always evaluated using R², regardless of the metric parameter.
    """
    metric = metric.lower()

    # Define metric prefixes
    metric_prefixes = {
        "r2": "image_test_r2",
        "ssim": "ssim_test",
        "lpips": "lpips_test",
    }

    if metric not in metric_prefixes:
        raise ValueError(f"Unknown metric: {metric}. Use 'r2', 'ssim', or 'lpips'.")

    prefix = metric_prefixes[metric]
    ood_prefix = prefix.replace("_test", "_ood_test")

    # SNBS mappings are always R²
    snbs_column_mappings = {
        "tr20ev20_mean": "snbs_test_r2_mean",
        "tr20ev20_std": "snbs_test_r2_std",
        "tr100ev100_mean": "snbs_test_r2_mean",
        "tr100ev100_std": "snbs_test_r2_std",
        "tr20ev100_mean": "snbs_ood_test_r2_mean",
        "tr20ev100_std": "snbs_ood_test_r2_std",
    }

    # Image mappings vary by metric
    image_column_mappings = {
        "tr20ev20_mean": f"{prefix}_mean",
        "tr20ev20_std": f"{prefix}_std",
        "tr100ev100_mean": f"{prefix}_mean",
        "tr100ev100_std": f"{prefix}_std",
        "tr20ev100_mean": f"{ood_prefix}_mean",
        "tr20ev100_std": f"{ood_prefix}_std",
    }

    return image_column_mappings, snbs_column_mappings


def load_model_with_best_seed(training_dir, seed=None):
    if seed == None:
        seed = get_best_seed(training_dir)
        print("For training_dir: ", training_dir, " the best seed is: ", seed)
    model_config_path = training_dir / "model_config.yaml"
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {model_config_path}")

    with open(model_config_path, "r") as file:
        model_config = yaml.safe_load(file)

    # Define the model
    model = init_model(model_config)

    # Load the model checkpoint
    checkpoint_path = training_dir / f"seed_{seed}_best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


# %%


def get_best_seed(training_dir):
    """
    Looks for the evaluation_results_{training_dir.name}.csv file in the training_dir
    and returns the top (first index) row as a dictionary.

    Args:
        training_dir (Path): Path to the training directory.

    Returns:
        dict: The top row of the CSV file as a dictionary, or None if the file doesn't exist.
    """
    csv_file = training_dir / f"evaluation_results_{training_dir.name}.csv"

    if not csv_file.exists():
        print(f"File {csv_file} does not exist.")
        return None

    # Read the CSV file
    results_df = pd.read_csv(csv_file)

    if results_df.empty:
        print("The CSV file is empty.")
        return None

    # Return the top row as a dictionary
    return results_df["seed"][0]


def populate_critical_cells_df(
    base_dir, filename="metrics_critical_cells_all_seeds_summary.csv"
):
    """
    Recursively reads all critical cells IoU summary CSVs under base_dir and returns a concatenated DataFrame.
    Adds a 'run_dir' column for the parent directory of each file.
    If no files are found, returns an empty DataFrame.
    """
    base_dir = Path(base_dir)
    files = list(base_dir.rglob(filename))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty or len(df.columns) == 0:
                continue  # skip empty files
            df["run_dir"] = f.parent.name
            dfs.append(df)
        except pd.errors.EmptyDataError:
            continue  # skip files that cannot be read
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        print(f"No critical cells summary files found under: {base_dir}")
        return pd.DataFrame()


def lookup_critical_cell_metric(
    critical_cells_df, model_name, dataset_name, metric_col="iou_mean"
):
    """
    Lookup one metric value from critical-cells summary table.
    Returns None if there is no matching row/metric.
    """
    if metric_col not in critical_cells_df.columns:
        return None

    match = critical_cells_df[
        (critical_cells_df["model"] == model_name)
        & (critical_cells_df["dataset"] == dataset_name)
    ]
    if match.empty:
        return None

    value = match.iloc[0][metric_col]
    return None if pd.isna(value) else float(value)


def build_critical_cells_md_table(
    critical_cells_df,
    model_pairs,
    table_spec,
    metric_col="iou_mean",
    digits=3,
    missing="-",
):
    """
    Build a markdown-ready DataFrame for critical-cells metrics.

    model_pairs: dict[str, dict], e.g. {"TAG": {"tr20": "run_x", "tr100": "run_y"}}
    table_spec: list[tuple], e.g. [("tr20ev20", "tr20", "ds20"), ...]
    """
    rows = []
    for model_label, runs in model_pairs.items():
        row = {"Model": model_label}
        for col_label, run_key, eval_dataset in table_spec:
            run_name = runs.get(run_key)
            if run_name is None:
                row[col_label] = missing
                continue

            value = lookup_critical_cell_metric(
                critical_cells_df,
                model_name=run_name,
                dataset_name=eval_dataset,
                metric_col=metric_col,
            )
            row[col_label] = missing if value is None else f"{value:.{digits}f}"
        rows.append(row)

    return pd.DataFrame(rows)


def dataframe_to_markdown_fallback(df):
    """
    Render a DataFrame as a GitHub-style markdown table without requiring tabulate.
    """
    if df is None or df.empty:
        return "| |\n|-|"

    headers = [str(col) for col in df.columns]

    def _cell(v):
        if pd.isna(v):
            text = ""
        else:
            text = str(v)
        return text.replace("|", "\\|")

    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = []
    for _, row in df.iterrows():
        data_rows.append(
            "| " + " | ".join(_cell(row[col]) for col in df.columns) + " |"
        )

    return "\n".join([header_row, sep_row] + data_rows)


def populate_critical_cells_results_df(
    result_rows, critical_cells_df, column_mappings, extra_grids=None
):
    """
    Populate a DataFrame for critical cells IoU results based on mappings and input rows.
    result_rows: dict mapping row names to dicts of dataset/model names
    critical_cells_df: DataFrame loaded from metrics_critical_cells_all_seeds_summary.csv
    column_mappings: dict mapping logical names to CSV columns
    extra_grids: list of extra grid names (optional)
    """
    rows = []
    for row_name, row_info in result_rows.items():
        row_dict = {"model": row_info.get("model", row_name)}
        for key, col_name in column_mappings.items():
            dataset_name = row_info.get(key)
            if dataset_name is None:
                row_dict[key] = None
                continue
            # Find the IoU mean for this model/dataset
            match = critical_cells_df[
                (critical_cells_df["model"] == row_dict["model"])
                & (critical_cells_df["dataset"] == key)
            ]
            if not match.empty:
                row_dict[key] = match.iloc[0]["iou_mean"]
            else:
                row_dict[key] = None
        # Optionally add extra grids
        if extra_grids:
            for grid in extra_grids:
                dataset_name = row_info.get(grid)
                if dataset_name is None:
                    row_dict[grid] = None
                    continue
                match = critical_cells_df[
                    (critical_cells_df["model"] == row_dict["model"])
                    & (critical_cells_df["dataset"] == grid)
                ]
                if not match.empty:
                    row_dict[grid] = match.iloc[0]["iou_mean"]
                else:
                    row_dict[grid] = None
        rows.append(row_dict)
    return pd.DataFrame(rows)
