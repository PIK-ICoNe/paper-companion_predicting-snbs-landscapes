# %%
import copy
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

root_dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir_path))

from src.eval_runs import load_model_with_best_seed
from src.training import _unwrap_model_output, load_datasets


def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_family_rank(model_name: str) -> int:
    upper = str(model_name).upper()
    if "MLP" in upper:
        return 0
    if "CNN" in upper:
        return 1
    if "VIT" in upper:
        return 2
    return 3


def sort_runtime_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "model" not in df.columns:
        return df
    out = df.copy()
    out["_family_rank"] = out["model"].map(_model_family_rank)
    out = out.sort_values(by=["_family_rank", "model"]).drop(columns=["_family_rank"])
    return out.reset_index(drop=True)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Create a simple GitHub-style markdown table without external dependencies."""
    if df.empty:
        return "| (empty) |\n|---|"

    headers = [str(c) for c in df.columns]
    rows = [["" if pd.isna(v) else str(v) for v in row] for row in df.to_numpy()]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt_row(cells):
        return (
            "| "
            + " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells)))
            + " |"
        )

    header_line = _fmt_row(headers)
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    data_lines = [_fmt_row(row) for row in rows]
    return "\n".join([header_line, sep_line, *data_lines])


def measure_inference_runtime(
    model,
    loader,
    device: torch.device,
    warmup_batches: int = 3,
):
    if loader is None or len(loader.dataset) == 0:
        return None, 0, None

    model.eval()

    # Warm-up
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= warmup_batches:
                break
            data = data.to(device)
            output = model(data)
            _unwrap_model_output(output)

    _sync_if_cuda(device)
    start = time.perf_counter()

    n_samples = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            _unwrap_model_output(output)
            n_samples += data.num_graphs if hasattr(data, "num_graphs") else len(data)

    _sync_if_cuda(device)
    elapsed_s = time.perf_counter() - start

    ms_per_sample = (elapsed_s / n_samples * 1000.0) if n_samples > 0 else None
    return elapsed_s, n_samples, ms_per_sample


# %%
if __name__ == "__main__":
    base_dir = Path("../ml_training")
    output_csv = root_dir_path / "csv_results" / "inference_runtime_results.csv"

    # Use the same model naming logic as in show_statistical_results.py
    model_runs = {
        "TAG-VIT": "run_ns20_ds20_TAG_vit",
        "TAG-MLP-ns10": "run_ns10_ds20_TAG",
        "TAG-MLP": "run_ns20_ds20_TAG",
        "TAG-CNN": "run_ns20_ds20_TAG_cnn",
        "DBGNN-CNN": "run_ns20_ds20_DBGNN_cnn",
        "DBGNN-VIT": "run_ns20_ds20_DBGNN_vit",
        "DBGNN-MLP": "run_ns20_ds20_DBGNN",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rows = []
    for model_name, run_name in model_runs.items():
        training_dir = base_dir / run_name
        if not training_dir.exists():
            print(f"Skipping {model_name}: missing directory {training_dir}")
            continue

        training_config_path = training_dir / "training_config.yaml"
        if not training_config_path.exists():
            print(f"Skipping {model_name}: missing {training_config_path}")
            continue

        with open(training_config_path, "r") as file:
            training_config = yaml.safe_load(file)

        model = load_model_with_best_seed(training_dir).to(device)
        best_seed = training_config.get("manual_seed", "unknown")

        # Measure inference time for ds20 and ds100
        ds20_time_s, ds20_n, ds20_ms = None, 0, None
        ds100_time_s, ds100_n, ds100_ms = None, 0, None

        for dataset_name in ["ds20", "ds100"]:
            config = copy.deepcopy(training_config)
            config["dataset_name"] = dataset_name
            loaders = load_datasets(config, train=False, val=False, test=True)
            test_loader = loaders.get("test_loader")
            time_s, n_samples, ms_per_sample = measure_inference_runtime(
                model, test_loader, device
            )
            if dataset_name == "ds20":
                ds20_time_s, ds20_n, ds20_ms = time_s, n_samples, ms_per_sample
            else:
                ds100_time_s, ds100_n, ds100_ms = time_s, n_samples, ms_per_sample

        rows.append(
            {
                "model": model_name,
                "run_name": run_name,
                "best_seed": best_seed,
                "ds20_test_split_runtime_s": ds20_time_s,
                "ds20_test_split_num_samples": ds20_n,
                "ds20_test_split_ms_per_sample": ds20_ms,
                "ds100_test_split_runtime_s": ds100_time_s,
                "ds100_test_split_num_samples": ds100_n,
                "ds100_test_split_ms_per_sample": ds100_ms,
            }
        )

        print(
            f"{model_name}: ds20={ds20_time_s:.3f}s ({ds20_ms:.3f} ms/sample), "
            f"ds100={ds100_time_s:.3f}s ({ds100_ms:.3f} ms/sample)"
        )

    runtime_df = pd.DataFrame(rows)
    if not runtime_df.empty:
        runtime_df = sort_runtime_table(runtime_df)
        runtime_df = runtime_df.round(
            {
                "ds20_test_split_runtime_s": 3,
                "ds20_test_split_ms_per_sample": 3,
                "ds100_test_split_runtime_s": 3,
                "ds100_test_split_ms_per_sample": 3,
            }
        )

    print("\nInference runtime table:")
    print(runtime_df.to_string(index=False))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    runtime_df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")


# %%
runtime_csv_path = root_dir_path / "csv_results" / "inference_runtime_results.csv"
if runtime_csv_path.exists():
    markdown_out_path = root_dir_path / "csv_results" / "inference_runtime_results.md"
    latex_out_path = root_dir_path / "csv_results" / "inference_runtime_results.tex"

    runtime_table_df = pd.read_csv(runtime_csv_path)
    runtime_table_df = sort_runtime_table(runtime_table_df)
    runtime_table_df = runtime_table_df.round(
        {
            "ds20_test_split_runtime_s": 3,
            "ds20_test_split_ms_per_sample": 3,
            "ds100_test_split_runtime_s": 3,
            "ds100_test_split_ms_per_sample": 3,
        }
    )

    required_cols = ["model", "ds20_test_split_runtime_s", "ds100_test_split_runtime_s"]
    missing_cols = [c for c in required_cols if c not in runtime_table_df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required runtime columns: {missing_cols}. "
            f"Available columns: {list(runtime_table_df.columns)}"
        )

    table_df = runtime_table_df[required_cols].rename(
        columns={
            "model": "Model",
            "ds20_test_split_runtime_s": "Runtime ds20 test split [s]",
            "ds100_test_split_runtime_s": "Runtime ds100 test split [s]",
        }
    )

    print("\nMarkdown table:")
    markdown_table = dataframe_to_markdown(table_df)
    print(markdown_table)
    markdown_out_path.write_text(markdown_table + "\n", encoding="utf-8")
    print(f"Saved markdown table: {markdown_out_path}")

    print("\nLaTeX table:")
    latex_table = table_df.to_latex(index=False, escape=False)
    print(latex_table)
    latex_out_path.write_text(latex_table + "\n", encoding="utf-8")
    print(f"Saved LaTeX table: {latex_out_path}")
else:
    print(f"CSV not found: {runtime_csv_path}")


# %%
