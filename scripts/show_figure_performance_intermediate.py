# %%
from __future__ import annotations

from pathlib import Path as _Path
import sys

root_dir_path = _Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir_path))

from src.eval_runs import (
    plot_metric_over_epoch_comparison,
    plot_metrics_grid_over_epoch_comparison,
)

training_dir = "../../ml_training"

# %%
# Runs used in the 2x2 comparison plots
runs = [
    _Path(training_dir, "run_ns20_ds20_DBGNN_intermediate"),
    _Path(training_dir, "run_ns20_ds100_DBGNN_intermediate_1000epochs"),
    _Path(training_dir, "run_ns20_ds20_TAG_intermediate"),
    _Path(training_dir, "run_ns20_ds100_TAG_intermediate"),
]

# %%
# Save SSIM comparison (2x2, each subplot contains all tasks)
ssim_out = _Path("../pics/ssim_comparison.png")
plot_metric_over_epoch_comparison(runs, "ssim", out_path=ssim_out, ylabel="SSIM")

# %%
# Save image R² comparison (2x2, each subplot contains all tasks)
image_r2_out = _Path("../pics/image_r2_comparison.png")
plot_metric_over_epoch_comparison(
    runs,
    "image_r2",
    out_path=image_r2_out,
    ylabel="Image R²",
    y_lim=(-1.0, 1.0),
)

# %%
# Save LPIPS comparison (2x2, each subplot contains all tasks)
lpips_out = _Path("../pics/lpips_comparison.png")
plot_metric_over_epoch_comparison(runs, "lpips", out_path=lpips_out, ylabel="LPIPS")

# %%
# Focused ds100 view: 6 subplots total
# rows = metrics [image_r2, ssim, lpips]
# cols = runs [DBGNN, TAG]
# each subplot contains all tasks/datasets
runs_ns20_ds100 = [
    _Path(training_dir, "run_ns20_ds100_DBGNN_intermediate_1000epochs"),
    _Path(training_dir, "run_ns20_ds100_TAG_intermediate"),
]

metrics_grid_out = _Path("../pics/ns20_ds100_dbg_vs_tag_metrics_grid.png")
plot_metrics_grid_over_epoch_comparison(
    runs_ns20_ds100,
    metrics=("image_r2", "ssim", "lpips"),
    out_path=metrics_grid_out,
    y_lims={"image_r2": (-1.0, 1.0)},
)

# %%
