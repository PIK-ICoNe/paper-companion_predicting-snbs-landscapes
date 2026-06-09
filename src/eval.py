import torch
import numpy as np


from src.shock_locator import (
    get_gridwise_top20_dangerous_cells_from_model,
    get_gridwise_top30_dangerous_cells_from_model,
    compute_topk_metrics,
    compute_topk_overlap_metrics,
)
from src.training import _unwrap_model_output


def get_true_and_predicted_heatmaps(data_loader, model, grid_index, node_index):
    """
    Returns (true_heatmap, predicted_heatmap, sample_heatmap) for a given grid and node.
    """
    true_heatmap = data_loader.dataset[grid_index].y[node_index]
    sample_heatmap = data_loader.dataset[grid_index].sample_heatmaps[node_index]
    model.eval()
    with torch.no_grad():
        output = model(data_loader.dataset[grid_index])
        predicted_heatmap, _, _ = _unwrap_model_output(output)
        predicted_heatmap = predicted_heatmap[node_index]
    return true_heatmap, predicted_heatmap, sample_heatmap


def evaluate_gridwise_top20(data_loader, model, threshold=0.7):
    all_metrics = []
    num_grids = len(data_loader.dataset)
    for grid_index in range(num_grids):
        grid_shape = data_loader.dataset[grid_index].y.shape
        # Compute exact top-20 for true and predicted heatmaps
        top20_indices_true, top20_ps_true, top20_indices_pred, top20_ps_pred = (
            get_gridwise_top20_dangerous_cells_from_model(
                data_loader, model, grid_index, threshold=threshold
            )
        )
        # Compute relaxed coverage metrics separately so they do not change the
        # exact top-20 critical-cell prediction.
        top30_indices_true, _, top30_indices_pred, _ = (
            get_gridwise_top30_dangerous_cells_from_model(
                data_loader, model, grid_index, threshold=threshold
            )
        )

        # Compute metrics for this grid.
        metrics = compute_topk_metrics(
            top20_indices_true, top20_indices_pred, grid_shape=grid_shape
        )
        metrics.update(
            compute_topk_overlap_metrics(
                top30_indices_true[:20], top30_indices_pred, pred_ks=(20, 25, 30)
            )
        )
        if len(top20_ps_true) > 0 and len(top20_ps_pred) > 0:
            from scipy.stats import spearmanr

            metrics["spearman"] = spearmanr(top20_ps_true, top20_ps_pred)[0]
        else:
            metrics["spearman"] = np.nan
        all_metrics.append(metrics)
        # print(f"Grid {grid_index}: {metrics}")

    # Aggregate metrics (mean over all grids)
    keys = all_metrics[0].keys()
    mean_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in keys}
    print("\nMean metrics over all grids of the dataloader:")
    for k, v in mean_metrics.items():
        print(f"{k}: {v:.4f}")
    return all_metrics, mean_metrics
