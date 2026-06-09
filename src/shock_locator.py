import torch
import numpy as np
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score
from scipy.stats import spearmanr

from src.training import _unwrap_model_output


def get_topk_location(heatmap, threshold=0.7, k=20):
    """
    Return the k indices closest to the center (mirrored to the top) and their probabilities.
    """
    mask = heatmap > threshold
    indices = mask.nonzero(as_tuple=False)  # (N, 2)
    ps = heatmap[mask]

    if indices.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long), torch.empty((0,), dtype=ps.dtype)

    center = torch.tensor(heatmap.shape, dtype=torch.float32) / 2 - 0.5
    distances = torch.norm(indices.float() - center, dim=1)

    distances_np = distances.cpu().numpy()
    ps_np = ps.cpu().numpy()
    indices_np = indices.cpu().numpy()

    sort_order = np.lexsort((-ps_np, distances_np))

    topk = indices_np[sort_order[:k]].copy()
    topk_ps = ps_np[sort_order[:k]].copy()

    height = heatmap.shape[0]
    topk[:, 0] = height - 1 - topk[:, 0]

    return torch.from_numpy(topk), torch.from_numpy(topk_ps)


def get_shock_location(heatmap, threshold=0.7):
    """
    Returns the 20 indices closest to the center (mirrored to the top) and their probabilities.
    Args:
        heatmap (torch.Tensor): 2D tensor (H, W)
        threshold (float): threshold for mask
    Returns:
        top20 (torch.Tensor): (20, 2) indices, mirrored on y-axis (row)
        top20_ps (torch.Tensor): (20,) probabilities
    """
    return get_topk_location(heatmap, threshold=threshold, k=20)


def get_gridwise_top20_dangerous_cells(heatmaps, threshold=0.7):
    """
    For a grid (all nodes), find the 20 unstable cells (across all nodes) that are closest to the center.
    Args:
        heatmaps (list or array): list of 2D torch.Tensor heatmaps (one per node)
        threshold (float): threshold for instability
    Returns:
        top20_indices (np.ndarray): (20, 3) array of (node_idx, y, x)
        top20_ps (np.ndarray): (20,) probabilities
    """
    return get_gridwise_topk_dangerous_cells(heatmaps, threshold=threshold, k=20)


def get_gridwise_topk_dangerous_cells(heatmaps, threshold=0.7, k=20):
    """
    For a grid (all nodes), find the k unstable cells across all nodes that are closest to the center.
    """
    all_indices = []
    all_ps = []
    all_node_idx = []
    for node_idx, heatmap in enumerate(heatmaps):
        mask = heatmap > threshold
        indices = mask.nonzero(as_tuple=False)
        ps = heatmap[mask]
        if indices.numel() > 0:
            all_indices.append(indices)
            all_ps.append(ps)
            all_node_idx.append(
                torch.full((indices.shape[0], 1), node_idx, dtype=torch.int)
            )
    if not all_indices:
        raise ValueError("no critical cells found")

    all_indices = torch.cat(all_indices, dim=0)
    all_ps = torch.cat(all_ps, dim=0)
    all_node_idx = torch.cat(all_node_idx, dim=0)
    all_full_indices = torch.cat([all_node_idx, all_indices], dim=1)

    h, w = heatmaps[0].shape
    center = torch.tensor([h / 2 - 0.5, w / 2 - 0.5], dtype=torch.float32)
    distances = torch.norm(all_indices.float() - center, dim=1)

    distances_np = distances.cpu().numpy()
    ps_np = all_ps.cpu().numpy()
    full_indices_np = all_full_indices.cpu().numpy()
    sort_order = np.lexsort((-ps_np, distances_np))
    topk_indices = full_indices_np[sort_order[:k]].copy()
    topk_ps = ps_np[sort_order[:k]].copy()
    return topk_indices, topk_ps


def compute_shock_metrics_per_node(true_heatmap, pred_heatmap, threshold=0.7):
    """
    Compute IoU, precision, recall, F1, and Spearman correlation for shock locations and probabilities.
    Args:
        true_heatmap (torch.Tensor): 2D tensor (H, W)
        pred_heatmap (torch.Tensor): 2D tensor (H, W)
        threshold (float): threshold for mask
    Returns:
        dict: metrics with keys 'iou', 'precision', 'recall', 'f1', 'spearman'
    """
    # Get top-k indices and probabilities; use top30 so coverage@25/@30 is available.
    shock_pos_true, shock_probs_true = get_topk_location(
        true_heatmap, threshold=threshold, k=30
    )
    shock_pos_pred, shock_probs_pred = get_topk_location(
        pred_heatmap, threshold=threshold, k=30
    )

    grid_shape = true_heatmap.shape
    metrics = compute_topk_metrics(
        shock_pos_true[:20], shock_pos_pred[:20], grid_shape=grid_shape
    )
    metrics.update(
        compute_topk_overlap_metrics(
            shock_pos_true[:20], shock_pos_pred, pred_ks=(20, 25, 30)
        )
    )

    spearman_corr, _ = spearmanr(
        shock_probs_true[:20].cpu().numpy(), shock_probs_pred[:20].cpu().numpy()
    )
    metrics["spearman"] = spearman_corr
    return metrics


def compute_topk_overlap_metrics(
    topk_indices_true, topk_indices_pred, pred_ks=(20, 25, 30)
):
    """
    Compute how much of the true top-k set is covered by increasingly larger predicted top-k sets.

    For example, coverage_at_25 is the fraction of true top-20 cells that appear in the
    first 25 predicted cells.
    """
    true_arr = np.array(topk_indices_true)
    pred_arr = np.array(topk_indices_pred)

    if true_arr.size == 0 or pred_arr.size == 0:
        return {f"coverage_at_{k}": np.nan for k in pred_ks}

    true_set = set(map(tuple, true_arr))
    true_denominator = len(true_set)
    if true_denominator == 0:
        return {f"coverage_at_{k}": np.nan for k in pred_ks}

    metrics = {}
    for k in pred_ks:
        pred_subset = pred_arr[:k]
        pred_set = set(map(tuple, pred_subset))
        metrics[f"coverage_at_{k}"] = len(true_set & pred_set) / true_denominator
    return metrics


def get_gridwise_top20_dangerous_cells_from_model(
    data_loader, model, grid_index, threshold=0.7
):
    """
    For a grid (all nodes), find the 20 unstable cells (across all nodes) that are closest to the center.
    Args:
        data_loader: DataLoader with dataset containing y and sample_heatmaps
        model: model to predict heatmaps
        grid_index: index of the grid in the dataset
        threshold (float): threshold for instability
    Returns:
        top20_indices_true, top20_ps_true: (20, 3) indices and probabilities for true heatmaps
        top20_indices_pred, top20_ps_pred: (20, 3) indices and probabilities for predicted heatmaps
    """
    dataset = data_loader.dataset
    num_nodes = dataset[grid_index].y.shape[0]
    true_heatmaps = []
    pred_heatmaps = []
    model.eval()
    with torch.no_grad():
        output = model(dataset[grid_index])
        pred_heatmap_full, _, _ = _unwrap_model_output(output)
    for node_idx in range(num_nodes):
        true_heatmap = dataset[grid_index].y[node_idx]
        pred_heatmap = pred_heatmap_full[node_idx]
        true_heatmaps.append(true_heatmap)
        pred_heatmaps.append(pred_heatmap)
    topk_indices_true, topk_ps_true = get_gridwise_topk_dangerous_cells(
        true_heatmaps, threshold, k=20
    )
    topk_indices_pred, topk_ps_pred = get_gridwise_topk_dangerous_cells(
        pred_heatmaps, threshold, k=20
    )
    return topk_indices_true, topk_ps_true, topk_indices_pred, topk_ps_pred


def get_gridwise_top30_dangerous_cells_from_model(
    data_loader, model, grid_index, threshold=0.7
):
    """
    For a grid (all nodes), find the 30 unstable cells (across all nodes) that are closest to the center.

    This helper is intended for relaxed coverage metrics only. It does not change
    the exact top-20 critical-cell prediction returned by
    get_gridwise_top20_dangerous_cells_from_model().
    """
    dataset = data_loader.dataset
    num_nodes = dataset[grid_index].y.shape[0]
    true_heatmaps = []
    pred_heatmaps = []
    model.eval()
    with torch.no_grad():
        output = model(dataset[grid_index])
        pred_heatmap_full, _, _ = _unwrap_model_output(output)
    for node_idx in range(num_nodes):
        true_heatmap = dataset[grid_index].y[node_idx]
        pred_heatmap = pred_heatmap_full[node_idx]
        true_heatmaps.append(true_heatmap)
        pred_heatmaps.append(pred_heatmap)
    topk_indices_true, topk_ps_true = get_gridwise_topk_dangerous_cells(
        true_heatmaps, threshold, k=30
    )
    topk_indices_pred, topk_ps_pred = get_gridwise_topk_dangerous_cells(
        pred_heatmaps, threshold, k=30
    )
    return topk_indices_true, topk_ps_true, topk_indices_pred, topk_ps_pred


def compute_topk_metrics(topk_indices_true, topk_indices_pred, grid_shape=None):
    """
    Compute IoU, precision, recall, and F1 score for two sets of top-k indices.
    Optionally, if grid_shape is provided, computes metrics using binary masks.
    Args:
        topk_indices_true: (K, 2) or (K, N) array-like, true indices
        topk_indices_pred: (K, 2) or (K, N) array-like, predicted indices
        grid_shape: tuple, shape of the grid (optional, for mask-based metrics)
    Returns:
        dict with iou, precision, recall, f1
    """
    # Ensure numpy arrays
    a = np.array(topk_indices_true)
    b = np.array(topk_indices_pred)
    # If either is empty, return NaN for all metrics
    if a.size == 0 or b.size == 0:
        return dict(iou=np.nan, precision=np.nan, recall=np.nan, f1=np.nan)
    set_a = set(map(tuple, a))
    set_b = set(map(tuple, b))

    intersection = set_a & set_b
    union = set_a | set_b

    iou = len(intersection) / len(union) if union else np.nan
    metrics = dict(iou=iou)

    # Optionally, compute mask-based metrics using torchmetrics
    if grid_shape is not None and a.size > 0 and b.size > 0:
        mask_true = torch.zeros(grid_shape, dtype=torch.bool)
        mask_pred = torch.zeros(grid_shape, dtype=torch.bool)
        for idx in a:
            mask_true[tuple(idx)] = 1
        for idx in b:
            mask_pred[tuple(idx)] = 1
        mask_true = mask_true.flatten()
        mask_pred = mask_pred.flatten()
        precision_metric = BinaryPrecision()
        recall_metric = BinaryRecall()
        f1_metric = BinaryF1Score()
        metrics["precision_mask"] = precision_metric(mask_pred, mask_true).item()
        metrics["recall_mask"] = recall_metric(mask_pred, mask_true).item()
        metrics["f1_mask"] = f1_metric(mask_pred, mask_true).item()

    return metrics
