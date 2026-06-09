# %%
from pathlib import Path
import sys
import matplotlib.pyplot as plt

root_dir_path = Path(__file__).resolve().parent.parent

root_dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir_path))

from src.eval_runs import load_model_with_best_seed


from src.training import load_datasets
from src.plotting import (
    show_heatmap,
    show_multiple_side_by_side_heatmaps,
    show_side_by_side_heatmaps,
)

from src.shock_locator import (
    get_shock_location,
    compute_shock_metrics_per_node,
    compute_topk_metrics,
    get_gridwise_top20_dangerous_cells_from_model,
)

from src.eval import get_true_and_predicted_heatmaps

# %%
dataset_config = {}
dataset_config["dataset_root_path"] = (
    "/home/nauck/joined_work/landscape_generation/datasets/root_path"
)
dataset_config["num_sections"] = 20
dataset_config["batch_size"] = 20
dataset_config["train_slice_index"] = [1, 7000]
dataset_config["val_slice_index"] = [7001, 8500]
dataset_config["test_slice_index"] = [8501, 10000]

dataset_20_config = dataset_config.copy()
dataset_20_config["dataset_name"] = "ds20"

dataset_100_config = dataset_config.copy()
dataset_100_config["dataset_name"] = "ds100"

dataset_france_config = dataset_config.copy()
dataset_france_config["dataset_name"] = "osf_france"
dataset_france_config["test_slice_index"] = [1, 1]


loaders_20 = load_datasets(dataset_20_config, train=True, val=True, test=True)
train_loader_20 = loaders_20.get("train_loader")
val_loader_20 = loaders_20.get("val_loader")
test_loader_20 = loaders_20.get("test_loader")

# loaders_100 = load_datasets(dataset_100_config, train=True, val=True, test=True)
# train_loader_100 = loaders_100.get("train_loader")
# val_loader_100 = loaders_100.get("val_loader")
# test_loader_100 = loaders_100.get("test_loader")

loaders_france = load_datasets(
    dataset_france_config, train=False, val=False, test=True
).get("test_loader")

# %%
# training_dir_TAGtr20 = Path("../ml_training/run_bo_opt010")
training_dir_TAGtr20 = Path("../ml_training/run_ns20_ds20_TAG")
model_TAG_tr20 = load_model_with_best_seed(training_dir_TAGtr20, 1)

# training_dir_TAGtr100 = Path("../ml_training/run_bo_opt017_1")
# model_TAG_tr100 = load_model_with_best_seed(training_dir_TAGtr100)

# training_dir_DBGNNtr20 = Path("../ml_training/run_bo_opt015")
# model_DBGNN_tr20 = load_model_with_best_seed(training_dir_DBGNNtr20)


# training_dir_DBGNNtr100 = Path("../ml_training/run_bo_opt016")
# model_DBGNN_tr100 = load_model_with_best_seed(training_dir_DBGNNtr100)


# %%
grid_index = 0
node_idxs = [0, 3, 7, 11, 15, 19]
# %%
data_loader = test_loader_20
# data_loader = loaders_france
model = model_TAG_tr20
top20_indices_true, top20_ps_true, top20_indices_pred, top20_ps_pred = (
    get_gridwise_top20_dangerous_cells_from_model(
        data_loader, model, grid_index, threshold=0.7
    )
)

metrics = compute_topk_metrics(top20_indices_true, top20_indices_pred)
true_heatmap_idx16, pred_heatmap_idx16, sample_heatmap_idx16 = (
    get_true_and_predicted_heatmaps(
        test_loader_20, model_TAG_tr20, grid_index=0, node_index=16
    )
)

true_heatmap_idx17, pred_heatmap_idx17, sample_heatmap_idx17 = (
    get_true_and_predicted_heatmaps(
        test_loader_20, model_TAG_tr20, grid_index=0, node_index=17
    )
)

top20_indices_true_node16 = top20_indices_true[top20_indices_true[:, 0] == 16][:, 1:]
top20_indices_true_node17 = top20_indices_true[top20_indices_true[:, 0] == 17][:, 1:]

top20_indices_pred_node16 = top20_indices_pred[top20_indices_pred[:, 0] == 16][:, 1:]
top20_indices_pred_node17 = top20_indices_pred[top20_indices_pred[:, 0] == 17][:, 1:]

# %%
# fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # landscape
fig, axes = plt.subplots(2, 1, figsize=(4, 6))  # portrait
show_heatmap(
    true_heatmap_idx16,
    sample_heatmap_idx16,
    highlight_indices=top20_indices_true_node16,
    flip_highlight_indices=True,
    vmin=0,
    vmax=1.0,
    ax=axes[0],
)

show_heatmap(
    pred_heatmap_idx16,
    sample_heatmap_idx16,
    highlight_indices=top20_indices_pred_node16,
    flip_highlight_indices=True,
    ax=axes[1],
)

plt.tight_layout(h_pad=0.0)
plt.savefig("../pics/cells_TAGMLP_ds20_testidx0_node16", bbox_inches="tight")

# %%
# fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # landscape
fig, axes = plt.subplots(2, 1, figsize=(4, 6))  # portrait
show_heatmap(
    true_heatmap_idx17,
    sample_heatmap_idx17,
    highlight_indices=top20_indices_true_node17,
    flip_highlight_indices=True,
    ax=axes[0],
)

show_heatmap(
    pred_heatmap_idx17,
    sample_heatmap_idx17,
    highlight_indices=top20_indices_pred_node17,
    flip_highlight_indices=True,
    ax=axes[1],
)

plt.tight_layout(h_pad=0.0)
plt.savefig("../pics/cells_TAGMLP_ds20_testidx0_node17", bbox_inches="tight")

# %%
