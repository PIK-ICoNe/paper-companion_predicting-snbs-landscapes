# %%
from pathlib import Path
import sys

root_dir_path = Path(__file__).resolve().parent.parent

root_dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir_path))

from src.eval_runs import load_model_with_best_seed


from src.training import load_datasets
from src.plotting import (
    show_heatmap,
    show_multiple_side_by_side_heatmaps,
)

from src.shock_locator import get_shock_location, compute_shock_metrics_per_node


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

loaders_20 = load_datasets(dataset_20_config, train=True, val=True, test=True)
train_loader_20 = loaders_20.get("train_loader")
val_loader_20 = loaders_20.get("val_loader")
test_loader_20 = loaders_20.get("test_loader")

# loaders_100 = load_datasets(dataset_100_config, train=True, val=True, test=True)
# train_loader_100 = loaders_100.get("train_loader")
# val_loader_100 = loaders_100.get("val_loader")
# test_loader_100 = loaders_100.get("test_loader")

# %%
training_dir_TAGtr20 = Path("../ml_training/run_bo_opt010")
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
show_multiple_side_by_side_heatmaps(
    test_loader_20,
    model_TAG_tr20.to("cpu"),
    grid_index=grid_index,
    node_range=node_idxs,
    save_fig_name=None,
    highlight_indices=True,
)

# %%
data_loader = test_loader_20
model = model_TAG_tr20
node_index = 0
true_heatmap = data_loader.dataset[grid_index].y[node_index]
sample_heatmap = data_loader.dataset[grid_index].sample_heatmaps[node_index]
model.eval()
predicted_heatmap = model(data_loader.dataset[grid_index])[node_index]
metrics_oer_node = compute_shock_metrics_per_node(
    true_heatmap, predicted_heatmap.detach()
)
# %%
show_heatmap(true_heatmap.numpy(), sample_heatmap.numpy())

# %%
show_heatmap(predicted_heatmap.detach().numpy(), sample_heatmap.numpy())
# %%
shock_pos_true, shock_probs_true = get_shock_location(true_heatmap)
show_heatmap(true_heatmap, sample_heatmap, highlight_indices=shock_pos_true)
# %%
shock_pos_pred, shock_probs_pred = get_shock_location(predicted_heatmap.detach())
show_heatmap(
    predicted_heatmap.detach(), sample_heatmap, highlight_indices=shock_pos_pred
)
# %%
metrics_oer_node = compute_shock_metrics_per_node(
    true_heatmap, predicted_heatmap.detach()
)
