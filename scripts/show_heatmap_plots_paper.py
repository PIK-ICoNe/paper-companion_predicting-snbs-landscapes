# %%
from pathlib import Path
import sys

root_dir_path = Path(__file__).resolve().parent.parent

root_dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir_path))

from src.eval_runs import load_model_with_best_seed


from src.training import load_datasets
from src.plotting import show_multiple_side_by_side_heatmaps


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

loaders_100 = load_datasets(dataset_100_config, train=True, val=True, test=True)
train_loader_100 = loaders_100.get("train_loader")
val_loader_100 = loaders_100.get("val_loader")
test_loader_100 = loaders_100.get("test_loader")

# %%
# training_dir_TAGtr20 = Path("../ml_training/run_bo_opt010")
training_dir_TAGtr20 = Path("../ml_training/run_ns20_ds20_TAG")
model_TAG_tr20 = load_model_with_best_seed(training_dir_TAGtr20, 1)

# training_dir_TAGtr100 = Path("../ml_training/run_bo_opt017_1")
training_dir_TAGtr100 = Path("../ml_training/run_ns20_ds100_TAG")
model_TAG_tr100 = load_model_with_best_seed(training_dir_TAGtr100)

# training_dir_DBGNNtr20 = Path("../ml_training/run_bo_opt015")
training_dir_DBGNNtr20 = Path("../ml_training/run_ns20_ds20_DBGNN")
model_DBGNN_tr20 = load_model_with_best_seed(training_dir_DBGNNtr20)


# training_dir_DBGNNtr100 = Path("../ml_training/run_bo_opt016")
training_dir_DBGNNtr100 = Path("../ml_training/run_ns20_ds100_DBGNN")
model_DBGNN_tr100 = load_model_with_best_seed(training_dir_DBGNNtr100)


# %%
grid_index = 0
node_idxs = [0, 3, 7, 11, 15, 19]

# # %%
show_multiple_side_by_side_heatmaps(
    test_loader_20,
    model_TAG_tr20.to("cpu"),
    grid_index=grid_index,
    node_range=node_idxs,
    save_fig_name=f"{root_dir_path}/pics/heatmaps_test_TAG_tr20ev20_grid_{grid_index}.png",
)

show_multiple_side_by_side_heatmaps(
    test_loader_100,
    model_TAG_tr20.to("cpu"),
    grid_index=grid_index,
    node_range=node_idxs,
    save_fig_name=f"{root_dir_path}/pics/heatmaps_test_TAG_tr20ev100_grid_{grid_index}.png",
)

show_multiple_side_by_side_heatmaps(
    test_loader_100,
    model_TAG_tr100.to("cpu"),
    grid_index=grid_index,
    node_range=node_idxs,
    save_fig_name=f"{root_dir_path}/pics/heatmaps_test_TAG_tr100ev100_grid_{grid_index}.png",
)

# %%
# Add extra grids if desired
extra_grids = ["osf_france", "osf_gb", "osf_spain", "elmod"]

# Example: Load and plot for extra grids
for grid_name in extra_grids:
    dataset_extra_config = dataset_config.copy()
    dataset_extra_config["dataset_name"] = grid_name
    # Let the dataset class auto-detect the available test indices
    dataset_extra_config["test_slice_index"] = [1, 1]
    loaders_extra = load_datasets(
        dataset_extra_config, train=False, val=False, test=True
    )
    test_loader_extra = loaders_extra.get("test_loader")
    # For extra grids we use the model trained on ds100 (tr100)
    show_multiple_side_by_side_heatmaps(
        test_loader_extra,
        model_TAG_tr100.to("cpu"),
        grid_index=0,
        node_range=node_idxs,
        save_fig_name=f"{root_dir_path}/pics/heatmaps_test_{grid_name}_grid_0.png",
    )

# %%
