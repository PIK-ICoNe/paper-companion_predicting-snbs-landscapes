# %%
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import yaml
import torch.nn as nn
import copy

root_dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir_path))


from src.hyperparameter_study import (
    print_plot_study,
    plot_loss_over_epochs,
    load_best_model,
)

from src.training import load_datasets, eval_loop
from src.plotting import (
    show_heatmaps_for_grid_node,
    show_multiple_side_by_side_heatmaps,
)

# %%
device = "cuda"

# Define the training directory
training_dir = Path(
    "/home/nauck/joined_work/landscape_generation/ml_training/ns10_ds20_op01"
)

# %%
print_plot_study(training_dir)

# %%
# Plot the loss over epochs
plot_loss_over_epochs(training_dir, metric="R2")

# %%
best_model = load_best_model(training_dir)
num_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
print("num_params: ", num_params)

best_model.to(device)
training_config_path = training_dir / "training_config.yaml"

with open(training_config_path, "r") as file:
    training_config = yaml.safe_load(file)

# %%
# Load the datasets
loaders = load_datasets(training_config, train=True, val=True, test=True)
train_loader = loaders.get("train_loader")
val_loader = loaders.get("val_loader")
test_loader = loaders.get("test_loader")

# %%
train_loss, train_R2 = eval_loop(
    best_model.to(device), nn.MSELoss(), train_loader, device
)
val_loss, val_R2 = eval_loop(best_model.to(device), nn.MSELoss(), val_loader, device)
# test_loss, test_R2 = eval_loop(best_model, nn.MSELoss(), test_loader, device)

print(f"train R2: {train_R2}")
print(f"valid R2: {val_R2}")
# print(f"test R2: "{test_R2})


# %%
show_heatmaps_for_grid_node(train_loader, best_model.to("cpu"), 5, 11)

# %%
show_heatmaps_for_grid_node(train_loader, best_model.to("cpu"), 6300, 8)
# %%
show_heatmaps_for_grid_node(val_loader, best_model.to("cpu"), 5, 19)
# %%
ood_100_config = copy.deepcopy(training_config)
ood_100_config["dataset_name"] = "ds100"
ood_100_config["num_sections"] = 20
ood_100_config["test_slice_index"] = [8501, 10000]
ood_100_loaders = load_datasets(ood_100_config, train=False, val=True, test=True)
ood_100_loader_val = ood_100_loaders["val_loader"]
ood_100_loader_test = ood_100_loaders["test_loader"]
# %%
ood_100_loss_val, ood_100_R2_val = eval_loop(
    best_model.to(device), nn.MSELoss(), ood_100_loader_val, device
)
print(f"valid ood 100 R2: {ood_100_R2_val}")
# %%
show_heatmaps_for_grid_node(ood_100_loader_val, best_model.to("cpu"), 5, 19)

# %%
show_multiple_side_by_side_heatmaps(
    train_loader, best_model.to("cpu"), grid_index=0, node_range=range(6)
)

# %%
grid_index = 0
node_idxs = [0, 3, 7, 11, 15, 19]
show_multiple_side_by_side_heatmaps(
    train_loader,
    best_model.to("cpu"),
    grid_index=grid_index,
    node_range=node_idxs,
    save_fig_name=f"{root_dir_path}/pics/heatmaps_train_grid_{grid_index}.png",
)


# %%
grid_index = 0
node_idxs = [0, 3, 7, 11, 15, 19]
show_multiple_side_by_side_heatmaps(
    test_loader,
    best_model.to("cpu"),
    grid_index=grid_index,
    node_range=node_idxs,
    save_fig_name=f"{root_dir_path}/pics/heatmaps_test_grid_{grid_index}.png",
)


# %%
grid_index = 0
node_idxs = [0, 3, 7, 11, 15, 19]
show_multiple_side_by_side_heatmaps(
    ood_100_loader_test,
    best_model.to("cpu"),
    grid_index=grid_index,
    node_range=node_idxs,
    save_fig_name=f"{root_dir_path}/pics/heatmaps_ood100_test_grid_{grid_index}.png",
)

# %%
