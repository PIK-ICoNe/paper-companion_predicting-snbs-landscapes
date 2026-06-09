# %%
import torch.optim as optim
from torch_geometric.data import DataLoader
import torch.nn as nn
import torch
import os
import yaml


# %%
import sys
from pathlib import Path

root_dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir_path))

from src.oscillator_landscape import oscillatorLandscapceDataset
from src.gnn import init_model
from src.training import load_datasets, eval_loop
from src.plotting import show_side_by_side_heatmaps, show_heatmaps_for_grid_node

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cpu"

training_dir = Path("/home/nauck/joined_work/landscape_generation/ml_training/run001")


training_config_path = training_dir / "training_config.yaml"

with open(training_config_path, "r") as file:
    training_config = yaml.safe_load(file)


# Load the datasets
loaders = load_datasets(training_config, train=True, val=True, test=False)
train_loader = loaders.get("train_loader")
val_loader = loaders.get("val_loader")

model_config_path = training_dir / "model_config.yaml"
with open(model_config_path, "r") as file:
    model_config = yaml.safe_load(file)

model = init_model(model_config)
model.load_checkpoint(os.path.join(training_dir, "best_model.pt"))
# %%

show_heatmaps_for_grid_node(train_loader, model, 0, 5)

# %%

show_heatmaps_for_grid_node(train_loader, model, 0, 10)
# %%
for i in range(20):
    show_heatmaps_for_grid_node(train_loader, model, 0, i)

# %%
