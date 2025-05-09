# %%
import torch.optim as optim
from torch_geometric.data import DataLoader
import torch.nn as nn
import torch
import os
import yaml
import argparse


# %%
import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_path))

from gnn import init_model
from training import (
    train_loop,
    eval_loop,
    setup_training,
    load_datasets,
    save_checkpoint,
    load_checkpoint,
)


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run Optuna hyperparameter study")
parser.add_argument(
    "job_id", type=str, nargs="?", default=None, help="Job ID for the study"
)

try:
    args = parser.parse_args()
    job_id = args.job_id
except SystemExit:
    job_id = "no_job_id"


# Load configurations
config_dir = Path(__file__).resolve().parent.parent / "config"
model_config_path = config_dir / "model_config.yaml"
training_config_path = config_dir / "training_config.yaml"

if not model_config_path.exists() or not training_config_path.exists():
    raise FileNotFoundError(
        "Configuration files not found. Ensure 'model_config.yaml' and 'training_config.yaml' are present in the 'config' directory."
    )

with open(model_config_path, "r") as file:
    model_config = yaml.safe_load(file)

with open(training_config_path, "r") as file:
    training_config = yaml.safe_load(file)


# Set image_size based on num_sections
model_config["image_size"] = training_config["num_sections"]

# Write the updated model_config to model_config.yaml
with open(model_config_path, "w") as file:
    yaml.safe_dump(model_config, file)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_dir = Path(training_config["training_dir"])

# Setup training directory and copy config files
setup_training(training_config, [model_config_path, training_config_path])

# Save a file named after the job_id
output_file = training_dir / f"job_id_{job_id}.txt"

# Load the datasets
loaders = load_datasets(training_config, train=True, val=True, test=False)
train_loader = loaders.get("train_loader")
val_loader = loaders.get("val_loader")

# Define the model, loss function, and optimizer
model = init_model(model_config).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("num_params: ", num_params)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])

# Load checkpoint if exists
checkpoint_path = os.path.join(training_dir, "best_model.pt")
if os.path.exists(checkpoint_path):
    start_epoch, val_loss_best = load_checkpoint(checkpoint_path, model, optimizer)
else:
    start_epoch = 0
    val_loss_best = 1e10

num_epochs = training_config["num_epochs"]
for epoch in range(start_epoch, num_epochs):
    train_loss, train_R2 = train_loop(model, optimizer, criterion, train_loader, device)
    if (epoch + 1) % training_config["eval_interval"] == 0:
        val_loss, val_R2 = eval_loop(model, criterion, val_loader, device)
        train_loss_sci = f"{train_loss/len(train_loader):.4e}"
        val_loss_sci = f"{val_loss/len(val_loader):.4e}"
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_sci}, Train R2: {train_R2:.3f}"
        )
        print(f"Valid Loss: {val_loss_sci}, Val R2: {val_R2:.3f}")
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            save_checkpoint(checkpoint_path, model, optimizer, epoch, val_loss_best)

# %%
