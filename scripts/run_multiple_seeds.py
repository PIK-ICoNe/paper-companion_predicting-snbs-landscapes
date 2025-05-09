import torch.optim as optim
from torch_geometric.data import DataLoader
import torch.nn as nn
import torch
import os
import yaml
import argparse
import optuna

import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_path))


from training import (
    setup_training,
    load_datasets,
    save_checkpoint,
    train_loop,
    eval_loop,
    set_seed,
)


from gnn import init_model

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run multiple seeds with Optuna-like setup"
)
parser.add_argument(
    "--seeds", type=int, nargs="+", required=True, help="List of seeds for training"
)

try:
    args = parser.parse_args()
    seeds = args.seeds
except SystemExit:
    print("Seed arguments invalid, using default seed parameters")
    seeds = [1, 2]

print("Using the following seeds: ", seeds)

# Load configurations
config_dir = Path(__file__).resolve().parent.parent / "config"
model_config_path = config_dir / "model_config.yaml"
training_config_path = config_dir / "training_config.yaml"

with open(model_config_path, "r") as file:
    model_config = yaml.safe_load(file)

with open(training_config_path, "r") as file:
    training_config = yaml.safe_load(file)

setup_training(training_config, [model_config_path, training_config_path])

# Extract study_name dynamically from training_dir
training_dir = Path(training_config["training_dir"])
study_name = training_dir.name

# Set image_size based on num_sections
model_config["image_size"] = training_config["num_sections"]

# Write the updated model_config to model_config.yaml
with open(model_config_path, "w") as file:
    yaml.safe_dump(model_config, file)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up Optuna study
study_db_path = training_dir / "seeds_study.db"
study = optuna.create_study(
    study_name=study_name,
    direction="minimize",
    storage=f"sqlite:///{study_db_path}",
    load_if_exists=True,
)


def objective(seed):
    training_config["manual_seed"] = seed
    # setup_training(training_config, [model_config_path, training_config_path])
    set_seed(training_config["manual_seed"])
    # Load datasets
    loaders = load_datasets(training_config, train=True, val=True, test=False)
    train_loader = loaders.get("train_loader")
    val_loader = loaders.get("val_loader")

    # Initialize model, loss, and optimizer
    model = init_model(model_config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])

    val_loss_best = float("inf")
    for epoch in range(training_config["num_epochs"]):
        train_loss, _ = train_loop(model, optimizer, criterion, train_loader, device)
        val_loss, _ = eval_loop(model, criterion, val_loader, device)

        if val_loss < val_loss_best:
            val_loss_best = val_loss
            save_checkpoint(
                training_dir / f"seed_{seed}_best_model.pt",
                model,
                optimizer,
                epoch,
                val_loss_best,
            )

    return val_loss_best


# Run the study for each seed
for seed in seeds:
    print(f"Running trial for seed: {seed}")
    study.enqueue_trial({"seed": seed})  # Ensure each seed is used exactly once
    study.optimize(lambda trial: objective(seed), n_trials=1)

# Print results
print("Best trial:", study.best_trial)
