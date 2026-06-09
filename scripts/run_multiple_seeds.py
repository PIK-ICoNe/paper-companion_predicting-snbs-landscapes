import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import yaml
import argparse
import optuna
import math
from pathlib import Path

root_dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir_path))

from src.training import (
    setup_training,
    load_datasets,
    save_checkpoint,
    train_loop,
    eval_loop,
    set_seed,
)

from src.gnn import init_model

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run multiple seeds with Optuna-like setup"
)
# parser.add_argument(
#     "--seeds", type=int, nargs="+", required=True, help="List of seeds for training"
# )
parser.add_argument(
    "--seeds",
    type=int,
    nargs="+",
    default=[1, 2, 3, 4, 5],
    help="List of seeds for training",
)
parser.add_argument(
    "--config_dir", type=str, required=True, help="Path to the configuration directory"
)

try:
    args = parser.parse_args()
    seeds = args.seeds
except SystemExit:
    print("Seed arguments invalid, using default seed parameters")
    seeds = [1, 2]

print("Using the following seeds: ", seeds)

# Load configurations using the provided config_dir
config_dir = Path(args.config_dir)

# Allow config override via environment variables for ablation
model_config_path = Path(
    os.environ.get("MODEL_CONFIG", config_dir / "model_config.yaml")
)
training_config_path = Path(
    os.environ.get("TRAINING_CONFIG", config_dir / "training_config.yaml")
)

with open(model_config_path, "r") as file:
    model_config = yaml.safe_load(file)

with open(training_config_path, "r") as file:
    training_config = yaml.safe_load(file)

training_config.setdefault("vae_beta", 0.0)
training_config.setdefault("save_model_indermediate_steps", False)
training_config.setdefault(
    "model_indermediate_num_checkpoints",
    12,
)
training_config.setdefault(
    "model_indermediate_late_fractions",
    [0.25, 0.5, 0.75, 1.0],
)
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


def build_relative_intermediate_steps(
    num_epochs, num_checkpoints=12, late_fractions=None
):
    if num_epochs <= 0 or num_checkpoints <= 0:
        return set()

    if late_fractions is None:
        late_fractions = [0.25, 0.5, 0.75, 1.0]

    late_steps = {
        min(num_epochs, max(1, int(round(float(frac) * num_epochs))))
        for frac in late_fractions
        if isinstance(frac, (int, float)) and float(frac) > 0
    }
    late_steps.add(num_epochs)
    late_steps = sorted(late_steps)

    early_target_count = max(0, int(num_checkpoints) - len(late_steps))
    early_max = max(0, late_steps[0] - 1) if late_steps else max(0, num_epochs - 1)

    early_steps = []
    if early_target_count > 0 and early_max > 0:
        if early_target_count == 1:
            early_steps = [1]
        else:
            raw = [
                int(
                    round(
                        math.exp(
                            math.log(1)
                            + (math.log(early_max) * i) / (early_target_count - 1)
                        )
                    )
                )
                for i in range(early_target_count)
            ]
            early_steps = sorted({max(1, min(early_max, val)) for val in raw})

            candidate = 1
            while len(early_steps) < early_target_count and candidate <= early_max:
                if candidate not in early_steps:
                    early_steps.append(candidate)
                candidate += 1
            early_steps = sorted(early_steps)

    return {step for step in (early_steps + late_steps) if 1 <= int(step) <= num_epochs}


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

    save_intermediate = bool(
        training_config.get("save_model_indermediate_steps", False)
    )
    intermediate_steps = build_relative_intermediate_steps(
        num_epochs=training_config["num_epochs"],
        num_checkpoints=training_config.get("model_indermediate_num_checkpoints", 12),
        late_fractions=training_config.get(
            "model_indermediate_late_fractions", [0.25, 0.5, 0.75, 1.0]
        ),
    )
    if save_intermediate:
        print(
            f"Seed {seed}: saving intermediate checkpoints at epochs {sorted(intermediate_steps)}"
        )

    val_loss_best = float("inf")
    for epoch in range(training_config["num_epochs"]):
        epoch_idx = epoch + 1
        train_loss, _ = train_loop(
            model,
            optimizer,
            criterion,
            train_loader,
            device,
            vae_beta=training_config["vae_beta"],
        )
        val_loss, _ = eval_loop(
            model, criterion, val_loader, device, vae_beta=training_config["vae_beta"]
        )

        if val_loss < val_loss_best:
            val_loss_best = val_loss
            save_checkpoint(
                training_dir / f"seed_{seed}_best_model.pt",
                model,
                optimizer,
                epoch,
                val_loss_best,
            )

        if save_intermediate and epoch_idx in intermediate_steps:
            save_checkpoint(
                training_dir / f"seed_{seed}_epoch_{epoch_idx}.pt",
                model,
                optimizer,
                epoch,
                val_loss,
            )

    return val_loss_best


# Run the study for each seed
for seed in seeds:
    print(f"Running trial for seed: {seed}")
    study.enqueue_trial({"seed": seed})  # Ensure each seed is used exactly once
    study.optimize(lambda trial: objective(seed), n_trials=1)

# Print results
print("Best trial:", study.best_trial)
