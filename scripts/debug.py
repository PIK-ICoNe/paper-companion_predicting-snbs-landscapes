# %%
from pathlib import Path

import torch
import yaml

import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_path))

from eval_runs import evaluate_seeds
from gnn import init_model

from training import (
    setup_training,
    load_datasets,
    save_checkpoint,
    train_loop,
    eval_loop,
)

# %%
seed = 1
training_dir = "/home/nauck/joined_work/landscape_generation/ml_training/run_bo_opt13"

model_path = Path(training_dir) / f"seed_{seed}_best_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_config_path = Path(training_dir) / "model_config.yaml"
training_config_path = Path(training_dir) / "training_config.yaml"

with open(model_config_path, "r") as file:
    model_config = yaml.safe_load(file)

with open(training_config_path, "r") as file:
    training_config = yaml.safe_load(file)

# Load model
model = init_model(model_config).to(device)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint["model_state_dict"])

# %%
loaders = load_datasets(training_config, train=True, val=True, test=True)

# %%
