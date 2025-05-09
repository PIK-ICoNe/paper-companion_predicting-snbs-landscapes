import sys
from pathlib import Path
import yaml
import argparse

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_path))

from hyperparameter_study import run_optuna_study
from training import setup_training


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

config_dir = Path(__file__).resolve().parent.parent / "config"
training_config_path = config_dir / "training_config.yaml"

if not training_config_path.exists():
    raise FileNotFoundError(
        "Configuration files not found. Ensure 'training_config.yaml' is present in the 'config' directory."
    )

with open(training_config_path, "r") as file:
    training_config = yaml.safe_load(file)

training_dir = Path(training_config["training_dir"])

# Setup training directory and copy config files
setup_training(
    training_config,
    [
        config_dir / "model_config.yaml",
        config_dir / "training_config.yaml",
        config_dir / "optuna_config.yaml",
    ],
)

# Save a file named after the job_id
output_file = training_dir / f"job_id_{job_id}.txt"
with open(output_file, "w") as file:
    file.write(f"Job ID: {job_id}\n")

run_optuna_study(config_dir, training_dir)

print("Hyperparameter study finished")
