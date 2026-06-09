import os
import yaml
import subprocess
from pathlib import Path
import shutil

# Paths
base_dir = Path(__file__).resolve().parent.parent
config_dir = base_dir / "config"
ablation_config_path = config_dir / "ablation_config.yaml"
model_config_path = config_dir / "model_config.yaml"
training_config_path = config_dir / "training_config.yaml"
run_multiple_seeds_path = base_dir / "scripts" / "run_multiple_seeds.py"
ablation_configs_root = config_dir / "configs_ablation"
ablation_configs_root.mkdir(exist_ok=True)

# Load ablation config
with open(ablation_config_path, "r") as f:
    ablation_cfg = yaml.safe_load(f)

ablation_id = str(ablation_cfg["ablation_id"])
ablation_cfg = {k: v for k, v in ablation_cfg.items() if k != "ablation_id"}
param_name, param_values = next(iter(ablation_cfg.items()))

ablation_id_dir = ablation_configs_root / f"ablation_{ablation_id}"
ablation_id_dir.mkdir(parents=True, exist_ok=True)

# Copy config files to ablation_id_dir for reference
shutil.copy(training_config_path, ablation_id_dir / "training_config.yaml")
shutil.copy(model_config_path, ablation_id_dir / "model_config.yaml")
shutil.copy(ablation_config_path, ablation_id_dir / "ablation_config.yaml")

for idx, value in enumerate(param_values, 1):
    ablation_subdir = ablation_id_dir / f"ablation_{idx}"
    ablation_subdir.mkdir(parents=True, exist_ok=True)

    # Prepare unique training dir for this ablation value
    with open(training_config_path, "r") as f:
        training_cfg = yaml.safe_load(f)
    training_dir = Path(training_cfg["training_dir"])
    ablation_dir = (
        training_dir.parent / f"{training_dir.stem}_{ablation_id}" / f"idx_{idx}"
    )
    training_cfg["training_dir"] = str(ablation_dir)
    # Save modified training config
    ablation_training_config_path = ablation_subdir / "training_config.yaml"
    with open(ablation_training_config_path, "w") as f:
        yaml.safe_dump(training_cfg, f)

    # Update model config with ablation parameter
    with open(model_config_path, "r") as f:
        model_cfg = yaml.safe_load(f)
    model_cfg[param_name] = value
    ablation_model_config_path = ablation_subdir / "model_config.yaml"
    with open(ablation_model_config_path, "w") as f:
        yaml.safe_dump(model_cfg, f)

    # copy ablation_config.yaml for reference
    shutil.copy(ablation_config_path, ablation_subdir / "ablation_config.yaml")

    # Run the multi-seed script with the ablation configs
    subprocess.run(
        [
            "python",
            str(run_multiple_seeds_path),
            "--seeds",
            "1",  # or any list of seeds
        ],
        env={
            **dict(os.environ),
            "MODEL_CONFIG": str(ablation_model_config_path),
            "TRAINING_CONFIG": str(ablation_training_config_path),
        },
    )
