import optuna
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
import torch.nn as nn
import yaml
import os

import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

from src.gnn import init_model
from src.training import train_loop, eval_loop, load_datasets, save_checkpoint


def load_configurations(config_dir):
    model_config_path = config_dir / "model_config.yaml"
    training_config_path = config_dir / "training_config.yaml"
    optuna_config_path = config_dir / "optuna_config.yaml"

    if (
        not model_config_path.exists()
        or not training_config_path.exists()
        or not optuna_config_path.exists()
    ):
        raise FileNotFoundError(
            "Configuration files not found. Ensure 'model_config.yaml', 'training_config.yaml', and 'optuna_config.yaml' are present in the 'config' directory."
        )

    with open(model_config_path, "r") as file:
        model_config = yaml.safe_load(file)

    with open(training_config_path, "r") as file:
        training_config = yaml.safe_load(file)

    with open(optuna_config_path, "r") as file:
        optuna_config = yaml.safe_load(file)

    # Convert values to the correct types
    for key, value in optuna_config.items():
        if key != "study_name" and isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_key in ["low", "high"]:
                    optuna_config[key][sub_key] = float(sub_value)
        elif key != "study_name":
            optuna_config[key] = int(value)

    return model_config, training_config, optuna_config


def save_best_model(
    trial,
    model,
    optimizer,
    epoch,
    val_loss_best,
    training_dir,
    model_config,
    hyperparameters,
):
    trial_dir = os.path.join(training_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
    save_checkpoint(
        os.path.join(trial_dir, "best_model.pt"), model, optimizer, epoch, val_loss_best
    )

    # Save the model configuration
    trial_model_config = model_config.copy()
    trial_model_config.update(hyperparameters)
    with open(os.path.join(trial_dir, "model_config.yaml"), "w") as file:
        yaml.safe_dump(trial_model_config, file)


def objective(
    trial,
    model_config,
    training_config,
    optuna_config,
    device,
    train_loader,
    val_loader,
    training_dir,
):
    # Suggest hyperparameters dynamically based on optuna_config
    hyperparameters = {}
    if "learning_rate" in optuna_config:
        hyperparameters["learning_rate"] = trial.suggest_float(
            "learning_rate",
            optuna_config["learning_rate"]["low"],
            optuna_config["learning_rate"]["high"],
            log=True,
        )
    if "gnn_hidden_channels" in optuna_config:
        hyperparameters["gnn_hidden_channels"] = trial.suggest_int(
            "gnn_hidden_channels",
            optuna_config["gnn_hidden_channels"]["low"],
            optuna_config["gnn_hidden_channels"]["high"],
        )
    if "gnn_out_channels" in optuna_config:
        hyperparameters["gnn_out_channels"] = trial.suggest_int(
            "gnn_out_channels",
            optuna_config["gnn_out_channels"]["low"],
            optuna_config["gnn_out_channels"]["high"],
        )
    if "gnn_dropout_rate" in optuna_config:
        hyperparameters["gnn_dropout_rate"] = trial.suggest_float(
            "gnn_dropout_rate",
            optuna_config["gnn_dropout_rate"]["low"],
            optuna_config["gnn_dropout_rate"]["high"],
        )
    if "gnn_num_layers" in optuna_config:
        hyperparameters["gnn_num_layers"] = trial.suggest_int(
            "gnn_num_layers",
            optuna_config["gnn_num_layers"]["low"],
            optuna_config["gnn_num_layers"]["high"],
        )
    if "mlp_hidden_channels" in optuna_config:
        hyperparameters["mlp_hidden_channels"] = trial.suggest_int(
            "mlp_hidden_channels",
            optuna_config["mlp_hidden_channels"]["low"],
            optuna_config["mlp_hidden_channels"]["high"],
        )
    if "mlp_dropout" in optuna_config:
        hyperparameters["mlp_dropout"] = trial.suggest_float(
            "mlp_dropout",
            optuna_config["mlp_dropout"]["low"],
            optuna_config["mlp_dropout"]["high"],
        )
    if "layer_type" in optuna_config:
        hyperparameters["layer_type"] = trial.suggest_categorical(
            "layer_type", optuna_config["layer_type"]["choices"]
        )
    if "dbgnn::num_steps" in optuna_config:
        hyperparameters["dbgnn::num_steps"] = trial.suggest_int(
            "dbgnn::num_steps",
            optuna_config["dbgnn::num_steps"]["low"],
            optuna_config["dbgnn::num_steps"]["high"],
        )

    # Update model_config with hyperparameters
    model_config.update(hyperparameters)

    # Define the model, loss function, and optimizer
    model = init_model(model_config).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparameters.get("learning_rate", training_config["learning_rate"]),
    )

    val_loss_best = 1e10

    # Initialize lists to store sequences
    train_losses = []
    val_losses = []
    train_R2_scores = []
    val_R2_scores = []

    # Training loop
    num_epochs = training_config["num_epochs"]
    for epoch in range(num_epochs):
        train_loss, train_R2 = train_loop(
            model,
            optimizer,
            criterion,
            train_loader,
            device,
            vae_beta=training_config["vae_beta"],
        )
        val_loss, val_R2 = eval_loop(
            model,
            criterion,
            val_loader,
            device,
            vae_beta=training_config["vae_beta"],
        )

        # Append values to sequences
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_R2_scores.append(train_R2.item())
        val_R2_scores.append(val_R2.item())

        # Report validation loss for live tracking and pruning
        trial.report(val_loss, epoch)

        # Print progress
        if (epoch + 1) % training_config["eval_interval"] == 0:
            train_loss_sci = f"{train_loss/len(train_loader):.4e}"
            val_loss_sci = f"{val_loss/len(val_loader):.4e}"
            print(
                f"Trial {trial.number}/{optuna_config['n_trials']}, Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_sci}, Train R2: {train_R2:.3f}"
            )
            print(f"Valid Loss: {val_loss_sci}, Val R2: {val_R2:.3f}")

        # Save the best model per trial
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            save_best_model(
                trial,
                model,
                optimizer,
                epoch,
                val_loss_best,
                training_dir,
                model_config,
                hyperparameters,
            )

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Log sequences to user attributes for persistent storage
    trial.set_user_attr("train_losses", train_losses)
    trial.set_user_attr("val_losses", val_losses)
    trial.set_user_attr("train_R2_scores", train_R2_scores)
    trial.set_user_attr("val_R2_scores", val_R2_scores)

    return val_loss


def run_optuna_study(config_dir, training_dir):
    model_config, training_config, optuna_config = load_configurations(config_dir)
    training_config.setdefault("vae_beta", 0.0)

    # Set image_size based on num_sections
    model_config["image_size"] = training_config["num_sections"]

    # Write the updated model_config to model_config.yaml
    model_config_path = config_dir / "model_config.yaml"
    with open(model_config_path, "w") as file:
        yaml.safe_dump(model_config, file)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the datasets
    loaders = load_datasets(training_config, train=True, val=True, test=False)
    train_loader = loaders.get("train_loader")
    val_loader = loaders.get("val_loader")

    # Use an SQLite database to store the study information
    study_db_path = os.path.join(training_dir, "optuna_study.db")
    study_name = optuna_config["study_name"]
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=f"sqlite:///{study_db_path}",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial,
            model_config,
            training_config,
            optuna_config,
            device,
            train_loader,
            val_loader,
            training_dir,
        ),
        n_trials=optuna_config["n_trials"],
    )

    # Save the best hyperparameters
    best_params = study.best_params
    with open(os.path.join(training_dir, "best_hyperparameters.yaml"), "w") as f:
        yaml.dump(best_params, f)

    print("Best hyperparameters:", best_params)


def load_best_model(training_dir):
    study = load_study(training_dir)

    # Get the best trial
    best_trial = study.best_trial
    best_trial_dir = training_dir / f"trial_{best_trial.number}"

    # Load the model configuration
    model_config_path = best_trial_dir / "model_config.yaml"
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {model_config_path}")

    with open(model_config_path, "r") as file:
        model_config = yaml.safe_load(file)

    # Define the model
    model = init_model(model_config)

    # Load the model checkpoint
    checkpoint_path = best_trial_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def load_study(training_dir, study_name=None):
    # Load the study
    study_db_path = training_dir / "optuna_study.db"

    # Check if the database file exists
    if not study_db_path.exists():
        raise FileNotFoundError(f"Database file not found: {study_db_path}")

    if study_name == None:
        study_summaries = optuna.get_all_study_summaries(
            storage=f"sqlite:///{study_db_path}"
        )
        if len(study_summaries) != 1:
            raise ("Can not obtain study name, len(study_sammaries) != 1")
        study_name = study_summaries[0].study_name
    study = optuna.load_study(
        study_name=study_name, storage=f"sqlite:///{study_db_path}"
    )
    return study


def print_plot_study(training_dir, study_name=None):
    study = load_study(training_dir, study_name)

    # %%
    # Print the best trial
    best_trial = study.best_trial
    print(f"Best trial number: {best_trial.number}")
    print(f"Best trial value: {best_trial.value}")
    print("Best trial parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Save the best hyperparameters to a YAML file
    best_params_path = training_dir / "best_hyperparameters.yaml"
    with open(best_params_path, "w") as f:
        yaml.dump(best_trial.params, f)

    # Print all trials with their objective values
    print("\nAll trials:")
    for trial in study.trials:
        print(f"Trial {trial.number}: Value = {trial.value}, Params = {trial.params}")

    # Plot optimization history
    optuna.visualization.plot_optimization_history(study).show()

    # Plot parameter importance
    optuna.visualization.plot_param_importances(study).show()

    # Plot parallel coordinate
    optuna.visualization.plot_parallel_coordinate(study).show()

    # Plot slice
    optuna.visualization.plot_slice(study).show()

    # Plot contour
    optuna.visualization.plot_contour(study).show()


def plot_loss_over_epochs(training_dir, trial_ids=None, metric="loss"):
    """
    Plot train and validation metrics over epochs for selected trials.

    Args:
        training_dir (str): Path to the training directory containing the study.
        trial_ids (list, optional): List of trial IDs to include in the plot. If None, include all trials.
        metric (str): Metric to plot. Options are "loss" or "R2".
    """
    study = load_study(training_dir)

    # Extract metric values over epochs for each trial
    epochs = []
    train_metrics = []
    val_metrics = []
    trial_numbers = []

    metric_key_map = {
        "loss": ("train_losses", "val_losses"),
        "R2": ("train_R2_scores", "val_R2_scores"),
    }

    if metric not in metric_key_map:
        raise ValueError(f"Invalid metric '{metric}'. Choose 'loss' or 'R2'.")

    train_key, val_key = metric_key_map[metric]

    # Sort trials by trial number
    sorted_trials = sorted(study.trials, key=lambda t: t.number)

    for trial in sorted_trials:
        if trial_ids is None or trial.number in trial_ids:
            train_values = trial.user_attrs.get(train_key, [])
            val_values = trial.user_attrs.get(val_key, [])
            if train_values and val_values:
                epochs.append(list(range(len(train_values))))
                train_metrics.append(train_values)
                val_metrics.append(val_values)
                trial_numbers.append(trial.number)

    # Use a consistent color palette for train and validation curves
    colors = cycle(sns.color_palette("tab10"))

    # Determine the number of subplots needed (4 trials per subplot)
    trials_per_subplot = 8
    num_subplots = (len(trial_numbers) + trials_per_subplot - 1) // trials_per_subplot

    # Create subplots
    fig, axes = plt.subplots(
        num_subplots, 1, figsize=(10, 6 * num_subplots), squeeze=False
    )
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        start_idx = i * trials_per_subplot
        end_idx = start_idx + trials_per_subplot
        for j in range(start_idx, min(end_idx, len(trial_numbers))):
            color = next(colors)  # Get the next color from the cycle
            ax.plot(
                epochs[j],
                train_metrics[j],
                linestyle="--",
                color=color,
                label=f"Train {metric.capitalize()} Trial {trial_numbers[j]}",
            )
            ax.plot(
                epochs[j],
                val_metrics[j],
                linestyle="-",
                color=color,
                label=f"Val {metric.capitalize()} Trial {trial_numbers[j]}",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)
        # Limit y-axis for R2 metric
        if metric == "R2":
            ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.show()
