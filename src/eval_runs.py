import torch
import numpy as np
import yaml
import optuna
import copy

from pathlib import Path
import torch
import pandas as pd
from sklearn.metrics import r2_score
from training import load_datasets, compute_snbs_pred_labels
from gnn import init_model
import torch.nn as nn

from training import compute_snbs_pred_labels


def evaluate_model(model, loader, criterion, device):
    """
    Evaluate the model on a given data loader.
    Returns predictions, targets, SNBS predictions, and SNBS labels.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_snbs_preds = []
    all_snbs_targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            predictions = model(data)
            loss = criterion(predictions, data.y)
            total_loss += loss.item()

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(data.y.cpu().numpy())

            snbs_preds, snbs_labels = compute_snbs_pred_labels(
                predictions, data.y, data.sample_heatmaps
            )
            all_snbs_preds.append(snbs_preds.cpu().numpy())
            all_snbs_targets.append(snbs_labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_snbs_preds = np.concatenate(all_snbs_preds)
    all_snbs_targets = np.concatenate(all_snbs_targets)

    return (
        total_loss / len(loader),
        all_preds,
        all_targets,
        all_snbs_preds,
        all_snbs_targets,
    )


def evaluate_loader(model, loader, criterion, device):
    """
    Helper function to evaluate a single data loader.
    Returns loss, predictions, targets, SNBS predictions, and SNBS labels.
    """
    loss, preds, targets, snbs_preds, snbs_targets = evaluate_model(
        model, loader, criterion, device
    )
    r2 = r2_score(targets.flatten(), preds.flatten())
    snbs_r2 = r2_score(snbs_targets, snbs_preds)
    return loss, r2, snbs_r2


def evaluate_seeds(training_dir, ood_eval=False, study_name=None):
    """
    Evaluate models for each seed and return a DataFrame with results.
    """

    model_config_path = training_dir / "model_config.yaml"
    training_config_path = training_dir / "training_config.yaml"

    with open(model_config_path, "r") as file:
        model_config = yaml.safe_load(file)

    with open(training_config_path, "r") as file:
        training_config = yaml.safe_load(file)

    # Load the study
    study_db_path = training_dir / "seeds_study.db"
    print("Loading study from :", study_db_path)
    if study_name == None:
        study_name = training_dir.name
    print("loading study with study_name: ", study_name)
    study = optuna.load_study(
        study_name=study_name, storage=f"sqlite:///{study_db_path}"
    )

    # Extract seeds from the study trials
    seeds = [
        trial.system_attrs["fixed_params"]["seed"]
        for trial in study.trials
        if "fixed_params" in trial.system_attrs
        and "seed" in trial.system_attrs["fixed_params"]
    ]

    # Load datasets
    loaders = load_datasets(training_config, train=False, val=True, test=True)
    test_loader = loaders.get("test_loader")
    val_loader = loaders.get("val_loader")

    if ood_eval:
        ood_config = copy.deepcopy(training_config)
        ood_config["dataset_name"] = "ds100"
        ood_config["num_sections"] = 20
        ood_config["test_slice_index"] = [8501, 10000]
        ood_loaders = load_datasets(ood_config, train=False, val=True, test=True)
        ood_val_loader = ood_loaders.get("val_loader")
        ood_test_loader = ood_loaders.get("test_loader")

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize loss function
    criterion = nn.MSELoss()

    results = []
    for seed in seeds:
        print(f"Evaluating model with seed: {seed}")
        model_path = Path(training_dir) / f"seed_{seed}_best_model.pt"
        if not model_path.exists():
            print(f"Model for seed {seed} not found. Skipping.")
            continue

        # Load model
        model = init_model(model_config).to(device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Evaluate on test and validation sets
        test_loss, image_test_r2, snbs_test_r2 = evaluate_loader(
            model, test_loader, criterion, device
        )
        val_loss, image_val_r2, snbs_val_r2 = evaluate_loader(
            model, val_loader, criterion, device
        )

        # Evaluate on OOD datasets if applicable
        ood_results = {}
        if ood_eval:
            ood_val_loss, ood_val_r2, ood_snbs_val_r2 = evaluate_loader(
                model, ood_val_loader, criterion, device
            )
            ood_test_loss, ood_test_r2, ood_snbs_test_r2 = evaluate_loader(
                model, ood_test_loader, criterion, device
            )
            ood_results = {
                "ood_val_loss": ood_val_loss,
                "ood_val_r2": ood_val_r2,
                "snbs_ood_val_r2": ood_snbs_val_r2,
                "ood_test_loss": ood_test_loss,
                "image_ood_test_r2": ood_test_r2,
                "snbs_ood_test_r2": ood_snbs_test_r2,
            }

        result = {
            "seed": seed,
            "test_loss": test_loss,
            "val_loss": val_loss,
            "image_test_r2": image_test_r2,
            "image_val_r2": image_val_r2,
            "snbs_test_r2": snbs_test_r2,
            "snbs_val_r2": snbs_val_r2,
        }

        # Add OOD results if available
        result.update(ood_results)

        results.append(result)
        print(
            f"Seed {seed}: Test Loss = {test_loss:.4f}, Image Test R2 = {image_test_r2:.4f}, SNBS Test R2 = {snbs_test_r2:.4f}"
        )
        print(
            f"Seed {seed}: Val Loss = {val_loss:.4f}, Image Val R2 = {image_val_r2:.4f}, SNBS Val R2 = {snbs_val_r2:.4f}"
        )

    # Create DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def populate_results_df(result_rows, statistics_df, column_mappings):
    """
    Populate a DataFrame with results based on mappings and input rows.
    """
    rows = []  # Use a list to collect rows
    for key, result_one_row in result_rows.items():
        row = {
            "model": result_one_row["model"],
            "tr20ev20_mean": statistics_df.loc[
                statistics_df["filename"] == result_one_row["tr20"],
                column_mappings["tr20ev20_mean"],
            ].values[0],
            "tr20ev20_std": statistics_df.loc[
                statistics_df["filename"] == result_one_row["tr20"],
                column_mappings["tr20ev20_std"],
            ].values[0],
            "tr100ev100_mean": statistics_df.loc[
                statistics_df["filename"] == result_one_row["tr100"],
                column_mappings["tr100ev100_mean"],
            ].values[0],
            "tr100ev100_std": statistics_df.loc[
                statistics_df["filename"] == result_one_row["tr100"],
                column_mappings["tr100ev100_std"],
            ].values[0],
            "tr20ev100_mean": statistics_df.loc[
                statistics_df["filename"] == result_one_row["tr20"],
                column_mappings["tr20ev100_mean"],
            ].values[0],
            "tr20ev100_std": statistics_df.loc[
                statistics_df["filename"] == result_one_row["tr20"],
                column_mappings["tr20ev100_std"],
            ].values[0],
        }
        rows.append(row)  # Add the row to the list
    return pd.DataFrame(rows)


def generate_latex_rows(results_df):
    """
    Generate LaTeX rows for a given results DataFrame.
    """
    latex_rows = []
    for _, row in results_df.iterrows():
        model = row["model"]
        tr20ev20 = (
            f"{row['tr20ev20_mean']:.2f} \\tiny{{$\\pm {row['tr20ev20_std']:.2f}$}}"
        )
        tr100ev100 = (
            f"{row['tr100ev100_mean']:.2f} \\tiny{{$\\pm {row['tr100ev100_std']:.2f}$}}"
        )
        tr20ev100 = (
            f"{row['tr20ev100_mean']:.2f} \\tiny{{$\\pm {row['tr20ev100_std']:.2f}$}}"
        )
        latex_row = f"{model} & {tr20ev20} & {tr100ev100} & {tr20ev100} \\\\"
        latex_rows.append(latex_row)
    return latex_rows


def create_statistics_df(base_dir):
    """
    Create a DataFrame containing statistics from CSV files in the given base directory.
    """
    import pandas as pd

    # Dictionary to store DataFrames
    csv_data = {}

    # Loop over all directories and find .csv files
    for csv_file in base_dir.rglob("*.csv"):  # Recursively find all .csv files
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        df.sort_values(by="val_loss", ascending=False, inplace=True)
        # Use the filename (without extension) as the key
        csv_data[csv_file.stem] = df

    # Initialize a list to store statistics for each file
    statistics = []

    # Loop over all keys in csv_data and calculate the requested statistics
    for filename, results_df in csv_data.items():
        try:
            stats = {
                "filename": filename,
                "image_test_r2_mean": 100 * results_df["image_test_r2"][0:3].mean(),
                "image_test_r2_std": 100 * results_df["image_test_r2"][0:3].std(),
                "snbs_test_r2_mean": 100 * results_df["snbs_test_r2"][0:3].mean(),
                "snbs_test_r2_std": 100 * results_df["snbs_test_r2"][0:3].std(),
                "image_ood_test_r2_mean": 100
                * results_df["image_ood_test_r2"][0:3].mean(),
                "image_ood_test_r2_std": 100
                * results_df["image_ood_test_r2"][0:3].std(),
                "snbs_ood_test_r2_mean": 100
                * results_df["snbs_ood_test_r2"][0:3].mean(),
                "snbs_ood_test_r2_std": 100 * results_df["snbs_ood_test_r2"][0:3].std(),
            }
            statistics.append(stats)
        except KeyError as e:
            print(f"Missing column in {filename}: {e}")

    # Convert the list of statistics into a DataFrame
    return pd.DataFrame(statistics)


def get_column_mappings():
    """
    Return column mappings for image and SNBS results.
    """
    image_column_mappings = {
        "tr20ev20_mean": "image_test_r2_mean",
        "tr20ev20_std": "image_test_r2_std",
        "tr100ev100_mean": "image_test_r2_mean",
        "tr100ev100_std": "image_test_r2_std",
        "tr20ev100_mean": "image_ood_test_r2_mean",
        "tr20ev100_std": "image_ood_test_r2_std",
    }

    snbs_column_mappings = {
        "tr20ev20_mean": "snbs_test_r2_mean",
        "tr20ev20_std": "snbs_test_r2_std",
        "tr100ev100_mean": "snbs_test_r2_mean",
        "tr100ev100_std": "snbs_test_r2_std",
        "tr20ev100_mean": "snbs_ood_test_r2_mean",
        "tr20ev100_std": "snbs_ood_test_r2_std",
    }

    return image_column_mappings, snbs_column_mappings


def load_model_with_best_seed(training_dir, seed=None):
    if seed == None:
        seed = get_best_seed(training_dir)
        print("For training_dir: ", training_dir, " the best seed is: ", seed)
    model_config_path = training_dir / "model_config.yaml"
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {model_config_path}")

    with open(model_config_path, "r") as file:
        model_config = yaml.safe_load(file)

    # Define the model
    model = init_model(model_config)

    # Load the model checkpoint
    checkpoint_path = training_dir / f"seed_{seed}_best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


# %%


def get_best_seed(training_dir):
    """
    Looks for the evaluation_results_{training_dir.name}.csv file in the training_dir
    and returns the top (first index) row as a dictionary.

    Args:
        training_dir (Path): Path to the training directory.

    Returns:
        dict: The top row of the CSV file as a dictionary, or None if the file doesn't exist.
    """
    csv_file = training_dir / f"evaluation_results_{training_dir.name}.csv"

    if not csv_file.exists():
        print(f"File {csv_file} does not exist.")
        return None

    # Read the CSV file
    results_df = pd.read_csv(csv_file)

    if results_df.empty:
        print("The CSV file is empty.")
        return None

    # Return the top row as a dictionary
    return results_df["seed"][0]
