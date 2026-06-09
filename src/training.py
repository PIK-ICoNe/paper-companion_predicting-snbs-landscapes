import torch
from torchmetrics.regression import R2Score
import os
import shutil
import subprocess
from torch_geometric.data import DataLoader
import random
import numpy as np
from pathlib import Path

from src.oscillator_landscape import oscillatorLandscapceDataset


def _unwrap_model_output(output):
    """
    Normalize model outputs:
    - standard decoders return Tensor
    - VAE decoder returns (images, mu, logvar)
    """
    if isinstance(output, tuple):
        if len(output) == 3:
            images, mu, logvar = output
        else:
            # Fallback: assume first element is images
            images, mu, logvar = output[0], None, None
    else:
        images, mu, logvar = output, None, None
    return images, mu, logvar


def _kl_divergence(mu, logvar):
    # KL for diagonal Gaussian
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def train_loop(model, optimizer, criterion, data_loader, device, vae_beta=0.0):
    r2_score = R2Score().to(device)
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        images, mu, logvar = _unwrap_model_output(output)
        recon_loss = criterion(images, data.y)
        kl_loss = (
            _kl_divergence(mu, logvar) / images.shape[0] if mu is not None else 0.0
        )
        loss = recon_loss + vae_beta * kl_loss
        snbs_predicted, snbs_labels = compute_snbs_pred_labels(
            images, data.y, data.sample_heatmaps
        )
        r2_score.update(snbs_predicted, snbs_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss, r2_score.compute()


def eval_loop(model, criterion, data_loader, device, vae_beta=0.0):
    r2_score = R2Score().to(device)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            images, mu, logvar = _unwrap_model_output(output)
            recon_loss = criterion(images, data.y)
            kl_loss = (
                _kl_divergence(mu, logvar) / images.shape[0] if mu is not None else 0.0
            )
            loss = recon_loss + vae_beta * kl_loss
            snbs_predicted, snbs_labels = compute_snbs_pred_labels(
                images, data.y, data.sample_heatmaps
            )
            r2_score.update(snbs_predicted, snbs_labels)
            total_loss += loss.item()
    return total_loss, r2_score.compute()


def compute_snbs_pred_labels(predictions, labels, samples):
    # Compute snbs for each heatmap individually
    snbs_predicted = []
    snbs_labels = []
    # SNBS_from_heatmap = 1 - sum(basin_heatmap .* samples_heatmap)/sum(samples_heatmap)
    for i in range(predictions.shape[0]):
        snbs_pred = 1 - torch.sum(predictions[i] * samples[i]) / torch.sum(samples[i])
        snbs_label = 1 - torch.sum(labels[i] * samples[i]) / torch.sum(samples[i])
        snbs_predicted.append(snbs_pred)
        snbs_labels.append(snbs_label)

    snbs_predicted = torch.stack(snbs_predicted)
    snbs_labels = torch.stack(snbs_labels)
    return snbs_predicted, snbs_labels


def setup_training(training_config, config_files):
    training_dir = Path(training_config["training_dir"])
    # Check if training_dir exists, if not, create it
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)

    # Get the current git commit hash
    git_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )

    # Write the git commit hash to a temporary file
    git_commit_path = os.path.join(training_dir, "git_commit.txt")
    with open(git_commit_path, "w") as f:
        f.write(git_commit)

    set_seed(training_config["manual_seed"])


def load_datasets(training_config, train=True, val=True, test=False):
    dataset_root_path = training_config["dataset_root_path"]
    loaders = {}

    if train:
        train_set = oscillatorLandscapceDataset(
            dataset_root_path,
            training_config["dataset_name"],
            training_config["num_sections"],
            split="train",
            slice_index=slice(*training_config["train_slice_index"]),
            normalize_targets=False,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            force_reload=False,
        )
        loaders["train_loader"] = DataLoader(
            train_set, batch_size=training_config["batch_size"], shuffle=False
        )

    if val:
        val_set = oscillatorLandscapceDataset(
            dataset_root_path,
            training_config["dataset_name"],
            training_config["num_sections"],
            split="val",
            slice_index=slice(*training_config["val_slice_index"]),
            normalize_targets=False,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            force_reload=False,
        )
        loaders["val_loader"] = DataLoader(
            val_set, batch_size=training_config["batch_size"], shuffle=False
        )

    if test:
        test_set = oscillatorLandscapceDataset(
            dataset_root_path,
            training_config["dataset_name"],
            training_config["num_sections"],
            split="test",
            slice_index=slice(*training_config["test_slice_index"]),
            normalize_targets=False,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            force_reload=False,
        )
        loaders["test_loader"] = DataLoader(
            test_set, batch_size=training_config["batch_size"], shuffle=False
        )

    return loaders


def save_checkpoint(path, model, optimizer, epoch, val_loss_best):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss_best": val_loss_best,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, allow_partial: bool = False):
    """
    Load checkpoint with optional tolerant loading (for changed decoder heads).
    If allow_partial is True, only parameters with matching keys and shapes are loaded.
    """
    checkpoint = torch.load(path)
    state_dict = checkpoint["model_state_dict"]
    if allow_partial:
        model_state = model.state_dict()
        compatible_state = {
            k: v
            for k, v in state_dict.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        skipped = set(state_dict.keys()) - set(compatible_state.keys())
        if skipped:
            print(f"load_checkpoint: skipped incompatible keys: {sorted(skipped)}")
        model.load_state_dict(compatible_state, strict=False)
    else:
        model.load_state_dict(state_dict)
    if optimizer is not None:
        if allow_partial:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError as e:
                # Optimizer param groups changed (e.g., decoder swap); proceed with fresh optimizer.
                print(
                    f"load_checkpoint: optimizer state mismatch, skipping load. Error: {e}"
                )
        else:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["val_loss_best"]


def set_seed(seed):
    """
    Set the seed for reproducibility across torch, numpy, and random.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
