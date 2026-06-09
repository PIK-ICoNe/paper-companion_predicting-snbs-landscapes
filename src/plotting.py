import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.shock_locator import get_shock_location
from src.training import _unwrap_model_output

# from utils import compute_snbs_from_heatmap


fontsize_ticks = 12
fontsize_labels = 14
fontsize_title = 18


def set_heatmap_properties(ax, input_heatmap, title):
    # Define the x and y axis labels
    x_labels = np.linspace(-np.pi, np.pi, input_heatmap.shape[1])
    y_labels = np.linspace(-15, 15, input_heatmap.shape[0])

    # Set axis labels with increased font size
    ax.set_xlabel(r"$\phi$", fontsize=fontsize_labels)
    ax.set_ylabel(r"$\dot{\phi}$", fontsize=fontsize_labels)
    if title != None:
        ax.set_title(title, fontsize=fontsize_title)

    # Round the x and y labels and show a reduced number of ticks
    num_ticks = 5  # Number of ticks to show
    x_ticks = np.linspace(-np.pi, np.pi, num_ticks)
    y_ticks = np.linspace(-15, 15, num_ticks)
    ax.set_xticks(np.linspace(0, len(x_labels) - 1, num_ticks))
    ax.set_xticklabels(np.round(x_ticks, 2), fontsize=fontsize_ticks, rotation=0)
    ax.set_yticks(np.linspace(0, len(y_labels) - 1, num_ticks))
    ax.set_yticklabels(np.round(y_ticks, 2), fontsize=fontsize_ticks)
    ax.invert_yaxis()  # Invert the y-axis


def show_heatmap(
    input_heatmap,
    samples,
    title=None,
    ax=None,
    vmin=None,
    vmax=None,
    highlight_indices=None,
    highlight_color="lime",
    marker_size=None,
    marker_edgewidth=2,
    flip_highlight_indices=False,
):
    if isinstance(input_heatmap, torch.Tensor):
        input_heatmap = input_heatmap.cpu().numpy()
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    if highlight_indices is not None and isinstance(highlight_indices, torch.Tensor):
        highlight_indices = highlight_indices.cpu().numpy()

    # Optionally flip highlight_indices
    if highlight_indices is not None and flip_highlight_indices:
        n_rows = input_heatmap.shape[0]
        highlight_indices = np.array(
            [[n_rows - 1 - y, x] for y, x in highlight_indices]
        )

    x_labels = np.linspace(-np.pi, np.pi, input_heatmap.shape[1])
    y_labels = np.linspace(-15, 15, input_heatmap.shape[0])

    if ax is None:
        fig, ax = plt.subplots()

    sns.heatmap(
        input_heatmap,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar_kws={"label": "ratio exceedings"},
        ax=ax,
        vmin=vmin,
        vmax=vmax,
    )
    snbs = 1 - np.sum(input_heatmap * samples) / np.sum(samples)
    if title != None:
        title = f"{title}_SNBS_{snbs:.4f}"
    set_heatmap_properties(ax, input_heatmap, title)

    # Optionally plot highlighted cells
    if highlight_indices is not None:
        n_rows = input_heatmap.shape[0]

        # Auto-compute marker size if not provided
        if marker_size is None:
            # Get axis bbox in display coordinates
            bbox = ax.get_window_extent()
            # Calculate points per data unit (average of x and y)
            points_per_cell = min(
                bbox.width / input_heatmap.shape[1],
                bbox.height / input_heatmap.shape[0],
            )
            # Set marker size to match cell size (reduce slightly for aesthetics)
            marker_size = points_per_cell * 0.9

        for idx in highlight_indices:
            y, x = idx
            y_plot = n_rows - 1 - y
            ax.plot(
                x + 0.5,
                y_plot + 0.5,
                marker="o",
                markerfacecolor="none",
                markeredgecolor=highlight_color,
                markersize=marker_size,
                markeredgewidth=marker_edgewidth,
            )

    if ax is None:
        plt.show()


def show_side_by_side_heatmaps(
    heatmap1, heatmap2, samples, title1=None, title2=None, axes=None
):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    vmin = 0.0
    vmax = 1.0

    # Create the first heatmap
    show_heatmap(heatmap1, samples, title1, ax=axes[0], vmin=vmin, vmax=vmax)

    # Create the second heatmap
    show_heatmap(heatmap2, samples, title2, ax=axes[1], vmin=vmin, vmax=vmax)

    if axes is None:
        plt.tight_layout()
        plt.show()


def show_heatmaps_for_grid_node(
    data_loader, model, grid_index, node_index, show_function=show_side_by_side_heatmaps
):
    """
    Plots side-by-side heatmaps for a specific grid and node.

    Parameters:
        data_loader: The data loader containing the dataset.
        model: The model used for predictions.
        grid_index: The index of the grid in the dataset.
        node_index: The index of the node in the grid.
        show_function: The function used to display the heatmaps (default: show_side_by_side_heatmaps).
    """
    true_heatmap = data_loader.dataset[grid_index].y[node_index].numpy()
    sample_heatmap = data_loader.dataset[grid_index].sample_heatmaps[node_index].numpy()
    output = model(data_loader.dataset[grid_index])
    predicted, _, _ = _unwrap_model_output(output)
    predicted_heatmap = predicted[node_index].detach().numpy()

    show_function(
        true_heatmap,
        predicted_heatmap,
        sample_heatmap,
        title1="true label",
        title2="prediction",
    )


def show_multiple_side_by_side_heatmaps(
    data_loader,
    model,
    grid_index,
    node_range,
    save_fig_name=None,
    highlight_indices=None,
):
    """
    Plots multiple side-by-side heatmaps in a single figure with 1 row for 2 nodes (4 heatmaps per row).

    Parameters:
        data_loader: The data loader containing the dataset.
        model: The model used for predictions.
        grid_index: The index of the grid in the dataset.
        node_range: A range of node indices to plot heatmaps for.
        save_fig_name: If provided, saves the figure to the specified file. If None, displays the plot.
        highlight_indices: If True, highlight shock positions using get_shock_location. If None/False, do not highlight.
    """
    num_nodes = len(node_range)
    num_rows = (num_nodes + 1) // 2  # Each row contains 2 nodes (4 heatmaps)
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, 4 * num_rows))

    # Ensure axes is always a 2D array for consistency
    if num_rows == 1:
        axes = axes.reshape(1, 4)

    for i, node_index in enumerate(node_range):
        row = i // 2  # Determine the row index
        col_offset = (i % 2) * 2  # Determine the column offset (0 or 2)

        true_heatmap = data_loader.dataset[grid_index].y[node_index].numpy()
        sample_heatmap = (
            data_loader.dataset[grid_index].sample_heatmaps[node_index].numpy()
        )
        output = model(data_loader.dataset[grid_index])
        predicted, _, _ = _unwrap_model_output(output)
        predicted_heatmap = predicted[node_index].detach().numpy()

        # Plot the true heatmap
        if highlight_indices:
            shock_pos, _ = get_shock_location(torch.from_numpy(true_heatmap))
            show_heatmap(
                true_heatmap,
                sample_heatmap,
                title=f"True Label",
                ax=axes[row, col_offset],
                vmin=0.0,
                vmax=1.0,
                highlight_indices=shock_pos,
            )
        else:
            show_heatmap(
                true_heatmap,
                sample_heatmap,
                title=f"True Label",
                ax=axes[row, col_offset],
                vmin=0.0,
                vmax=1.0,
            )

        # Plot the predicted heatmap
        if highlight_indices:
            shock_pos, _ = get_shock_location(torch.from_numpy(predicted_heatmap))
            show_heatmap(
                predicted_heatmap,
                sample_heatmap,
                title=f"Prediction",
                ax=axes[row, col_offset + 1],
                vmin=0.0,
                vmax=1.0,
                highlight_indices=shock_pos,
            )
        else:
            show_heatmap(
                predicted_heatmap,
                sample_heatmap,
                title=f"Prediction",
                ax=axes[row, col_offset + 1],
                vmin=0.0,
                vmax=1.0,
            )

    # Adjust layout
    plt.tight_layout()

    # Save or show the figure
    if save_fig_name:
        plt.savefig(save_fig_name, bbox_inches="tight")
        plt.close(fig)  # Close the figure to avoid displaying it
    else:
        plt.show()
