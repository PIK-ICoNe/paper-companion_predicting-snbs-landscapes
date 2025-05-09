# gnn_landscape_generation

This repository aims to predict landscapes as thresholded heatmaps using Graph Neural Networks (GNNs). It includes tools for training, evaluating, and analyzing models across multiple seeds, as well as hyperparameter optimization using Optuna.

## Features

- **Training**: Train GNN models on landscape datasets with configurable parameters.
- **Evaluation**: Evaluate models across multiple seeds and generate detailed performance metrics.
- **Hyperparameter Optimization**: Use Optuna to optimize model hyperparameters and save the best-performing models.
- **OOD Evaluation**: Evaluate models on out-of-distribution (OOD) datasets to assess generalization.
- **Visualization**: Generate and display heatmaps for grid nodes to visualize predictions.
- **Reproducibility**: Set seeds for reproducible experiments.

## Repository Structure

- **`src/`**: Contains core functionality, including training, evaluation, and model initialization.
  - `eval_runs.py`: Functions for evaluating models and generating performance statistics.
  - `training.py`: Functions for training and saving models, as well as dataset loading.
  - `hyperparameter_study.py`: Tools for managing hyperparameter optimization and saving the best models.
  - `gnn.py`: Initialization and configuration of GNN models.
- **`scripts/`**: Contains scripts for running experiments and analyzing results.
  - `run_multiple_seeds.py`: Train models across multiple seeds.
  - `eval_multiple_seeds.py`: Evaluate models across multiple seeds and save results.
  - `analyze_model.py`: Analyze and visualize model predictions.
- **`config/`**: Configuration files for training and model parameters.
- **`ml_training/`**: Directory for storing training outputs, including models, logs, and evaluation results.
- **`env/`**: Contains the `environment.yaml` file for setting up the Conda environment.

## Setting Up the Environment

To set up the environment using the provided `environment.yaml` file:

1. Install [Conda](https://docs.conda.io/en/latest/miniconda.html) if not already installed.
2. Create the environment:
   ```bash
   conda env create -f env/environment.yaml


## Reproducibility

To reproduce the results and generate the figures, the necessary resources are available on Zenodo: [https://doi.org/10.5281/zenodo.15373799](https://doi.org/10.5281/zenodo.15373799).

### Steps to Reproduce:
1. **Generate Performance Tables**:  
   Run the `show_statistical_results.py` script to generate LaTeX rows for the performance tables:
   ```bash
   python scripts/show_statistical_results.py
   ```

2. **Generate Heatmaps**:  
   Run the `show_heatmap_plots_paper.py` script to create heatmaps comparing predictions with the true results:
   ```bash
   python scripts/show_heatmap_plots_paper.py
   ```

3. **Basin Heatmaps**:  
   The Zenodo repository also includes the code and resources required to generate basin heatmaps based on dynamical simulations.