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
- **`pyproject.toml`**: Project dependencies and `uv` configuration.
- **`.venv/`**: Local virtual environment created and managed by `uv`.

## Setting Up the Environment

To set up the environment with `uv`:

1. Install [uv](https://docs.astral.sh/uv/) if not already installed.
2. Create or sync the environment:
   ```bash
   uv sync
   ```
3. Run scripts through `uv` so the pinned dependencies are used:
   ```bash
   uv run --offline python scripts/show_statistical_results.py
   ```


## Reproducibility

To reproduce the results and generate the figures, the necessary resources are available on Zenodo: [https://doi.org/10.5281/zenodo.15373799](https://doi.org/10.5281/zenodo.15373799).

### Main Scripts to Reproduce Paper Results

The following scripts generate all tables and figures from the paper:

1. **Generate Performance Tables (LaTeX)**:  
   Run `show_statistical_results.py` to generate all LaTeX rows for the performance tables:
   ```bash
   uv run --offline python scripts/show_statistical_results.py
   ```
   This script generates:
   - Image R² tables (core splits and real grids)
   - SSIM tables (core splits and real grids)
   - LPIPS tables (core splits and real grids)
   - SNBS R² tables (core splits and real grids)
   - Critical cells IoU tables (core splits and countries)
   - Critical cells Coverage@30 tables (core splits and countries)
   - Exported CSVs in `csv_results/` directory

2. **Generate Heatmap Comparisons**:  
   Run `show_heatmap_plots_paper.py` to create heatmaps comparing predictions with true results:
   ```bash
   uv run --offline python scripts/show_heatmap_plots_paper.py
   ```
   Saves heatmaps to `pics/` directory.

3. **Generate Contingency Grid Shock Analysis**:  
   Run `show_contingency_grid_shock_identification.py` to visualize critical cells on specific nodes:
   ```bash
   uv run --offline python scripts/show_contingency_grid_shock_identification.py
   ```

4. **Generate Performance Over Epoch Plots**:  
   Run `make_figure_performance_intermediate.py` to generate metric evolution plots:
   ```bash
   uv run --offline python scripts/make_figure_performance_intermediate.py
   ```

### Basin Heatmaps

The Zenodo repository also includes the code and resources required to generate basin heatmaps based on dynamical simulations.


### Basin Heatmap Label Convention

The dataset's `basin_heatmap_i` arrays store the **non-recovered fraction**: for each of the 20×20 cells of the (ϕ, ϕ̇) perturbation plane, the fraction of Monte-Carlo perturbation samples in that cell whose maximum final frequency deviation *exceeds* the threshold 0.1, i.e. perturbations that fail to recover. This is the **complement** of the recovered fraction L⁽ⁱ⁾ defined in Eq. (5) of the paper:

    L⁽ⁱ⁾ = 1 − basin_heatmap_i

All code in this repository consumes the stored convention directly: SNBS is computed as `1 - sum(basin_heatmap_i · samples_heatmap_i) / sum(samples_heatmap_i)`. When loading the files (e.g. with `h5py`), **do not apply a complement or a transposition**: as read by h5py, rows index the ϕ̇ (frequency-deviation) bin and columns index the ϕ (phase-angle) bin, both ascending from the sampled minimum. Bin edges are computed per grid from the empirical extrema of the sampled perturbations, which are drawn uniformly from the nominal box [−π, π] × [−15, 15]. `samples_heatmap_i` stores the per-cell sample counts used as weights above.



## Training and Evaluation

### Loss and Metrics

All comparator models in *Predicting Single-Node Basin Landscapes of Kuramoto Oscillators Using Graph Neural Networks* are trained with the **plain, unweighted mean squared error** of Eq. (8) (`nn.MSELoss()` over the stored `basin_heatmap_i` targets). The per-cell Monte-Carlo sample counts `samples_heatmap_i` are used **only** to aggregate the SNBS metric, `SNBS_i = 1 − Σ(H_i·q_i)/Σq_i`.

Appendix A.4 derives a weighted-MSE upper bound on the SNBS error; that bound is an analysis tool, not the training objective.

### Running Training

1. **Train with Optuna hyperparameter optimization**:
   ```bash
   uv run --offline python scripts/start_optuna.py
   ```

2. **Train multiple seeds for a fixed configuration**:
   ```bash
   uv run --offline python scripts/run_multiple_seeds.py
   ```

### Evaluating Models

1. **Basic evaluation across seeds**:
   ```bash
   uv run --offline python scripts/eval_multiple_seeds.py --training_dir ml_training/run_ns20_ds20_TAG
   ```

2. **Evaluation with extra grids (France, GB, Spain, Germany)**:
   ```bash
   uv run --offline python scripts/eval_multiple_seeds.py --training_dir ml_training/run_ns20_ds20_TAG --extra_grids osf_france osf_gb osf_spain elmod
   ```

3. **Evaluation with LPIPS metric** (slow, requires GPU):
   ```bash
   uv run --offline python scripts/eval_multiple_seeds.py --training_dir ml_training/run_ns20_ds20_TAG --extra_grids osf_france osf_gb osf_spain elmod --eval_lpips
   ```

4. **Critical cells evaluation**:
   ```bash
   uv run --offline python scripts/eval_critical_cells_multiple_seeds.py --training_dir ml_training/run_ns20_ds20_TAG
   ```

### Ablation Studies

To run ablation studies on hyperparameters:

```bash
uv run --offline python scripts/run_ablation_study.py
uv run --offline python scripts/eval_multiple_seeds.py --training_dir ml_training/run_ABLATION_NAME --extra_grids osf_france osf_gb osf_spain elmod
```

## License

The code in this repository is licensed under the MIT License — see the
[LICENSE](LICENSE) file.

The datasets used by this repository are licensed separately under CC BY 4.0:

- Hugging Face: [Stability Landscapes](https://huggingface.co/datasets/PIK-ICoNe-landscape/Stability_Landscapes)
- Zenodo: [10.5281/zenodo.15373799](https://doi.org/10.5281/zenodo.15373799)

The two licences are independent: MIT covers the code only, not the data.

This repository accompanies our ICML paper. If you use this code, please cite the paper.
