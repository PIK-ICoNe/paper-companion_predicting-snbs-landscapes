import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

from gnn_models_from_grids_ml import gnn_model_generic
from DBGNN import DBGNN


class GNNImageGenerator(nn.Module):
    def __init__(
        self,
        gnn_in_channels,
        gnn_out_channels,
        gnn_hidden_channels,
        gnn_num_layers,
        gnn_dropout_rate,
        gnn_layer_type,
        mlp_hidden_channels,
        mlp_out_channels,
        mlp_dropout,
        image_size,
        dbgnn_num_steps=0,
    ):
        super(GNNImageGenerator, self).__init__()
        if gnn_layer_type == "DBGNN":
            self.gnn = DBGNN(
                gnn_in_channels,
                gnn_hidden_channels,
                gnn_out_channels,
                1,
                gnn_hidden_channels,
                gnn_num_layers,
                dbgnn_num_steps,
                "LeakyReLU",
                "LeakyReLU",
                gnn_dropout_rate,
                gnn_dropout_rate,
            )
        else:
            self.gnn = gnn_model_generic(
                gnn_in_channels,
                gnn_hidden_channels,
                gnn_out_channels,
                gnn_num_layers,
                layer_type=gnn_layer_type,
                dropout_rate=gnn_dropout_rate,
                input_edge_dim=1,
            )
        self.mlp = nn.Sequential(
            nn.Linear(
                gnn_hidden_channels, mlp_hidden_channels
            ),  # Ensure the input dimension matches the output of GNN
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels, mlp_out_channels),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_out_channels, image_size * image_size),
        )
        self.image_size = image_size

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        x = self.gnn(x, edge_index, edge_attr, batch)
        images = self.mlp(x)
        images = images.view(
            -1, self.image_size, self.image_size
        )  # Adjusted to match target dimensions
        return images


def init_model(model_config):
    if "dbgnn::num_steps" not in model_config:
        model_config["dbgnn::num_steps"] = 0
    # Define the model, loss function, and optimizer
    return GNNImageGenerator(
        gnn_in_channels=model_config["gnn_in_channels"],
        gnn_hidden_channels=model_config["gnn_hidden_channels"],
        gnn_out_channels=model_config["gnn_out_channels"],
        gnn_num_layers=model_config["gnn_num_layers"],
        gnn_dropout_rate=model_config["gnn_dropout_rate"],
        gnn_layer_type=model_config["gnn_layer_type"],
        mlp_hidden_channels=model_config["mlp_hidden_channels"],
        mlp_out_channels=model_config["mlp_out_channels"],
        mlp_dropout=model_config["mlp_dropout"],
        image_size=model_config["image_size"],
        dbgnn_num_steps=model_config["dbgnn::num_steps"],
    )
