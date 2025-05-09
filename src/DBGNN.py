import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import torch_scatter
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch.nn import ModuleList


class identity_scaling(torch.nn.Module):
    def __init__(self):
        super(identity_scaling, self).__init__()
        self.identity = nn.Identity()

    def forward(self, x, e):
        return self.identity(x), self.identity(e)


class scale_features_mean(torch.nn.Module):
    def __init__(self):
        super(scale_features_mean, self).__init__()

    def forward(self, x, e):
        s = 0.5 * (torch.mean(x) + torch.mean(e))
        return x / s, e / s


class scaling_features(torch.nn.Module):
    def __init__(self, scale_features=False):
        super(scaling_features, self).__init__()
        if scale_features == False:
            self.scaling = identity_scaling()
        else:
            self.scaling = scale_features_mean()

    def forward(self, x, e):
        return self.scaling(x, e)


class activation_function(torch.nn.Module):
    def __init__(self, activation_name):
        super(activation_function, self).__init__()
        if activation_name == "None":
            self.activation = nn.Identity()
        elif activation_name == "ReLU":
            self.activation = nn.ReLU()
        elif activation_name == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        else:
            print("unsupported activation_name: " + str(activation_name))

    def forward(self, x):
        return self.activation(x)


class LinDBLayer(MessagePassing):
    """
    Linear Dirac Bianconi Layer.
    Updates node and edge features based on the Linear Dirac Bianconi equation.
    """

    def __init__(
        self,
        n_channels,
        e_channels,
        activation_name_n,
        activation_name_e,
        dropout_n=0.0,
        dropout_e=0.0,
        Δ=0.01,
        scale_features=False,
    ):
        super(LinDBLayer, self).__init__(aggr="add", flow="target_to_source")

        # Linear transformations
        self.W_ne = nn.Parameter(Δ * torch.randn(e_channels, n_channels))
        self.W_en = nn.Parameter(Δ * torch.randn(n_channels, e_channels))

        # Beta parameters
        self.beta_e = nn.Parameter(Δ * torch.eye(e_channels))
        self.beta_n = nn.Parameter(-Δ * torch.eye(n_channels))

        # Dropout layers
        self.dropout_n = nn.Dropout(dropout_n)
        self.dropout_e = nn.Dropout(dropout_e)

        # Activation functions
        self.activation_n = activation_function(activation_name_n)
        self.activation_e = activation_function(activation_name_e)

        # Feature scaling
        self.scaling_features = scaling_features(scale_features)

    def forward(self, x, edge_index, e):

        # Edge Update
        e_new = self.edge_updater(edge_index, x=x, e=e, size=(x.size(0), x.size(0)))

        # Node Update
        x_new = self.propagate(edge_index, x=x, e=e_new, size=(x.size(0), x.size(0)))

        x_new = self.dropout_n(x_new)
        e_new = self.dropout_e(e_new)
        x_new = self.activation_n(x_new)
        e_new = self.activation_e(e_new)

        x_new, e_new = self.scaling_features(x_new, e_new)

        return x_new, e_new

    def edge_update(self, x_i, x_j, e):
        # x_i: source node features (x[edge_index[0]])
        # x_j: target node features (x[edge_index[1]])
        node_diff = x_j - x_i  # [num_edges, n_channels]
        e_new = node_diff @ self.W_en + e @ self.beta_e.T + e
        return e_new

    def message(self, e):
        # Compute messages based on edge features
        msg = e @ self.W_ne  ## [num_edges, n_channels]
        return msg

    def update(self, aggr_out, x):
        x_new = aggr_out + x @ self.beta_n.T + x
        return x_new


class DBGNN(MessagePassing):
    """
    Dirac Bianconi Graph Neural Network (DBGNN).

    This network uses the LinDBLayer for iterative node and edge feature updates.
    """

    def __init__(
        self,
        in_channels_n,
        hidden_channels_n,
        out_channels_n,
        in_channels_e,
        hidden_channels_e,
        num_layers,
        num_steps,
        activation_name_n,
        activation_name_e,
        dropout_n_in_lay=0.0,
        dropout_e_in_lay=0.0,
        dropout_n_after_lay=0.0,
        dropout_e_after_lay=0.0,
        dropout_final=0.0,
        Δ=0.01,
        skip_connection_n=True,
        skip_connection_e=True,
        dense_after_linDB=False,
        pool=None,
        bias_zero=False,
        scale_features=False,
    ):
        super(DBGNN, self).__init__(aggr="add")

        # Dense layers for initial transformation of node and edge input features
        self.dense_n_in = nn.Linear(in_channels_n, hidden_channels_n)
        self.dense_e_in = nn.Linear(in_channels_e, hidden_channels_e)

        self.dense_n_in.bias.data = torch.zeros_like(self.dense_n_in.bias)
        self.dense_e_in.bias.data = torch.zeros_like(self.dense_e_in.bias)

        # LinDBLayer for iterative updates
        self.linDB = ModuleList(
            [
                LinDBLayer(
                    hidden_channels_n,
                    hidden_channels_e,
                    activation_name_n,
                    activation_name_e,
                    dropout_n_in_lay,
                    dropout_e_in_lay,
                    Δ,
                    scale_features,
                )
                for _ in range(num_layers)
            ]
        )

        # Optional skip connection layers
        self.skip_connection_n = skip_connection_n
        self.skip_connection_e = skip_connection_e
        if skip_connection_n:
            self.skip_dense_n = nn.Linear(
                in_channels_n + hidden_channels_n, hidden_channels_n
            )
        if skip_connection_e:
            self.skip_dense_e = nn.Linear(
                in_channels_e + hidden_channels_e, hidden_channels_e
            )
        # We have the option to have different weights for each of the skips:
        # if skip_connection_n:
        #     self.skip_dense_n = [nn.Linear(in_channels_n + hidden_channels_n, hidden_channels_n for _ in range(num_layers)
        # if skip_connection_e:
        #     self.skip_dense_e = [nn.Linear(in_channels_e + hidden_channels_e, hidden_channels_e for _ in range(num_layers)

        # Number of layers for iterative updates
        self.num_layers = num_layers
        self.num_steps = num_steps

        # Optional pre-pooling dense layer
        self.dense_after_linDB = dense_after_linDB
        if dense_after_linDB != False:
            self.dense_after_linDB = nn.Linear(hidden_channels_n, dense_after_linDB)
            self.final_dense = nn.Linear(dense_after_linDB, out_channels_n)
            self.activation_n = activation_function(activation_name_n)
        else:
            # Final dense layer for output
            self.final_dense = nn.Linear(hidden_channels_n, out_channels_n)

        self.dropout_n_after_lay = nn.Dropout(dropout_n_after_lay)

        self.dropout_e_after_lay = nn.Dropout(dropout_e_after_lay)

        self.dropout_final = nn.Dropout(dropout_final)

        # set pool layer
        self.pool = pool

    def forward(self, x_n_in, edge_index, x_e_in, batch):
        """
        Forward pass for the DBGNN.

        Args:
        - x_n (Tensor): Input node features.
        - edge_index (Tensor): Edge indices.
        - x_e (Tensor): Input edge features.

        Returns:
        - x_n (Tensor): Output node features.
        """
        # Initial transformation of node and edge features
        x_n = self.dense_n_in(x_n_in)
        x_e = self.dense_e_in(x_e_in)

        # Iteratively update node and edge features using LinDBLayer
        for lay in range(self.num_layers):
            for _ in range(self.num_steps):
                x_n, x_e = self.linDB[lay](x_n, edge_index, x_e)
            x_n = self.dropout_n_after_lay(x_n)
            x_e = self.dropout_e_after_lay(x_e)

            # Optional skip connections
            if self.skip_connection_n:
                x_n = self.skip_dense_n(torch.cat([x_n_in, x_n], dim=1))
            if self.skip_connection_e:
                x_e = self.skip_dense_e(torch.cat([x_e_in, x_e], dim=1))

            # We have the option to have different weights for each of the skips:

            # if self.skip_connection_n:
            #     x_n = self.skip_dense_n[lay](torch.cat([x_n_in, x_n], dim=1))
            # if self.skip_connection_e:
            #     x_e = self.skip_dense_e[lay](torch.cat([x_e_in, x_e], dim=1))

        # Optional pre-pooling dense layer
        if self.dense_after_linDB != False:
            x_n = self.dense_after_linDB(x_n)
            x_n = self.activation_n(x_n)

        if self.dropout_final != False:
            x_n = self.dropout_final(x_n)

        if self.pool == "mean":
            x = global_mean_pool(x, batch)
        elif self.pool == "max":
            x = global_max_pool(x, batch)

        # Final transformation of node features and transpose the output
        # x_n = self.final_dense(x_n).T
        x_n = self.final_dense(x_n)

        return x_n
