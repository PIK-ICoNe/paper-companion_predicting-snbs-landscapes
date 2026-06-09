import torch
import torch.nn as nn

from src.gnn_models_from_grids_ml import gnn_model_generic
from src.DBGNN import DBGNN


class VisionTransformerDecoder(nn.Module):
    """
    Lightweight ViT-style decoder for 2D heatmaps.

    It projects each node embedding to a latent dimension, broadcasts it
    across a learnable 2D positional token grid, runs a few Transformer
    encoder blocks, and maps tokens back to pixels.
    """

    def __init__(
        self,
        in_dim: int,
        image_size: int,
        embed_dim: int = 256,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_sigmoid: bool = False,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_tokens = image_size * image_size

        self.input_proj = (
            nn.Linear(in_dim, embed_dim) if in_dim != embed_dim else nn.Identity()
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.output_activation = nn.Sigmoid() if use_sigmoid else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Args:
            x: node embeddings with shape (batch, in_dim)
        Returns:
            Heatmaps shaped (batch, H, W)
        """
        x = self.input_proj(x)
        batch_size = x.size(0)
        tokens = x.unsqueeze(1).expand(batch_size, self.num_tokens, -1)
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)
        tokens = self.head(tokens).squeeze(-1)
        tokens = self.output_activation(tokens)
        return tokens.view(batch_size, self.image_size, self.image_size)


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
        decoder_type="mlp",
        cnn_base_channels=64,
        vae_latent_dim=128,
        vit_hidden_dim=256,
        vit_num_heads=4,
        vit_depth=2,
        vit_mlp_ratio=4.0,
        vit_dropout=0.1,
        vit_use_sigmoid=False,
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
        decoder_type = decoder_type.lower()
        if decoder_type == "cnn":
            # Map node embedding to a small feature map and upsample with transposed convolutions.
            self.decoder = nn.Sequential(
                nn.Linear(gnn_hidden_channels, cnn_base_channels * 5 * 5),
                nn.ReLU(),
                nn.Unflatten(1, (cnn_base_channels, 5, 5)),
                nn.ConvTranspose2d(
                    cnn_base_channels, cnn_base_channels // 2, kernel_size=4, stride=2, padding=1
                ),  # 5 -> 10
                nn.ReLU(),
                nn.ConvTranspose2d(
                    cnn_base_channels // 2, cnn_base_channels // 4, kernel_size=4, stride=2, padding=1
                ),  # 10 -> 20
                nn.ReLU(),
                nn.Conv2d(cnn_base_channels // 4, 1, kernel_size=3, padding=1),
                nn.Sigmoid(),  # constrain to [0, 1] to match heatmap targets
            )
            self.decoder_type = "cnn"
        elif decoder_type in ("vae", "variational_autoencoder"):
            # VAE branch: learn mean/logvar, reparameterize, then decode with an MLP head.
            self.mu_head = nn.Linear(gnn_hidden_channels, vae_latent_dim)
            self.logvar_head = nn.Linear(gnn_hidden_channels, vae_latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(vae_latent_dim, mlp_hidden_channels),
                nn.ReLU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(mlp_hidden_channels, mlp_out_channels),
                nn.ReLU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(mlp_out_channels, image_size * image_size),
            )
            self.decoder_type = "vae"
            self.vae_latent_dim = vae_latent_dim
        elif decoder_type in ("vit", "vision_transformer", "transformer"):
            self.decoder = VisionTransformerDecoder(
                in_dim=gnn_hidden_channels,
                image_size=image_size,
                embed_dim=vit_hidden_dim,
                depth=vit_depth,
                num_heads=vit_num_heads,
                mlp_ratio=vit_mlp_ratio,
                dropout=vit_dropout,
                use_sigmoid=vit_use_sigmoid,
            )
            self.decoder_type = "vit"
        else:
            # Original MLP decoder
            self.decoder = nn.Sequential(
                nn.Linear(gnn_hidden_channels, mlp_hidden_channels),
                nn.ReLU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(mlp_hidden_channels, mlp_out_channels),
                nn.ReLU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(mlp_out_channels, image_size * image_size),
            )
            self.decoder_type = "mlp"
        self.image_size = image_size

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        x = self.gnn(x, edge_index, edge_attr, batch)
        if self.decoder_type == "vae":
            mu = self.mu_head(x)
            logvar = self.logvar_head(x)
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
            else:
                z = mu  # deterministic at eval time
            images = self.decoder(z).view(-1, self.image_size, self.image_size)
            return images, mu, logvar
        images = self.decoder(x)
        if self.decoder_type == "cnn":
            images = images.squeeze(1)
        elif self.decoder_type == "mlp":
            images = images.view(-1, self.image_size, self.image_size)
        return images


def init_model(model_config):
    model_config.setdefault("gnn_generic_regression_head", None)
    model_config.setdefault("dbgnn::num_steps", None)
    model_config.setdefault("decoder_type", "mlp")
    model_config.setdefault("cnn_base_channels", 64)
    model_config.setdefault("vae_latent_dim", 128)
    model_config.setdefault(
        "vit_hidden_dim",
        min(model_config.get("gnn_hidden_channels", 256), 256),
    )
    model_config.setdefault("vit_num_heads", 4)
    model_config.setdefault("vit_depth", 2)
    model_config.setdefault("vit_mlp_ratio", 4.0)
    model_config.setdefault("vit_dropout", 0.1)
    model_config.setdefault("vit_use_sigmoid", False)
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
        decoder_type=model_config["decoder_type"],
        cnn_base_channels=model_config["cnn_base_channels"],
        vae_latent_dim=model_config["vae_latent_dim"],
        vit_hidden_dim=model_config["vit_hidden_dim"],
        vit_num_heads=model_config["vit_num_heads"],
        vit_depth=model_config["vit_depth"],
        vit_mlp_ratio=model_config["vit_mlp_ratio"],
        vit_dropout=model_config["vit_dropout"],
        vit_use_sigmoid=model_config["vit_use_sigmoid"],
    )
