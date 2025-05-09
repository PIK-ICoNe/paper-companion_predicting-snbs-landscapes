import os
import numpy as np
import h5py
import re

import torch

from torch_geometric.data import Data as gData
from torch_geometric.data import InMemoryDataset as InMemoryDataset

heatmaps_path = "/home/nauck/joined_work/landscape_generation/datasets/combined/"


class oscillatorLandscapceDataset(InMemoryDataset):
    r"""Oscillator networks from
    <https://doi.org/10.1063/5.0160915>`_

    The grid data has been published in the following papers:
    Toward dynamic stability assessment of power grid topologies using graph neural networks
    <https://doi.org/10.1063/5.0160915>
    Towards dynamic stability analysis of sustainable power grids using graph neural networks
    <https://www.climatechange.ai/papers/neurips2022/16>

    The grid data is available on Zenodo: <https://zenodo.org/records/8204334>

    The landscapes will be made available on Zenodo.

    """

    def __init__(
        self,
        root,
        name,
        num_sections,
        split="train",
        slice_index=None,
        normalize_targets=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
    ):
        self.name = name
        self.split = split
        self.normalize_targets = normalize_targets
        assert name in ["ds20", "ds100"]
        assert num_sections in [20]

        self.heatmap_path = os.path.join(
            heatmaps_path, f"num_sections_{num_sections}", self.name
        )
        self.root = os.path.join(
            root, name
        )  # Set self.root before calling the superclass __init__
        if slice_index == None:
            if self.split == "train":
                self.slice_index = slice(1, 7000)
            elif self.split == "val":
                self.slice_index = slice(7001, 8500)
            elif self.split == "test":
                self.slice_index = slice(8501, 10000)
        else:
            self.slice_index = slice_index
        super().__init__(
            self.root, transform, pre_transform, pre_filter, force_reload=force_reload
        )  # Use self.root here
        path = os.path.join(self.processed_dir, f"{split}.pt")
        if not os.path.exists(path) or force_reload:
            self.process()
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ["input_data.h5"]

    @property
    def processed_file_names(self):
        return ["train.pt", "valid.pt", "test.pt"]

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")

    def read_data(self):
        # Load heatmaps as targets
        targets = {}
        samples = {}
        node_features = {}
        edge_index = {}
        edge_attr = {}
        for index_grid in range(self.slice_index.start, self.slice_index.stop + 1):
            heatmap_file = os.path.join(
                self.heatmap_path, f"heatmap_grid_{index_grid:05d}.h5"
            )
            with h5py.File(heatmap_file, "r") as heatmap_hf:
                pattern = re.compile(r"basin_heatmap_(\d+)")
                node_indices = [
                    int(pattern.match(key).group(1))
                    for key in heatmap_hf.keys()
                    if pattern.match(key)
                ]
                max_node_idx = max(node_indices) if node_indices else -1
                heatmaps = []
                sample_heatmaps = []
                for node_idx in range(1, max_node_idx + 1):
                    key_heatmap = f"basin_heatmap_{node_idx}"
                    key_samples = f"samples_heatmap_{node_idx}"
                    if key_heatmap in heatmap_hf:
                        sample_heatmap = torch.tensor(
                            heatmap_hf[key_samples][()]
                        ).float()
                        heatmap_float = torch.tensor(
                            heatmap_hf[key_heatmap][()]
                        ).float()
                        heatmaps.append(
                            heatmap_float
                        )  # Ensure float32 and normalize by num_samples
                        sample_heatmaps.append(sample_heatmap)
                    else:
                        print(f"{key_heatmap} not found in the file")

                dset_grids = heatmap_hf["grids"]
                node_features[index_grid] = np.array(
                    dset_grids[str(index_grid)].get("node_features"), dtype="float32"
                ).transpose()

                edge_index[index_grid] = (
                    np.array(
                        dset_grids[str(index_grid)].get("edge_index"), dtype="int64"
                    )
                    - 1
                ).transpose()

                edge_attr[index_grid] = np.array(
                    dset_grids[str(index_grid)].get("edge_attr"), dtype="float32"
                )

                targets[index_grid] = torch.stack(heatmaps)
                samples[index_grid] = torch.stack(sample_heatmaps)

        return targets, samples, node_features, edge_index, edge_attr

    def process(self):
        print("Processing data")
        targets, samples, node_features, edge_index, edge_attr = self.read_data()
        data_list = []
        for index_grid in range(self.slice_index.start, self.slice_index.stop + 1):
            data = gData(
                x=(torch.tensor(node_features[index_grid]).unsqueeze(-1)),
                edge_index=torch.tensor(edge_index[index_grid]),
                edge_attr=torch.tensor(edge_attr[index_grid]).unsqueeze(-1),
                y=torch.tensor(targets[index_grid], dtype=torch.float32),
                sample_heatmaps=torch.tensor(samples[index_grid], dtype=torch.float32),
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), os.path.join(self.processed_dir, f"{self.split}.pt"))
        print(
            f"Processed data saved to {os.path.join(self.processed_dir, f'{self.split}.pt')}"
        )
