# %%
import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_path))

from oscillator_landscape import oscillatorLandscapceDataset

# %%
# grid_path = "/home/nauck/joined_work/landscape_generation/datasets/grids4ML/ds20"
# heatmap_path = "/home/nauck/joined_work/landscape_generation/datasets/num_sections_20/ds20"
root_path = "/home/nauck/joined_work/landscape_generation/datasets/root_path"
train_set = oscillatorLandscapceDataset(
    root_path,
    "ds20",
    20,
    split="train",
    normalize_targets=False,
    transform=None,
    pre_transform=None,
    pre_filter=None,
    force_reload=False,
)
# %%
