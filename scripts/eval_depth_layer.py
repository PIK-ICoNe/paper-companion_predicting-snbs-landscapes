# %%
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

root_dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir_path))

ablation_1_path = "/home/nauck/joined_work/landscape_generation/ml_training/ablation_1/evaluation_results_ablation_1.csv"
res_abl1 = pd.read_csv(ablation_1_path)

ablation_2_path = "/home/nauck/joined_work/landscape_generation/ml_training/ablation_2/evaluation_results_ablation_2.csv"
res_abl2 = pd.read_csv(ablation_2_path)
res_abl2 = res_abl2.sort_values(by="idx")
# %%
# To sort a DataFrame by its row index:
# res_abl1 = res_abl1.sort_index()
num_steps_abl1 = [2, 4, 6, 8, 10]

# res_abl2 = res_abl2.sort_index()
num_steps_abl2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

# %%
plt.figure(figsize=(8, 5))
plt.plot(num_steps_abl1, res_abl1["image_test_r2"], marker="o", label="image_test_r2")
plt.plot(num_steps_abl1, res_abl1["snbs_test_r2"], marker="s", label="snbs_test_r2")
plt.xlabel("num_steps")
plt.ylabel("R2 Score")
plt.title("Test R2 Scores over num_steps (Ablation 1)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
num_steps = num_steps_abl2[0:12]
res_abl = res_abl2[0:12]
plt.figure(figsize=(8, 5))
plt.plot(num_steps, res_abl["image_test_r2"], marker="o", label="image_test_r2")
plt.plot(num_steps, res_abl["snbs_test_r2"], marker="s", label="snbs_test_r2")
plt.xlabel("num_steps")
plt.ylabel("R2 Score")
plt.title("Test R2 Scores over num_steps (Ablation 1)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
