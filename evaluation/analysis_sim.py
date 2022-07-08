import os
import warnings
import numpy as np
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance

from classes.processed_workload import ProcessedWorkloadModel
from classes.workload_dataset import WorkloadTask
from optimization.methods.karasu.selection_strategies import SimilarWorkloadSelectionStrategy3
from evaluation import load_and_prepare

sns.set_theme()
warnings.filterwarnings("ignore", module="matplotlib\..*")

root_dir = os.path.dirname(os.path.dirname(__file__))
opt_name = "cherrypick"
num_tasks = 1
iteration = 1
percentile = 10

artifact_result_path: str = os.path.join(root_dir,
                                         "artifacts",
                                         f"RQ2_{opt_name.lower()}",
                                         f"multiple_soo_rgpe_strategies_{opt_name.lower()}_num_tasks={num_tasks}.csv")

df = load_and_prepare(artifact_result_path)
df = df[df["identifier"].str.contains("SimilarWorkloadSelectionStrategy3")]
df = df[df["profiling_counter"] == 10]
df = df[df["iteration"] == iteration]
df = df[df["percentile"] == percentile]
df = df.reset_index(drop=True)
print(df.shape)

np.random.seed(42)
index = np.random.randint(0, len(df))
workloads = df.iloc[index]["profiled_workloads"][:7]
print(f"Workloads: {len(workloads)}, {workloads[0].workload_name}")

artifact_source_path: str = os.path.join(root_dir,
                                         "artifacts",
                                         f"RQ0_{opt_name.lower()}",
                                         f"multiple_soo_{opt_name.lower()}.csv")
strategy = SimilarWorkloadSelectionStrategy3(artifact_source_path, num_tasks)
strategy = strategy.set_root_scope(percentile, iteration, "CherryPick")

data = []
for i in range(1, len(workloads)):
    sub_workloads = workloads[:i]
    workload_task: WorkloadTask = WorkloadTask.create(sub_workloads)
    score_tuples: List[Tuple[Optional[float], float, WorkloadTask]] = strategy.__get_candidates__(workload_task)
    tuples = [(tup[0], tup[-1]) for tup in score_tuples if tup[-1].iteration == 9 and tup[-1].percentile == 90]
    tuples = list(sorted(tuples, key=lambda tup: tup[-1].workloads[0].workload_name))
    data.append([tup[0] for tup in tuples])

data = np.array(data)
yticklabels = [f"Iter. {idx}" for idx in range(1, len(workloads))]
xticklabels = [f"$w_{{{idx}}}$" for idx in range(1, data.shape[-1] + 1)]
# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(5, 4.5))
sns.heatmap(data, annot=True, linewidths=.5, ax=ax,
            vmin=0, vmax=distance.euclidean([0] * 12, [1] * 12),
            xticklabels=xticklabels, yticklabels=yticklabels, fmt=".2f")
plt.savefig(os.path.join(root_dir, "artifacts", "sim.pdf"), dpi=300, bbox_inches='tight')
