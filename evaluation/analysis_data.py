import os
import pickle
import warnings
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from classes.processed_workload import ProcessedWorkloadModel

warnings.filterwarnings("ignore", module="matplotlib\..*")

root_dir = os.path.dirname(os.path.dirname(__file__))
with open(os.path.join(root_dir, "data", "scout_multiple_raw.p"), "rb") as f:
    workloads: List[ProcessedWorkloadModel] = pickle.load(f)

rows = []
for workload in workloads:
    rows.append({
        "cost": workload.cost,
        "energy": workload.energy,
        "machine_name": workload.machine_name,
        "instance_type": workload.machine_name.split("4")[0],
        "instance_size": workload.machine_name.split("4")[1],
        "node_count": workload.node_count,
        "framework_name": workload.framework_name,
        "algorithm_name": workload.algorithm_name,
        "dataset_name": workload.dataset_name,
        "runtime": workload.runtime,
        "completed": workload.completed,
    })
df = pd.DataFrame(rows)

sns.set_style("ticks")


def cost_energy_relplot(df, ax):
    sns.scatterplot(data=df, x="cost", y="energy",
                    hue="instance_type",
                    marker="X",
                    ax=ax)


f, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2.5))
cost_energy_relplot(df, ax=ax)

ax.set_xlim(0, 5.3)
ax.set_xlabel("Cost ($)")

ax.set_ylim(0, 530)
ax.set_ylabel("Energy (Wh)")

ax.get_legend().remove()
handles, labels = ax.get_legend_handles_labels()
f.legend(handles, ["m4", "c4", "r4"], loc='upper left', title="           Machine type",
         bbox_to_anchor=(0.06, 0.95), frameon=False, handlelength=1)

f.tight_layout()
plt.savefig(os.path.join(root_dir, "artifacts", "objective_correlation.pdf"), dpi=300)

framework_name, algorithm_name, dataset_name = ("hadoop", "terasort", "bigdata")
example = df[(df["framework_name"] == framework_name) & (df["algorithm_name"] == algorithm_name) & (
        df["dataset_name"] == dataset_name)]

f, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(5, 2.7))
sns.stripplot(data=df, x="runtime", y="instance_size", hue="instance_type", dodge=True, marker=".",
              order=["L", "XL", "XXL"], hue_order=["M", "C", "R"], palette=sns.dark_palette("#69d", reverse=True),
              ax=ax)
sns.boxplot(data=df, x="runtime", y="instance_size", hue="instance_type", showfliers=False, order=["L", "XL", "XXL"],
            hue_order=["M", "C", "R"], palette=sns.dark_palette("#69d", reverse=True), ax=ax)

ax.set_xlabel("Runtime (minutes)")
ax.set_xticks([600 * i for i in range(13)])
ax.set_xticklabels([f"{10 * i}" for i in range(13)])

ax.set_ylabel("")
ax.set_yticklabels(["m4.large\nm4.xlarge\nm4.2xlarge",
                    "c4.large\nc4.xlarge\nc4.2xlarge",
                    "r4.large\nr4.xlarge\nr4.2xlarge"])

ax.get_legend().remove()

f.tight_layout()
plt.savefig(os.path.join(root_dir, "artifacts", "machine_type_runtimes.pdf"), dpi=300)
