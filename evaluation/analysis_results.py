import ast
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy

from classes.processed_workload import ProcessedWorkloadModel

warnings.filterwarnings("ignore", module="matplotlib\..*")

root_dir = os.path.dirname(os.path.dirname(__file__))


def get_workload_cost_valid(workload: ProcessedWorkloadModel, rt_target: float):
    return workload.cost if workload.completed and workload.runtime <= rt_target else np.nan


def get_workload_energy_valid(workload: ProcessedWorkloadModel, rt_target: float):
    return workload.energy if workload.completed and workload.runtime <= rt_target else np.nan


def eval_objective(my_df: pd.DataFrame, target_col: str, ref_col: str, cond):
    my_df[target_col] = my_df[['runtime_target', 'profiled_workloads']].apply(cond, axis=1)
    my_df[target_col] = my_df[target_col] / my_df[ref_col]
    return my_df


def load_and_prepare(path_to_file: str):
    df = pd.read_csv(path_to_file)
    print(df.shape, path_to_file)

    if "selection_strategy" not in list(df.columns):
        df["selection_strategy"] = ""

    df["optimizer_strategy_sub"] = "NaiveBO" if "cherrypick" in path_to_file else "AugmentedBO"

    cols = ['optimizer_strategy', 'selection_strategy']
    df['identifier'] = df[cols].apply(lambda x: '-'.join([str(e) for e in x.tolist() if len(str(e))]), axis=1)

    df["profiled_workloads"] = df['profiled_workloads'].map(lambda s: [ProcessedWorkloadModel.parse_raw(o)
                                                                       for o in ast.literal_eval(s)])

    df = eval_objective(df, "best_cost_found", "best_cost",
                        lambda row: min([get_workload_cost_valid(w, row.to_list()[0]) for w in row.to_list()[1]]))
    df = eval_objective(df, "best_energy_found", "best_energy",
                        lambda row: min([get_workload_energy_valid(w, row.to_list()[0]) for w in row.to_list()[1]]))
    df = eval_objective(df, "total_search_cost", "best_cost", lambda row: sum([w.cost for w in row.to_list()[1]]))
    df["total_search_time"] = df["profiled_workloads"].map(lambda pw: sum([e.runtime for e in pw]))
    df["total_search_cost"] = df["profiled_workloads"].map(lambda pw: sum([e.cost for e in pw]))
    df["profiling_counter_completed"] = df["profiled_workloads"].map(lambda pw: sum([e.completed for e in pw]))
    df["profiling_counter_not_completed_not_abandon"] = df["profiled_workloads"].map(
        lambda pw: sum([(not e.completed and not e.abandon) for e in pw]))

    if "num_tasks" in list(df.columns):
        df["num_tasks"] = df["num_tasks"].astype('int')
    else:
        df["num_tasks"] = None

    return df


cherrypick = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ0_cherrypick/multiple_soo_cherrypick.csv"))
arrow = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ0_arrow/multiple_soo_arrow.csv"))

rq1_cherrypick_k1 = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ1_cherrypick/multiple_soo_rgpe_strategies_cherrypick_num_tasks=1.csv"))
rq1_cherrypick_k5 = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ1_cherrypick/multiple_soo_rgpe_strategies_cherrypick_num_tasks=5.csv"))
rq1_cherrypick_k9 = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ1_cherrypick/multiple_soo_rgpe_strategies_cherrypick_num_tasks=9.csv"))
rq1_arrow_k1 = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ1_arrow/multiple_soo_rgpe_strategies_arrow_num_tasks=1.csv"))
rq1_arrow_k5 = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ1_arrow/multiple_soo_rgpe_strategies_arrow_num_tasks=5.csv"))
rq1_arrow_k9 = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ1_arrow/multiple_soo_rgpe_strategies_arrow_num_tasks=9.csv"))

rq2_cherrypick_k1 = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ2_cherrypick/multiple_soo_rgpe_strategies_cherrypick_num_tasks=1.csv"))
rq2_cherrypick_k3 = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ2_cherrypick/multiple_soo_rgpe_strategies_cherrypick_num_tasks=3.csv"))
rq2_cherrypick_k5 = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ2_cherrypick/multiple_soo_rgpe_strategies_cherrypick_num_tasks=5.csv"))
rq2_arrow_k1 = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ2_arrow/multiple_soo_rgpe_strategies_arrow_num_tasks=1.csv"))
rq2_arrow_k3 = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ2_arrow/multiple_soo_rgpe_strategies_arrow_num_tasks=3.csv"))
rq2_arrow_k5 = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ2_arrow/multiple_soo_rgpe_strategies_arrow_num_tasks=5.csv"))

rq3_moo_cherrypick = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ3_cherrypick/multiple_moo_cherrypick.csv"))
rq3_moo_cherrypick_k3 = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ3_cherrypick/multiple_moo_rgpe_strategies_cherrypick_num_tasks=3.csv"))


def cost_and_time_at_stopping_condition(df, id_, n):
    for _, row in df.iterrows():
        if row.acqf_value < 0.1 and (n is None or row.profiling_counter >= n):
            break
    return {
        id_: row[id_],
        "timeout": row.profiling_counter_not_completed_not_abandon / row.profiling_counter,
        "total_search_cost": row.total_search_cost,
        "total_search_time": row.total_search_time,
        "best_cost_found": row.best_cost_found,
    }


def cost_and_time_df(g, id_, n=None):
    result = []
    for name, group in g:
        result.append(cost_and_time_at_stopping_condition(group, id_, n))
    return pd.DataFrame(result)


df = pd.concat((
    cherrypick,
    arrow,
    rq1_cherrypick_k1,
    rq1_cherrypick_k5,
    rq1_cherrypick_k9,
    rq1_arrow_k1,
    rq1_arrow_k5,
    rq1_arrow_k9,
), axis=0, ignore_index=True)

df = df[(df["iteration"] == 1) | (df["iteration"] == 5) | (df["iteration"] == 9)]
df = df[df["identifier"] != "KarasuOptimizerRGPE-ExactSameWorkloadSelectionStrategy"]  # filter out Karasu-RQ1-UC1

df["approach"] = "w/o Karasu"
df.loc[(df[
            "identifier"] == "KarasuOptimizerRGPE-AlmostExactSameWorkloadSelectionStrategy"), "approach"] = "Karasu (#Models=" + \
                                                                                                            df[
                                                                                                                "num_tasks"].astype(
                                                                                                                str) + ")"


sns.set_style("whitegrid")

f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharey=True, figsize=(5, 5))
f.tight_layout(h_pad=2.5, w_pad=0)

sns.boxplot(data=df[df["optimizer_strategy_sub"] == "NaiveBO"], x="profiling_counter", y="best_cost_found",
            hue="approach", ax=ax1,
            showfliers=False, palette=["#968D88", *sns.color_palette("flare")[:3]])
sns.boxplot(data=df[df["optimizer_strategy_sub"] == "AugmentedBO"], x="profiling_counter", y="best_cost_found",
            hue="approach", ax=ax2,
            showfliers=False, palette=["#968D88", *sns.color_palette("flare")[:3]])

lim = (1, 2.5)

ax1.set_ylim(lim)
ax1.get_legend().remove()
ax1.set_ylabel("Best cost (normalized)")
ax1.set_xlabel("")
ax1.set_title("NaiveBO (CherryPick)")

ax2.set_ylim(lim)
ax2.get_legend().remove()
ax2.set_ylabel("Best cost (normalized)")
ax2.set_xlabel("Number of profiling runs")
ax2.set_title("AugmentedBO (Arrow)")

handles, labels = ax1.get_legend_handles_labels()
f.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.96, 0.97), framealpha=1, edgecolor="white")

plt.savefig(os.path.join(root_dir, "artifacts", "rq1.pdf"), dpi=300, bbox_inches='tight')

df2 = df[(df["optimizer_strategy_sub"] == "NaiveBO")]
g = df2.groupby(by=["framework_name", "algorithm_name", "dataset_name", "percentile", "iteration", "approach"])
y = cost_and_time_df(g, "approach")

f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(4.8, 5))
f.tight_layout(h_pad=2, w_pad=0)

sns.boxplot(data=y, y="approach", x="total_search_time", ax=ax1,
            order=["w/o Karasu", "Karasu (#Models=1)", "Karasu (#Models=5)", "Karasu (#Models=9)"],
            showfliers=False, palette=["#968D88", *sns.color_palette("flare")[:3]])

sns.boxplot(data=y, y="approach", x="total_search_cost", ax=ax2,
            order=["w/o Karasu", "Karasu (#Models=1)", "Karasu (#Models=5)", "Karasu (#Models=9)"],
            showfliers=False, palette=["#968D88", *sns.color_palette("flare")[:3]])

sns.boxplot(data=y, y="approach", x="best_cost_found", ax=ax3,
            order=["w/o Karasu", "Karasu (#Models=1)", "Karasu (#Models=5)", "Karasu (#Models=9)"],
            showfliers=False,
            palette=["#968D88", *sns.color_palette("flare")[:3]])

sns.barplot(data=y, y="approach", x="timeout", ax=ax4,
            order=["w/o Karasu", "Karasu (#Models=1)", "Karasu (#Models=5)", "Karasu (#Models=9)"],
            ci=None, palette=["#968D88", *sns.color_palette("flare")[:3]])

ax1.set_ylabel("")
ax1.set_xlabel("Total search time (hours)")
ax1.set_xlim((0, 8 * 3600))
ax1.set_xticks([3600 * i for i in range(9)])
ax1.set_xticklabels([f"{i}" for i in range(9)])

ax2.set_ylabel("")
ax2.set_xlabel("Total search cost ($)")
ax2.set_xlim((0, 11))

ax3.set_ylabel("")
ax3.set_xlabel("Best cost (normalized)")
ax3.set_xlim((1, 1.82))

ax4.set_ylabel("")
ax4.set_xlabel("Percentage of profilings with timeout")  # timeout
ax4.set_xlim((0, 0.06))
ax4.set_xticks([0, .01, .02, .03, .04, .05, .06])
ax4.set_xticklabels(["0%", "1%", "2%", "3%", "4%", "5%", "6%"])

plt.savefig(os.path.join(root_dir, "artifacts", "rq1_total_cost.pdf"), dpi=300, bbox_inches='tight')


def extend_by_num_tasks(df, num_tasks):
    temp_df = copy.deepcopy(df)
    temp_df["num_tasks"] = num_tasks
    temp_df["num_tasks"] = temp_df["num_tasks"].astype('int')
    return temp_df


df = pd.concat((
    pd.concat((extend_by_num_tasks(cherrypick, num_tasks) for num_tasks in [1, 3, 5]), axis=0, ignore_index=True),
    pd.concat((extend_by_num_tasks(arrow, num_tasks) for num_tasks in [1, 3, 5]), axis=0, ignore_index=True),
    rq2_cherrypick_k1,
    rq2_cherrypick_k3,
    rq2_cherrypick_k5,
    rq2_arrow_k1,
    rq2_arrow_k3,
    rq2_arrow_k5,
), axis=0, ignore_index=True)

df = df[(df["iteration"] == 1) | (df["iteration"] == 5) | (df["iteration"] == 9)]
df = df[df["num_tasks"] == 3]

df.loc[df.identifier == "KarasuOptimizerTASK", "identifier"] = "w/o Karasu"
df.loc[df.identifier == "KarasuOptimizerRGPE-SimilarWorkloadSelectionStrategy1", "identifier"] = "Karasu (Policy A)"
df.loc[df.identifier == "KarasuOptimizerRGPE-SimilarWorkloadSelectionStrategy2", "identifier"] = "Karasu (Policy B)"
df.loc[df.identifier == "KarasuOptimizerRGPE-SimilarWorkloadSelectionStrategy3", "identifier"] = "Karasu (Policy C)"


f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharey=True, figsize=(5, 5))
f.tight_layout(h_pad=2.5, w_pad=0)

sns.boxplot(data=df[df["optimizer_strategy_sub"] == "NaiveBO"], x="profiling_counter", y="best_cost_found",
            hue="identifier", ax=ax1,
            showfliers=False, palette=["#968D88", *sns.color_palette("crest")[:3]])
sns.boxplot(data=df[df["optimizer_strategy_sub"] == "AugmentedBO"], x="profiling_counter", y="best_cost_found",
            hue="identifier", ax=ax2,
            showfliers=False, palette=["#968D88", *sns.color_palette("crest")[:3]])

lim = (1, 2.5)

ax1.set_ylim(lim)
ax1.get_legend().remove()
ax1.set_ylabel("Best cost (normalized)")
ax1.set_xlabel("")
ax1.set_title("NaiveBO (CherryPick)")

ax2.set_ylim(lim)
ax2.get_legend().remove()
ax2.set_ylabel("Best cost (normalized)")
ax2.set_xlabel("Number of profiling runs")
ax2.set_title("AugmentedBO (Arrow)")

handles, labels = ax1.get_legend_handles_labels()
f.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.96, 0.97), framealpha=1, edgecolor="white")

plt.savefig(os.path.join(root_dir, "artifacts", "rq2.pdf"), dpi=300, bbox_inches='tight')


df2 = df[(df["optimizer_strategy_sub"] == "NaiveBO")]
g = df2.groupby(by=["framework_name", "algorithm_name", "dataset_name", "percentile", "iteration", "identifier"])
y = cost_and_time_df(g, "identifier")


f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharey=True, figsize=(5, 5))
f.tight_layout(h_pad=2, w_pad=0)

sns.boxplot(data=y, y="identifier", x="total_search_time", ax=ax1,
            order=["w/o Karasu", "Karasu (Policy A)", "Karasu (Policy B)", "Karasu (Policy C)"],
            showfliers=False, palette=["#968D88", *sns.color_palette("crest")[:3]])

sns.boxplot(data=y, y="identifier", x="total_search_cost", ax=ax2,
            order=["w/o Karasu", "Karasu (Policy A)", "Karasu (Policy B)", "Karasu (Policy C)"],
            showfliers=False, palette=["#968D88", *sns.color_palette("crest")[:3]])

sns.boxplot(data=y, y="identifier", x="best_cost_found", ax=ax3,
            order=["w/o Karasu", "Karasu (Policy A)", "Karasu (Policy B)", "Karasu (Policy C)"],
            showfliers=False, palette=["#968D88", *sns.color_palette("crest")[:3]])

sns.barplot(data=y, y="identifier", x="timeout", ax=ax4,
            order=["w/o Karasu", "Karasu (Policy A)", "Karasu (Policy B)", "Karasu (Policy C)"],
            ci=None, palette=["#968D88", *sns.color_palette("crest")[:3]])

ax1.set_ylabel("")
ax1.set_xlabel("Total search time (hours)")
ax1.set_xlim((0, 8 * 3600))
ax1.set_xticks([3600 * i for i in range(9)])
ax1.set_xticklabels([f"{i}" for i in range(9)])

ax2.set_ylabel("")
ax2.set_xlabel("Total search cost ($)")
ax2.set_xlim((0, 11))

ax3.set_ylabel("")
ax3.set_xlabel("Best cost (normalized)")
ax3.set_xlim((1, 1.82))

ax4.set_ylabel("")
ax4.set_xlabel("Percentage of profilings with timeout")
ax4.set_xlim((0, 0.06))
ax4.set_xticks([0, .01, .02, .03, .04, .05, .06])
ax4.set_xticklabels(["0%", "1%", "2%", "3%", "4%", "5%", "6%"])

plt.savefig(os.path.join(root_dir, "artifacts", "rq2_total_cost.pdf"), dpi=300, bbox_inches='tight')


def filter_rows(df, workload=None):
    if workload:
        framework_name, algorithm_name, dataset_name = workload
        df = df[(df["framework_name"] == framework_name) & (df["algorithm_name"] == algorithm_name) & (
                    df["dataset_name"] == dataset_name)]
    df = df[(df["iteration"] == 1) | (df["iteration"] == 5) | (df["iteration"] == 9)]
    return df[["percentile", "iteration", "profiling_counter", "best_cost_found", "best_energy_found"]]


def mmo_plot_df(df):
    df = df.groupby(["profiling_counter"]).mean()
    return df[["best_cost_found", "best_energy_found"]]


sns.set_style("ticks")


def rq3_soo_vs_moo(f, ax):
    soo = mmo_plot_df(filter_rows(rq2_cherrypick_k3, workload=("hadoop", "terasort", "bigdata")))
    moo = mmo_plot_df(filter_rows(rq3_moo_cherrypick_k3, workload=("hadoop", "terasort", "bigdata")))

    best_cost = 0.371218
    best_energy = 33.384828

    ax.plot(soo.best_cost_found * best_cost, soo.best_energy_found * best_energy, "-o", label="SOO NaiveBO\nw/ Karasu",
            color="#1481BA")
    ax.plot(moo.best_cost_found * best_cost, moo.best_energy_found * best_energy, "-o", label="MOO NaiveBO\nw/ Karasu",
            color="#FC7A57")

    # label points on the plot
    for i, x, y in zip(range(1, 11), soo.best_cost_found * best_cost, soo.best_energy_found * best_energy):
        if i in [5, 7, 8, 9]:
            continue
        x_pad = 0.014 if i != 10 else 0.018
        ax.text(x=x - x_pad,
                y=y - 0.35,
                s='{:.0f}'.format(i),
                color='black')

    for i, x, y in zip(range(1, 11), moo.best_cost_found * best_cost, moo.best_energy_found * best_energy):
        if i in [8, 9]:
            continue
        ax.text(x=x + 0.005,
                y=y - 0.35,
                s='{:.0f}'.format(i),
                color='black')

    ax.text(x=0.455,
            y=38.2,
            s="1...10 profiling runs",
            color='black')

    ax.set_ylim((37, 58))
    ax.set_xlim((0.36, 0.58))

    ax.set_xlabel("Cost ($)")
    ax.set_ylabel("Energy (Wh)")

    handles, labels = ax.get_legend_handles_labels()
    f.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.07, 0.945), frameon=False)


def rq3_baseline_vs_karasu(f, ax):
    baseline = mmo_plot_df(filter_rows(rq3_moo_cherrypick))
    karasu = mmo_plot_df(filter_rows(rq3_moo_cherrypick_k3))
    ax.plot(baseline.best_cost_found, baseline.best_energy_found, "-o", label="MOO NaiveBO", color="#827873")
    ax.plot(karasu.best_cost_found, karasu.best_energy_found, "-o", label="MOO NaiveBO\nw/ Karasu", color="#FC7A57")

    # label points on the plot
    for i, x, y in zip(range(1, 11), baseline.best_cost_found, baseline.best_energy_found):
        if i in [7, 8, 9]:
            continue
        x_pad = 0.09 if i != 10 else 0.12
        ax.text(x=x - x_pad,
                y=y - 0.025,
                s='{:.0f}'.format(i),
                color='black')

    for i, x, y in zip(range(1, 11), karasu.best_cost_found, karasu.best_energy_found):
        if i in [7, 8, 9]:
            continue
        ax.text(x=x + 0.05,
                y=y - 0.03,
                s='{:.0f}'.format(i),
                color='black')

    ax.text(x=1.57,
            y=1.08,
            s="1...10 profiling runs",
            color='black')

    minmax = (1, 2.35)
    ax.set_ylim(minmax)
    ax.set_xlim(minmax)

    ax.set_xlabel("Best cost (normalized)")
    ax.set_ylabel("Best energy (normalized)")

    handles, labels = ax.get_legend_handles_labels()
    f.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.57, 0.945), frameon=False)


f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(5.8, 2.8))
f.tight_layout(w_pad=2)
rq3_soo_vs_moo(f, ax1)
rq3_baseline_vs_karasu(f, ax2)
plt.savefig(os.path.join(root_dir, "artifacts", "rq3.pdf"), dpi=300, bbox_inches='tight')
