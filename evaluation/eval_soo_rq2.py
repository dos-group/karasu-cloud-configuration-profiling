import copy
from typing import List
import pandas as pd
import torch.multiprocessing as torch_mp
import os
import torch

import numpy as np

from evaluation.experiment import Experiment, ExperimentConfig
from optimization.methods.karasu.optimizer_rgpe import KarasuOptimizerRGPE
from optimization.methods.karasu.selection_strategies import ExactSameWorkloadSelectionStrategy, \
    AlmostExactSameWorkloadSelectionStrategy, SimilarWorkloadSelectionStrategy1, SimilarWorkloadSelectionStrategy2, \
    SimilarWorkloadSelectionStrategy3


class SooRGPEStrategiesExperiment(Experiment):
    def _extract_tuples(self, *args, **kwargs) -> List[tuple]:
        if os.path.exists(self.save_file):
            cols = ["percentile", "iteration", "framework_name", "algorithm_name", "dataset_name",
                    "optimizer_strategy", "selection_strategy", "optimizer_strategy_sub", "num_tasks"]
            temp_df: pd.DataFrame = pd.read_csv(self.save_file, usecols=cols)
            temp_df = temp_df[cols]
            return list(set([tuple(my_list) for my_list in temp_df.values.tolist()]))
        else:
            return []

    def _run(self, lock):
        for _, task_obj in sorted(self.dataset.workload_tasks.items(), key=lambda tup: tup[0]):
            seed_arr = np.arange(100 * 20).reshape(100, 20).astype(int)
            for iteration in self.experiment_config.iterations:
                opt_specs_list = []
                for percentile in self.experiment_config.percentiles:
                    for selection_strategy in self.selection_strategies:
                        for optimizer_class, optimizer_class_sub in self.optimization_classes:

                            if (percentile, iteration, task_obj.framework_name, task_obj.algorithm_name,
                                task_obj.dataset_name,
                                optimizer_class.__name__, selection_strategy.__class__.__name__,
                                optimizer_class_sub, selection_strategy.num_tasks) in self.existing_tuples:
                                continue

                            my_selection_strategy = copy.deepcopy(selection_strategy).set_root_scope(percentile,
                                                                                                     iteration,
                                                                                                     optimizer_class_sub)

                            seed_val: int = int(seed_arr[percentile, iteration] * 10) + 10

                            completed_workloads = [w for w in task_obj.workloads if w.completed]

                            rt_target: float = np.percentile([w.runtime for w in completed_workloads], percentile)

                            workloads_valid = [w for w in completed_workloads if w.runtime <= rt_target]
                            best_cost: float = min([w.cost for w in completed_workloads if w.runtime <= rt_target])
                            best_energy: float = min([w.energy for w in completed_workloads if w.runtime <= rt_target])

                            exp_config: dict = {
                                "full_task": copy.deepcopy(task_obj),
                                "selection_strategy": my_selection_strategy,
                                "exp_config": self.optimizer_config.dict(),
                                "seed": seed_val,
                                "num_objectives": 1,
                                "num_init": selection_strategy.min_required_profilings(),
                                "normalize": True,
                                "save_file": self.save_file,
                                "base_entry": {
                                    "#all_candidates": len(task_obj.workloads),
                                    "#valid_candidates": len(workloads_valid),
                                    "framework_name": task_obj.framework_name,
                                    "algorithm_name": task_obj.algorithm_name,
                                    "dataset_name": task_obj.dataset_name,
                                    "percentile": percentile,
                                    "iteration": iteration,
                                    "runtime_target": rt_target,
                                    "best_cost": best_cost,
                                    "best_energy": best_energy,
                                    "optimizer_strategy": optimizer_class.__name__,
                                    "optimizer_strategy_sub": optimizer_class_sub,
                                    "selection_strategy": selection_strategy.__class__.__name__,
                                    "num_tasks": my_selection_strategy.num_tasks
                                }}

                            opt_specs_list.append((
                                optimizer_class,
                                (task_obj, rt_target, self.experiment_config.num_profilings),
                                exp_config
                            ))

                if len(opt_specs_list):
                    torch.set_num_threads(1)
                    with torch_mp.Pool(self.dataset_config.max_parallel, initializer=self.pool_init) as pool:
                        pool.map(self.run_optimizer, [(lock,) + opt_specs for opt_specs in opt_specs_list])


if __name__ == '__main__':
    torch_mp.set_sharing_strategy('file_system')
    torch_mp.set_start_method('spawn')

    for num_tasks in [1, 3, 5]:
        for opt_class, opt_name in [
            (KarasuOptimizerRGPE, "CherryPick"),
            (KarasuOptimizerRGPE, "Arrow")
        ]:
            artifact_source_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                     "artifacts",
                                                     f"RQ0_{opt_name.lower()}",
                                                     f"multiple_soo_{opt_name.lower()}.csv")

            strategies = [strategy_cls(artifact_source_path, num_tasks)
                          for strategy_cls in [SimilarWorkloadSelectionStrategy1,
                                               SimilarWorkloadSelectionStrategy2,
                                               SimilarWorkloadSelectionStrategy3]]

            multiple_experiment: SooRGPEStrategiesExperiment = SooRGPEStrategiesExperiment(
                "scout_multiple", experiment_config=ExperimentConfig(
                    percentiles=[10, 30, 50, 70, 90],
                    iterations=[1, 5, 9],
                    num_profilings=10)) \
                .set_experiment_name(f"multiple_soo_rgpe_strategies_{opt_name.lower()}_num_tasks={num_tasks}") \
                .set_selection_strategies(strategies) \
                .set_optimization_classes([(opt_class, opt_name)])
            multiple_experiment.run("INFO", f"RQ2_{opt_name.lower()}")
