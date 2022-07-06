import ast
import copy
from random import shuffle
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler

from classes.processed_workload import ProcessedWorkloadModel, DataStatisticsModel
from classes.workload_dataset import WorkloadTask
from optimization.methods.baselines.utils_arrow import get_arrow_metrics
from optimization.methods.optimizer import OptimizerBO
from optimization.methods.karasu.common import CustomModelListGP, AbstractSelectionStrategy
from optimization.methods.karasu.optimizer_task import KarasuOptimizerTASK


class SelectionStrategy(AbstractSelectionStrategy):
    def __init__(self, path_to_file: str, num_tasks: int):
        super().__init__()
        df: pd.DataFrame = pd.read_csv(path_to_file)
        # we only need data from KarasuOptimizerTASK
        df = df.loc[df.optimizer_strategy == "KarasuOptimizerTASK", :]
        # we only take those rows with final results
        max_profile_counter: int = df["profiling_counter"].max()
        df = df.loc[df.profiling_counter == max_profile_counter, :]

        df["profiled_workloads"] = df['profiled_workloads'].map(lambda s: [ProcessedWorkloadModel.parse_raw(o)
                                                                           for o in ast.literal_eval(s)])

        self.num_tasks: int = num_tasks

        info_dict: [str, Dict[str, List[Tuple[int, int, float, WorkloadTask]]]] = {}
        for _, row in df.iterrows():
            workload_task: WorkloadTask = WorkloadTask.create(row["profiled_workloads"],
                                                              percentile=row["percentile"],
                                                              runtime_target=row["runtime_target"],
                                                              iteration=row["iteration"])

            new_entry: Tuple[int, int, float, WorkloadTask] = (row["percentile"], row["iteration"],
                                                               row["runtime_target"], workload_task)

            selector: str = row["optimizer_strategy_sub"]
            inner_dict: Dict[str, List[Tuple[int, int, float, WorkloadTask]]] = info_dict.get(selector, {})
            inner_dict[workload_task.identifier] = inner_dict.get(workload_task.identifier, []) + [new_entry]
            info_dict[selector] = inner_dict
        self.__info_dict__ = info_dict

        self.percentile: Optional[int] = None
        self.iteration: Optional[int] = None
        self.opt_class: Optional[str] = None

        self.full_task: Optional[WorkloadTask] = None
        self.rgpe_kwargs: Optional[dict] = None

    @property
    def info_dict(self):
        return self.__info_dict__[self.opt_class]

    def set_root_scope(self, percentile: int, iteration: int, opt_class: str):
        self.percentile = percentile
        self.iteration = iteration
        self.opt_class = opt_class
        return self

    def set_hetero_lengths(self, seed_val: int):
        np.random.seed(seed=seed_val)
        for k, v in self.__info_dict__.items():
            for kk, vv in v.items():
                for idx, tup in enumerate(vv):
                    t = tup[-1]
                    indices = [i for i, w in enumerate(t.workloads) if not w.abandon]
                    rand_num = np.random.choice(list(range(3, 11)))
                    winner = max(indices[2] + 1, rand_num)
                    new_t = WorkloadTask.create(t.workloads[:winner],
                                                percentile=t.percentile,
                                                runtime_target=t.runtime_target,
                                                iteration=t.iteration)
                    new_tup = (tup[0], tup[1], tup[2], new_t)
                    self.__info_dict__[k][kk][idx] = new_tup
        return self

    def set_rgpe_scope(self, rgpe_kwargs):
        self.rgpe_kwargs = copy.deepcopy(rgpe_kwargs)
        self.full_task = self.rgpe_kwargs.pop("full_task", None)
        self.rgpe_kwargs.pop("save_file", None)  # we do not want to write to file
        self.rgpe_kwargs["num_init"] = 0  # no sobol steps
        return self

    def get_prior_tasks(self, workloads: List[ProcessedWorkloadModel]) -> Tuple[List[CustomModelListGP], MinMaxScaler]:
        workload_task: WorkloadTask = WorkloadTask.create(workloads)
        candidates: List[Tuple[float, WorkloadTask]] = self.__get_candidates__(workload_task)

        self.selected_tasks = copy.deepcopy([tup[1] for tup in candidates])

        _, root_scaler = get_arrow_metrics(sum([copy.deepcopy(tup[1].workloads) for tup in candidates],
                                               copy.deepcopy(workloads)))

        result_list: List[CustomModelListGP] = []
        for rt_target, cand in candidates:
            optimizer: OptimizerBO = KarasuOptimizerTASK(self.full_task, rt_target, len(cand.workloads),
                                                         root_scaler=root_scaler,
                                                         initial_workloads=cand.workloads, **self.rgpe_kwargs)
            optimizer.run()
            result_list.append(optimizer.best_model)
        return result_list, root_scaler

    def __get_candidates__(self, workload_task: WorkloadTask) -> List[Tuple[float, WorkloadTask]]:
        raise NotImplementedError


class AbstractSameWorkloadSelectionStrategy(SelectionStrategy):
    """Return tasks from the same workload, but different runtime target (percentile) or iteration."""

    def min_required_profilings(self) -> int:
        return 1

    def validation_condition(self, cand: List[Tuple[int, int, float, WorkloadTask]]):
        raise NotImplementedError

    def __get_candidates__(self, curr_task: WorkloadTask) -> List[Tuple[float, WorkloadTask]]:
        # rt_target, task
        candidates: List[Tuple[float, WorkloadTask]] = []
        for cand_tuple in self.info_dict.get(curr_task.identifier, []):
            if self.validation_condition(cand_tuple):
                candidates.append(cand_tuple[2:])
        shuffle(candidates)
        return candidates[:self.num_tasks]


class ExactSameWorkloadSelectionStrategy(AbstractSameWorkloadSelectionStrategy):
    def validation_condition(self, cand: List[Tuple[int, int, float, WorkloadTask]]):
        return cand[0] == self.percentile and cand[1] != self.iteration


class AlmostExactSameWorkloadSelectionStrategy(AbstractSameWorkloadSelectionStrategy):
    def validation_condition(self, cand: List[Tuple[int, int, float, WorkloadTask]]):
        return cand[0] != self.percentile and cand[1] != self.iteration


class AbstractSimilarWorkloadSelectionStrategy(SelectionStrategy):
    """Return tasks from presumably similar workloads, based on metrics."""

    def min_required_profilings(self) -> int:
        return 1

    def validation_condition(self, curr_task: WorkloadTask, cand_task: WorkloadTask):
        raise NotImplementedError
    
    @staticmethod
    def get_metric_vector(metrics: Dict[str, DataStatisticsModel]):
        metric_vector = sum([[e.percentile_dict[perc] / 100.0 for perc in [10, 50, 90]] for e in metrics.values()], [])
        return metric_vector
    
    @staticmethod
    def cap(arr: np.ndarray):
        arr_mod = np.minimum(np.maximum(arr, 0), 1)
        return arr_mod
    
    def __get_candidates__(self, curr_task: WorkloadTask) -> List[Tuple[float, WorkloadTask]]:
        # similarity_score, rt_target, task
        candidates: List[Tuple[float, float, WorkloadTask]] = []
        for identifier, cand_tuple_list in self.info_dict.items():
            if self.validation_condition(curr_task, cand_tuple_list[0][-1]):
                for cand_tuple in cand_tuple_list:
                    perc, it, rt_target, cand_task = cand_tuple
                    temp_scores: List[float] = []
                    for wl in [w for w in curr_task.workloads if not w.abandon]:
                        wl_metrics: List[float] = self.get_metric_vector(wl.metrics)
                        default_value: float = distance.euclidean([0] * len(wl_metrics), [1] * len(wl_metrics))
                        direct_match: Optional[ProcessedWorkloadModel] = next((w for w in cand_task.workloads
                                                                               if wl.node_count == w.node_count
                                                                               and wl.machine_name == w.machine_name
                                                                               and not w.abandon
                                                                               ), None)
                        if direct_match:
                            direct_match_metrics: List[float] = self.get_metric_vector(direct_match.metrics)
                            temp_scores.append(distance.euclidean(self.cap(np.array(wl_metrics)),
                                                                  self.cap(np.array(direct_match_metrics))))
                        else:
                            f_ws = [w for w in cand_task.workloads if
                                    wl.machine_name == w.machine_name and not w.abandon]
                            temp_temp_scores: List[float] = []
                            for f_w in f_ws:
                                f_w_metrics: List[float] = self.get_metric_vector(f_w.metrics)
                                log_diff: float = np.log2(wl.node_count) - np.log2(f_w.node_count)
                                dist: float = distance.euclidean(self.cap(np.array(wl_metrics)),
                                                                 self.cap(np.array(f_w_metrics) / (2 ** log_diff)))
                                temp_temp_scores.append(dist)

                            temp_scores.append(sum(temp_temp_scores) / len(temp_temp_scores)
                                               if len(temp_temp_scores) else default_value)

                    candidates.append((sum(temp_scores) / len(temp_scores)
                                       if len(temp_scores) else default_value, rt_target, cand_task))

        sorted_candidates = list(sorted(candidates, key=lambda cand: cand[0]))
        return [c[1:] for c in sorted_candidates][:self.num_tasks]


class SimilarWorkloadSelectionStrategy1(AbstractSimilarWorkloadSelectionStrategy):
    def validation_condition(self, curr_task: WorkloadTask, cand_task: WorkloadTask):
        return curr_task.framework_name == cand_task.framework_name and \
               curr_task.algorithm_name == cand_task.algorithm_name and \
               curr_task.dataset_name != cand_task.dataset_name


class SimilarWorkloadSelectionStrategy2(AbstractSimilarWorkloadSelectionStrategy):
    def validation_condition(self, curr_task: WorkloadTask, cand_task: WorkloadTask):
        return curr_task.framework_name == cand_task.framework_name and \
               curr_task.algorithm_name != cand_task.algorithm_name and \
               curr_task.dataset_name != cand_task.dataset_name


class SimilarWorkloadSelectionStrategy3(AbstractSimilarWorkloadSelectionStrategy):
    def validation_condition(self, curr_task: WorkloadTask, cand_task: WorkloadTask):
        return curr_task.framework_name != cand_task.framework_name and \
               curr_task.algorithm_name != cand_task.algorithm_name and \
               curr_task.dataset_name != cand_task.dataset_name
