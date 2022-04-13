import numpy as np
from typing import List, Union, Dict
from classes.processed_workload import ProcessedWorkloadModel
from optimization.methods.embedder import EmbedderBO


class EvaluatorBO:
    def __init__(self, rt_target: float, workloads: List[ProcessedWorkloadModel], embedder: EmbedderBO, opt_type: str):
        self.runtime_target: float = rt_target
        self.embedder: EmbedderBO = embedder
        self.workloads: List[ProcessedWorkloadModel] = workloads
        self.features: List[List[Union[int, float, str]]] = [self.embedder.vectorize(w) for w in self.workloads]
        self.opt_type: str = opt_type

    @staticmethod
    def cost_transform(workload: ProcessedWorkloadModel) -> float:
        return np.log(workload.cost)

    @staticmethod
    def energy_transform(workload: ProcessedWorkloadModel) -> float:
        return np.log(workload.energy)

    def __call__(self, parameterization, *args, **kwargs) -> Dict[str, tuple]:
        feature_list: List[Union[int, float, str]] = self.embedder.reconstruct(parameterization)
        objective: float
        constraint: float

        workload: ProcessedWorkloadModel = self.workloads[self.features.index(feature_list)]

        cost_objective = self.cost_transform(workload)
        energy_objective = self.energy_transform(workload)
        rt_constraint = np.log(workload.runtime)

        noise_sd = 0.1
        cost_obj_noise, energy_obj_noise, const_noise = np.random.normal(0, noise_sd, 3)

        result: dict = {
            "objective_cost": (cost_objective + cost_obj_noise, noise_sd),
            "constraint_rt": (rt_constraint + const_noise, noise_sd)
        }

        if self.opt_type == "SOO":
            return result
        else:
            return {
                **result,
                "objective_energy": (energy_objective + energy_obj_noise, noise_sd)
            }
