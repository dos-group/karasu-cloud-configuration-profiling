import copy
from typing import List
from ax import Models
from ax.modelbridge.generation_node import GenerationStep
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.registry import Cont_X_trans, Y_trans
from gpytorch.mlls import SumMarginalLogLikelihood

from classes.processed_workload import ProcessedWorkloadModel
from classes.workload_dataset import WorkloadTask
from optimization.methods.embedder import EmbedderBO, CherryPickEmbedderBO, ArrowEmbedderBO
from optimization.methods.evaluator import EvaluatorBO
from optimization.methods.karasu.models import task_resolver
from optimization.methods.optimizer import OptimizerBO
from optimization.methods.optimizer_helpers import setup_moo_experiment, manually_attach_trials, \
    create_sobol_generation_step, setup_soo_experiment
from optimization.methods.karasu.common import CustomSurrogate


class KarasuOptimizerTASK(OptimizerBO):

    def __init__(self, task: WorkloadTask, rt_target: float, num_profilings: int, **kwargs):
        super().__init__(task, rt_target, num_profilings, **kwargs)

        initial_workloads: List[ProcessedWorkloadModel] = kwargs.get("initial_workloads", [])

        task_class, acqf_class, opt_type = task_resolver[self.num_objectives]

        # vectorizer and evaluator
        embedder_class = ArrowEmbedderBO if self.optimizer_strategy_sub == "Arrow" else CherryPickEmbedderBO
        self.embedder: EmbedderBO = embedder_class(task.workloads)
        self.evaluator: EvaluatorBO = EvaluatorBO(rt_target, task.workloads, self.embedder, opt_type)

        transforms = Cont_X_trans if self.normalize else Cont_X_trans + Y_trans
        # setup experiment, get initial starting points
        gen_stra: GenerationStrategy = GenerationStrategy(
            steps=[gs for gs in [
                # quasi-random generation of initial points
                create_sobol_generation_step(self.num_init, self.seed) if self.num_init > 0 else None,
                GenerationStep(
                    model=Models.BOTORCH_MODULAR if opt_type == "SOO" else Models.MOO_MODULAR,
                    num_trials=-1,
                    enforce_num_trials=True,
                    max_parallelism=1,  # sequential evaluation
                    model_gen_kwargs=self.exp_config.get("model_gen_kwargs", None),
                    model_kwargs={
                        **self.exp_config.get("model_kwargs", {}),
                        "transforms": transforms,
                        # Wrap MOOModel with CustomSurrogate
                        "surrogate": CustomSurrogate(botorch_model_class=task_class,
                                                     model_options={
                                                         "task": copy.deepcopy(task),
                                                         "normalize": self.normalize,
                                                         "opt_class": self.optimizer_strategy_sub,
                                                         # this is an important ref-link! DO NOT REMOVE
                                                         "profiled_workloads": self.profiled_workloads,
                                                         "root_scaler": kwargs.get("root_scaler", None)
                                                     },
                                                     mll_class=SumMarginalLogLikelihood),
                        # MC-based batch Noisy Expected (HyperVolume) Improvement
                        "botorch_acqf_class": acqf_class,
                        "acquisition_options": self.exp_config.get("acquisition_options", None)
                    }
                ),
            ] if gs is not None]
        )

        if opt_type == "SOO":
            exp_name: str = "karasu_experiment_soo"
            self.ax_client = setup_soo_experiment(exp_name, gen_stra, self.embedder, self.evaluator, task)
        else:
            exp_name: str = "karasu_experiment_moo"
            self.ax_client = setup_moo_experiment(exp_name, gen_stra, self.embedder, self.evaluator, task)

        for workload in initial_workloads:
            self.ax_client = manually_attach_trials(self.ax_client, [workload], self.embedder, self.evaluator)
            self.profile(workload, optional_properties={"generation_strategy": "Historical"})

        # will do nothing if at least self.num_init workloads have actually been attached in the previous step
        self.init()
