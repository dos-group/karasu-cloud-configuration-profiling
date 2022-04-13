import copy
from ax import Models
from ax.modelbridge.generation_node import GenerationStep
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.registry import Cont_X_trans, Y_trans
from gpytorch.mlls import SumMarginalLogLikelihood

from classes.workload_dataset import WorkloadTask
from optimization.methods.embedder import EmbedderBO, CherryPickEmbedderBO, ArrowEmbedderBO
from optimization.methods.evaluator import EvaluatorBO
from optimization.methods.karasu.models import rgpe_resolver
from optimization.methods.optimizer import OptimizerBO
from optimization.methods.optimizer_helpers import setup_moo_experiment, create_sobol_generation_step, \
    setup_soo_experiment
from optimization.methods.karasu.common import CustomSurrogate


class KarasuOptimizerRGPE(OptimizerBO):

    def __init__(self, task: WorkloadTask, rt_target: float, num_profilings: int, **kwargs):
        super().__init__(task, rt_target, num_profilings, **kwargs)

        self.selection_strategy = copy.deepcopy(kwargs["selection_strategy"]).set_rgpe_scope(kwargs)

        rgpe_class, acqf_class, opt_type = rgpe_resolver[self.num_objectives]

        # vectorizer and evaluator
        embedder_class = ArrowEmbedderBO if self.optimizer_strategy_sub == "Arrow" else CherryPickEmbedderBO
        self.embedder: EmbedderBO = embedder_class(task.workloads)
        self.evaluator: EvaluatorBO = EvaluatorBO(rt_target, task.workloads, self.embedder, opt_type)

        transforms = Cont_X_trans if self.normalize else Cont_X_trans + Y_trans
        # setup experiment, get initial starting points
        gen_stra: GenerationStrategy = GenerationStrategy(
            steps=[
                # quasi-random generation of initial points
                create_sobol_generation_step(self.num_init, self.seed),
                GenerationStep(
                    model=Models.BOTORCH_MODULAR if opt_type == "SOO" else Models.MOO_MODULAR,
                    num_trials=-1,
                    enforce_num_trials=True,
                    max_parallelism=1,  # sequential evaluation
                    model_gen_kwargs=self.exp_config.get("model_gen_kwargs", None),
                    model_kwargs={
                        **self.exp_config.get("model_kwargs", {}),
                        "transforms": transforms,
                        # Wrap RGPEModel with CustomSurrogate
                        "surrogate": CustomSurrogate(botorch_model_class=rgpe_class,
                                                     model_options={
                                                         "task": task,
                                                         "num_posterior_samples": 256,
                                                         # this is an important ref-link! DO NOT REMOVE
                                                         "profiled_workloads": self.profiled_workloads,
                                                         # this is an important ref-link! DO NOT REMOVE
                                                         "selection_strategy": self.selection_strategy,
                                                         "normalize": self.normalize,
                                                         "opt_class": self.optimizer_strategy_sub
                                                     },
                                                     mll_class=SumMarginalLogLikelihood),
                        # MC-based batch Noisy Expected (HyperVolume) Improvement
                        "botorch_acqf_class": acqf_class,
                        "acquisition_options": self.exp_config.get("acquisition_options", None)
                    }
                ),
            ]
        )

        exp_name: str = "karasu_experiment_rgpe"

        if opt_type == "SOO":
            self.ax_client = setup_soo_experiment(exp_name, gen_stra, self.embedder, self.evaluator, task)
        else:
            self.ax_client = setup_moo_experiment(exp_name, gen_stra, self.embedder, self.evaluator, task)

        self.init()
