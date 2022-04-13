import dataclasses
from typing import List, Union, Type

from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_modular.surrogate import Surrogate
from botorch import fit_gpytorch_model
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.acquisition.input_constructors import acqf_input_constructor, construct_inputs_qNEHVI, \
    construct_inputs_qNEI
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from botorch.models import ModelListGP
from botorch.utils.containers import TrainingData
from gpytorch.mlls import SumMarginalLogLikelihood

from classes.workload_dataset import WorkloadTask
from optimization.methods.baselines.utils_arrow import ArrowModel
from optimization.methods.baselines.utils_cherrypick import CherryPickModel


class CustomModelListGP(ModelListGP):
    def __init__(self, *models: Union[CherryPickModel, ArrowModel]):
        super().__init__(*models)
        self.custom_models: List[Union[CherryPickModel, ArrowModel]] = [*models]
        self.opt_class: str = self.custom_models[0].opt_class


class CustomSurrogate(Surrogate):
    def fit(
            self,
            training_data: TrainingData,
            search_space_digest: SearchSpaceDigest,
            metric_names: List[str],
            **kwargs
    ) -> None:
        """We override this function because we don't fit the model the 'classical way'."""
        self.construct(
            training_data=training_data,
            metric_names=metric_names,
            **dataclasses.asdict(search_space_digest),
            **kwargs
        )


class CustomqNoisyExpectedHypervolumeImprovement(qNoisyExpectedHypervolumeImprovement):
    pass


class CustomqNoisyExpectedImprovement(qNoisyExpectedImprovement):
    pass


@acqf_input_constructor(CustomqNoisyExpectedHypervolumeImprovement)
def construct_custom_inputs_qNEHVI(*args, **kwargs):
    inputs = construct_inputs_qNEHVI(*args, **kwargs)
    inputs["cache_root"] = kwargs.get("cache_root", True)
    return inputs


@acqf_input_constructor(CustomqNoisyExpectedImprovement)
def construct_custom_inputs_qNEI(*args, **kwargs):
    inputs = construct_inputs_qNEI(*args, **kwargs)
    inputs["cache_root"] = kwargs.get("cache_root", True)
    return inputs


class AbstractSelectionStrategy:
    def __init__(self):
        self.selected_tasks: List[WorkloadTask] = []

    def get_prior_tasks(self, *args, **kwargs):
        raise NotImplementedError

    def min_required_profilings(self) -> int:
        raise NotImplementedError


def get_fitted_karasu_model(train_X: list, train_Y: list, train_Yvar: list, arrow_metrics: list,
                            opt_class: Type[Union[CherryPickModel, ArrowModel]],
                            state_dict=None, refit: bool = True, normalize: bool = True, **kwargs):
    models = []
    for idx, (train_x, train_y, train_yvar, arrow_met) in enumerate(zip(train_X, train_Y, train_Yvar, arrow_metrics)):
        model = opt_class(train_x, train_y, train_yvar, arrow_met, normalize=normalize)
        models.append(model)

    model = CustomModelListGP(*models)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    if state_dict is None or refit:
        mll = SumMarginalLogLikelihood(model.likelihood, model).to(train_X[0])
        fit_gpytorch_model(mll)
    return model



