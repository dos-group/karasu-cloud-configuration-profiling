import logging
from typing import List, Optional, Dict

import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.containers import TrainingData
from gpytorch import lazify
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.lazy import PsdSumLazyTensor, BlockDiagLazyTensor
from gpytorch.likelihoods import LikelihoodList
from gpytorch.models import GP

from classes.processed_workload import ProcessedWorkloadModel
from optimization.methods.baselines.utils_arrow import get_arrow_metrics, ArrowModel
from optimization.methods.baselines.utils_cherrypick import CherryPickModel
from optimization.methods.karasu.common import get_fitted_karasu_model, CustomModelListGP, AbstractSelectionStrategy, \
    CustomqNoisyExpectedImprovement, CustomqNoisyExpectedHypervolumeImprovement


class TASKModel1(CustomModelListGP):
    _num_outputs = 2  # metadata for botorch

    @classmethod
    def construct_inputs(cls, training_data: TrainingData, **kwargs):
        profiled_workloads: List[ProcessedWorkloadModel] = kwargs["profiled_workloads"]
        profiled_workloads = [w for w in profiled_workloads if not w.abandon]
        opt_class: str = kwargs["opt_class"]

        root_scaler = kwargs.get("root_scaler", None)
        arrow_metrics, _ = get_arrow_metrics(profiled_workloads, root_scaler=root_scaler)
        arrow_metrics = arrow_metrics.to(training_data.Xs[0])
        arrow_metrics = [arrow_metrics] * len(training_data.Xs)

        model = get_fitted_karasu_model(training_data.Xs, training_data.Ys, training_data.Yvars,
                                        arrow_metrics,
                                        ArrowModel if opt_class == "Arrow" else CherryPickModel,
                                        state_dict=kwargs.get("state_dict", None),
                                        refit=kwargs.get("refit", True),
                                        normalize=kwargs.get("normalize", True))

        logging.info(f"TASK model (opt_class={opt_class}) obtained with {len(profiled_workloads)} samples... "
                     f"[{kwargs['task'].identifier}]")
        return {"model": model}

    def __init__(self, model: CustomModelListGP):
        super().__init__(*model.models)


class TASKModel2(TASKModel1):
    _num_outputs = 3  # metadata for botorch


class RGPEModel1(GP, GPyTorchModel):
    _num_outputs = 2  # metadata for botorch

    @staticmethod
    def compute_ranking_loss(f_samps, target_y):
        def roll_col(X, shift):
            return torch.cat((X[..., -shift:], X[..., :-shift]), dim=-1)

        n = target_y.shape[0]
        if f_samps.ndim == 3:
            # Compute ranking loss for target model
            # take cartesian product of target_y
            cartesian_y = torch.cartesian_prod(
                target_y.squeeze(-1),
                target_y.squeeze(-1),
            ).view(n, n, 2)
            # the diagonal of f_samps are the out-of-sample predictions
            # for each LOO model, compare the out of sample predictions to each in-sample prediction
            rank_loss = (
                    (f_samps.diagonal(dim1=1, dim2=2).unsqueeze(-1) < f_samps) ^
                    (cartesian_y[..., 0] < cartesian_y[..., 1])
            ).sum(dim=-1).sum(dim=-1)
        else:
            rank_loss = torch.zeros(f_samps.shape[0], dtype=torch.long, device=target_y.device)
            y_stack = target_y.squeeze(-1).expand(f_samps.shape)
            for i in range(1, target_y.shape[0]):
                rank_loss += (
                        (roll_col(f_samps, i) < f_samps) ^ (roll_col(y_stack, i) < y_stack)
                ).sum(dim=-1)
        return rank_loss

    @staticmethod
    def get_target_model_loocv_sample_preds(training_data: TrainingData, arrow_metrics_list, target_model, num_samples):
        batch_size = len(training_data.Xs[0])
        masks = torch.eye(len(training_data.Xs[0]), dtype=torch.uint8, device=training_data.Xs[0].device).bool()

        train_X_cv = [torch.stack([train_x[~m] for m in masks]) for train_x in training_data.Xs]
        train_Y_cv = [torch.stack([train_y[~m] for m in masks]) for train_y in training_data.Ys]
        train_Yvar_cv = [torch.stack([train_yvar[~m] for m in masks]) for train_yvar in training_data.Yvars]
        arrow_metrics_cv = [torch.stack([arrow_metrics[~m] for m in masks]) for arrow_metrics in arrow_metrics_list]

        state_dict = target_model.state_dict()
        # expand to batch size of batch_mode LOOCV model
        state_dict_expanded = {
            name: t.expand(batch_size, *[-1 for _ in range(t.ndim)])
            for name, t in state_dict.items()
        }
        model = get_fitted_karasu_model(train_X_cv, train_Y_cv, train_Yvar_cv, arrow_metrics_cv,
                                        ArrowModel if target_model.opt_class == "Arrow" else CherryPickModel,
                                        state_dict=state_dict_expanded, refit=False, normalize=True)
        with torch.no_grad():
            posterior = model.posterior(training_data.Xs[0])
            # Since we have a batch mode gp and model.posterior always returns an output dimension,
            # the output from `posterior.sample()` here `num_samples x n x n x 1`, so let's squeeze
            # the last dimension.
            sampler = SobolQMCNormalSampler(num_samples=num_samples)
            return sampler(posterior).squeeze(-1)

    @staticmethod
    def compute_rank_weights(training_data: TrainingData, arrow_metrics, base_models, target_model, num_samples):
        ranking_losses = [[] for _ in range(len(training_data.Ys))]

        def compute_and_append(samples):
            slices = [torch.index_select(samples, -1, idx) for idx in
                      torch.tensor(list(range(len(training_data.Ys))), device=samples.device)]
            for idx, (samples_slice, train_y) in enumerate(zip(slices, training_data.Ys)):
                samples_slice = torch.squeeze(samples_slice)
                # compute and save ranking loss
                ranking_losses[idx].append(RGPEModel1.compute_ranking_loss(samples_slice, train_y))

        def count_and_weight(tensor_list: List[torch.Tensor]):
            if not len(tensor_list):
                return torch.tensor([]).reshape(1, -1)

            t_tensor: torch.Tensor = torch.stack(tensor_list)
            # compute best model (minimum ranking loss) for each sample
            min_indices_tensor = torch.nonzero(t_tensor == torch.amin(t_tensor, dim=0, keepdim=True), as_tuple=False)
            min_indices_tensor = min_indices_tensor[torch.randperm(min_indices_tensor.shape[0])]
            min_indices_tensor = min_indices_tensor[min_indices_tensor[:, -1].sort()[1]]
            # select first occurrence of each "sample"
            # ref: (https://discuss.pytorch.org/t/first-occurrence-of-unique-values-in-a-tensor/81100/2)
            unique, inverse = torch.unique_consecutive(min_indices_tensor[:, -1], return_inverse=True)
            perm = torch.arange(inverse.size(0)).to(inverse)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
            # If the argmin is not unique, the weight is given to the target model if it is part of the tie,
            # otherwise the tie is broken randomly
            best_models = min_indices_tensor[perm][:, 0]
            if target_model is not None:
                # whenever target model is best, override previously randomly selected min-idx-value
                target_best = torch.argmin(t_tensor, dim=0)
                best_models[target_best == 0] = 0

            # compute proportion of samples for which each model is best
            proportions = best_models.bincount(minlength=t_tensor.shape[0]).type_as(t_tensor) / t_tensor.shape[1]
            # prevent weight dilution, according to RGPE-paper
            if target_model is not None:
                t_tensor = t_tensor.to(torch.double)
                # get 95th percentile of target model (target model is first row)
                target_95th_perc = torch.quantile(t_tensor[0, :], 0.95, dim=-1)
                # get median of all models
                all_50_th_perc = torch.quantile(t_tensor, 0.5, dim=-1)
                # we discard models where median > 95th percentile of target model
                proportions[all_50_th_perc > target_95th_perc] = 0
                # re-normalize
                proportions /= proportions.sum()
            return proportions.reshape(1, -1)

        if target_model is not None:
            # compute ranking loss for target model using LOOCV
            target_f_samps = RGPEModel1.get_target_model_loocv_sample_preds(training_data, arrow_metrics,
                                                                            target_model, num_samples)
            compute_and_append(target_f_samps)

        # compute ranking loss for each base model
        for task in range(len(base_models)):
            model = base_models[task]
            # compute posterior over training points for target task
            posterior = model.posterior(training_data.Xs[0])
            sampler = SobolQMCNormalSampler(num_samples=num_samples)
            base_f_samps = sampler(posterior).squeeze(-1)
            compute_and_append(base_f_samps)

        # numObjectives x numModels
        rank_weights = torch.cat([count_and_weight(rl) for rl in ranking_losses], dim=0).to(training_data.Xs[0])
        # numModels x numObjectives
        return rank_weights.transpose(0, 1)

    @classmethod
    def construct_inputs(cls, training_data: TrainingData, **kwargs):
        profiled_workloads: List[ProcessedWorkloadModel] = kwargs["profiled_workloads"]
        profiled_workloads = [w for w in profiled_workloads if not w.abandon]
        opt_class: str = kwargs["opt_class"]

        num_posterior_samples = kwargs["num_posterior_samples"]
        selection_strategy: AbstractSelectionStrategy = kwargs["selection_strategy"]

        prior_tasks, root_scaler = selection_strategy.get_prior_tasks(profiled_workloads)
        arrow_metrics, _ = get_arrow_metrics(profiled_workloads, root_scaler=root_scaler)
        arrow_metrics = arrow_metrics.to(training_data.Xs[0])
        arrow_metrics = [arrow_metrics] * len(training_data.Xs)

        task: Optional[CustomModelListGP] = None
        if len(profiled_workloads) >= 3:
            task = get_fitted_karasu_model(training_data.Xs, training_data.Ys, training_data.Yvars,
                                           arrow_metrics,
                                           ArrowModel if kwargs["opt_class"] == "Arrow" else CherryPickModel,
                                           state_dict=kwargs.get("state_dict", None),
                                           refit=kwargs.get("refit", True),
                                           normalize=kwargs.get("normalize", True))

        rank_weights: torch.Tensor = RGPEModel1.compute_rank_weights(
            training_data,
            arrow_metrics,
            prior_tasks,
            task,
            num_posterior_samples,
        )

        if torch.numel(rank_weights) == 0:
            raise ValueError("Base-model is undefined + no support models found! "
                             "Most likely reason: Emulated data repository is incomplete!")

        logging.info(f"RGPE model (opt_class={opt_class}) obtained with {len(profiled_workloads)} samples, "
                     f"{len(prior_tasks)} prior tasks... "
                     f"[{kwargs['task'].identifier}]")

        task_list = prior_tasks if task is None else prior_tasks + [task]
        return {"models": task_list, "weights": rank_weights, "target_task": task}

    def __init__(self, models: List[CustomModelListGP], weights: torch.Tensor,
                 target_task: Optional[CustomModelListGP]):
        super().__init__()
        self.models: List[CustomModelListGP] = models
        self.weights: torch.Tensor = weights
        self.target_task: Optional[CustomModelListGP] = target_task
        self.likelihood = LikelihoodList(*[m.likelihood for m in models])
        self.to(weights)

    def forward(self, x):

        weighted_means = []
        weighted_covars = []

        x = x.unsqueeze(-3) if x.ndim == 2 else x
        sample_count: int = x.size()[-2]

        # make local copy
        all_weights = torch.tensor(self.weights)
        # filter model with zero weights
        # weights on covariance matrices are weight**2
        all_weights[all_weights ** 2 == 0] = 0
        # re-normalize
        all_weights = all_weights / all_weights.sum(dim=0, keepdim=True)

        for model, weight_tensor in zip(self.models, all_weights):
            posterior = model.posterior(x)
            posterior_mean = posterior.mean
            posterior_cov = posterior.mvn.covariance_matrix

            for idx, (gp_model, weight) in enumerate(zip(model.custom_models, weight_tensor)):

                if weight.item() == 0:
                    continue

                mean_slice = idx
                cov_slice = torch.tensor(list(range(idx * sample_count, (idx + 1) * sample_count)))

                custom_y_std = gp_model.custom_y_std.squeeze(-1)
                custom_y_mean = gp_model.custom_y_mean.squeeze(-1)

                # MEAN: unstandardize predictions + apply weight + add to list
                tmp_posterior_mean = torch.zeros_like(posterior_mean)
                tmp_posterior_mean[..., mean_slice] = posterior_mean[..., mean_slice]
                tmp_posterior_mean[..., mean_slice] *= custom_y_std
                tmp_posterior_mean[..., mean_slice] += custom_y_mean
                tmp_posterior_mean[..., mean_slice] *= weight
                weighted_means.append(tmp_posterior_mean)

                # COV: unstandardize predictions + apply weight + add to list
                tmp_posterior_cov = torch.zeros_like(posterior_cov)
                tmp_posterior_cov[..., cov_slice] = posterior_cov[..., cov_slice]
                tmp_posterior_cov[..., cov_slice] *= torch.pow(custom_y_std, 2)
                tmp_posterior_cov[..., cov_slice] *= torch.pow(weight, 2)
                tmp_posterior_cov = tmp_posterior_cov.unsqueeze(-3)
                weighted_covars.append(lazify(tmp_posterior_cov))

        # set mean and covariance to be the rank-weighted sum the means and covariances of the
        # base models and target model
        mean_x = torch.stack(weighted_means).sum(dim=0)
        covar_x = PsdSumLazyTensor(*weighted_covars)
        covar_x = BlockDiagLazyTensor(covar_x, block_dim=-3)
        # try to squeeze first dimension, if possible
        mean_x = torch.atleast_2d(mean_x.squeeze(0))
        covar_x = covar_x.squeeze(0)
        return MultitaskMultivariateNormal(mean_x, covar_x, interleaved=False)


class RGPEModel2(RGPEModel1):
    _num_outputs = 3  # metadata for botorch


task_resolver: Dict[int, tuple] = {
    1: (TASKModel1, CustomqNoisyExpectedImprovement, "SOO"),
    2: (TASKModel2, CustomqNoisyExpectedHypervolumeImprovement, "MOO")
}

rgpe_resolver: Dict[int, tuple] = {
    1: (RGPEModel1, CustomqNoisyExpectedImprovement, "SOO"),
    2: (RGPEModel2, CustomqNoisyExpectedHypervolumeImprovement, "MOO")
}
