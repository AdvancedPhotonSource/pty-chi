from typing import Type, Optional

import torch
import tqdm
from torch.utils.data import Dataset

from ptychointerim.ptychotorch.data_structures import VariableGroup
from ptychointerim.ptychotorch.forward_models import ForwardModel
from ptychointerim.ptychotorch.reconstructors.base import IterativeReconstructor, LossTracker


class AutodiffReconstructor(IterativeReconstructor):

    def __init__(self,
                 variable_group: VariableGroup,
                 dataset: Dataset,
                 forward_model_class: Type[ForwardModel],
                 forward_model_params: Optional[dict] = None,
                 batch_size: int = 1,
                 loss_function: torch.nn.Module = None,
                 n_epochs: int = 100,
                 metric_function: Optional[torch.nn.Module] = None,
                 *args, **kwargs
    ) -> None:
        super().__init__(
            variable_group=variable_group,
            dataset=dataset,
            batch_size=batch_size,
            n_epochs=n_epochs,
            metric_function=metric_function,
            *args, **kwargs
        )
        self.forward_model_class = forward_model_class
        self.forward_model_params = forward_model_params if forward_model_params is not None else {}
        self.forward_model = None
        self.loss_function = loss_function

        self.check_inputs()

    def check_inputs(self, *args, **kwargs):
        for var in self.variable_group.get_optimizable_variables():
            assert var.optimizer is not None, \
                "Variable {} is optimizable but no optimizer is specified".format(var.name)

    def build(self) -> None:
        super().build()
        self.build_forward_model()
        
    def build_loss_tracker(self):
        f = self.loss_function if self.metric_function is None else self.metric_function
        # LossTracker should always compute the loss using data if metric function and loss function
        # are not the same type.
        always_compute_loss = (self.metric_function is not None) and (type(self.metric_function) != type(self.loss_function))
        self.loss_tracker = LossTracker(metric_function=f, always_compute_loss=always_compute_loss)

    def build_forward_model(self):
        self.forward_model = self.forward_model_class(self.variable_group, **self.forward_model_params)
        if not torch.get_default_device().type == 'cpu':
            self.forward_model = torch.nn.DataParallel(self.forward_model)
            self.forward_model.to(torch.get_default_device())

    def run(self, *args, **kwargs):
        for i_epoch in tqdm.trange(self.n_epochs):
            for batch_data in self.dataloader:
                input_data = [x.to(torch.get_default_device()) for x in batch_data[:-1]]
                y_true = batch_data[-1].to(torch.get_default_device())

                y_pred = self.forward_model(*input_data)
                batch_loss = self.loss_function(y_pred, y_true)

                batch_loss.backward()
                self.get_forward_model().post_differentiation_hook(*input_data, y_true)
                self.step_all_optimizers()
                self.forward_model.zero_grad()
                self.run_post_update_hooks()

                self.loss_tracker.update_batch_loss(y_pred=y_pred, y_true=y_true, loss=batch_loss.item())
            self.loss_tracker.conclude_epoch(epoch=i_epoch)
            self.loss_tracker.print_latest()

    def step_all_optimizers(self):
        for var in self.variable_group.get_optimizable_variables():
            var.optimizer.step()

    def get_forward_model(self) -> ForwardModel:
        if isinstance(self.forward_model, torch.nn.DataParallel):
            return self.forward_model.module
        else:
            return self.forward_model

    def get_config_dict(self) -> dict:
        d = super().get_config_dict()
        d.update({'forward_model_class': str(self.forward_model_class),
                  'loss_function': str(self.loss_function)})
        return d
    