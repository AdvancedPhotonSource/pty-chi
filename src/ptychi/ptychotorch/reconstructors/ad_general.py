from typing import Optional, TYPE_CHECKING

import torch
from torch.utils.data import Dataset


import ptychi.forward_models as fm
from ptychi.ptychotorch.reconstructors.base import IterativeReconstructor, LossTracker
import ptychi.maps as maps
if TYPE_CHECKING:
    import ptychi.data_structures.parameter_group as pg
    import ptychi.api as api


class AutodiffReconstructor(IterativeReconstructor):
    def __init__(
        self,
        parameter_group: "pg.ParameterGroup",
        dataset: Dataset,
        options: Optional["api.options.ad_general.AutodiffReconstructorOptions"] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            parameter_group=parameter_group,
            dataset=dataset,
            options=options,
            *args,
            **kwargs,
        )
        self.forward_model_class = options.forward_model_class
        self.forward_model_class = maps.get_forward_model_by_enum(self.forward_model_class)
        self.forward_model_params = options.forward_model_params if options.forward_model_params is not None else {}
        self.forward_model = None
        self.loss_function = maps.get_loss_function_by_enum(options.loss_function)()

    def build(self) -> None:
        super().build()
        self.build_forward_model()

    def build_loss_tracker(self):
        f = self.loss_function if self.metric_function is None else self.metric_function
        # LossTracker should always compute the loss using data if metric function and loss function
        # are not the same type.
        always_compute_loss = (self.metric_function is not None) and (
            type(self.metric_function) is not type(self.loss_function)
        )
        self.loss_tracker = LossTracker(metric_function=f, always_compute_loss=always_compute_loss)

    def build_forward_model(self):
        self.forward_model = self.forward_model_class(
            self.parameter_group, **self.forward_model_params
        )
        if not torch.get_default_device().type == "cpu":
            self.forward_model = torch.nn.DataParallel(self.forward_model)
            self.forward_model.to(torch.get_default_device())

    def run_post_differentiation_hooks(self, input_data, y_true):
        self.get_forward_model().post_differentiation_hook(*input_data, y_true)
        self.apply_regularizers()

    def apply_regularizers(self) -> None:
        """
        Apply Tikonov regularizers.
        """
        pass

    def run_minibatch(self, input_data, y_true, *args, **kwargs):
        y_pred = self.forward_model(*input_data)
        batch_loss = self.loss_function(y_pred, y_true)

        batch_loss.backward()
        self.run_post_differentiation_hooks(input_data, y_true)
        self.step_all_optimizers()
        self.forward_model.zero_grad()
        self.run_post_update_hooks()

        self.loss_tracker.update_batch_loss(y_pred=y_pred, y_true=y_true, loss=batch_loss.item())

    def step_all_optimizers(self):
        for var in self.parameter_group.get_optimizable_parameters():
            if var.optimization_enabled(self.current_epoch):
                var.optimizer.step()

    def get_forward_model(self) -> "fm.ForwardModel":
        if isinstance(self.forward_model, torch.nn.DataParallel):
            return self.forward_model.module
        else:
            return self.forward_model

    def get_config_dict(self) -> dict:
        d = super().get_config_dict()
        d.update(
            {
                "forward_model_class": str(self.forward_model_class),
                "loss_function": str(self.loss_function),
            }
        )
        return d
