from typing import Optional

import torch
from torch.utils.data import Dataset

import ptychointerim.ptychotorch.data_structures as ds
import ptychointerim.forward_models as fm
from ptychointerim.ptychotorch.reconstructors.ad_general import AutodiffReconstructor
from ptychointerim.ptychotorch.reconstructors.base import IterativePtychographyReconstructor
import ptychointerim.api as api


class AutodiffPtychographyReconstructor(AutodiffReconstructor, IterativePtychographyReconstructor):
    def __init__(
        self,
        parameter_group: "ds.PtychographyParameterGroup",
        dataset: Dataset,
        options: Optional["api.options.ad_ptychography.AutodiffPtychographyOptions"] = None,
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

    def check_inputs(self, *args, **kwargs):
        super().check_inputs(*args, **kwargs)
        if type(self.parameter_group.object) is ds.MultisliceObject:
            if self.forward_model_class != fm.MultislicePtychographyForwardModel:
                raise ValueError(
                    "If the object is multislice, the forward model must be MultislicePtychographyForwardModel."
                )
        if type(self.parameter_group.object) is ds.Object2D:
            if self.forward_model_class != fm.Ptychography2DForwardModel:
                raise ValueError(
                    "If the object is 2D, the forward model must be Ptychography2DForwardModel."
                )
                
    def build_forward_model(self):
        if self.forward_model_class is fm.MultislicePtychographyForwardModel:
            if 'wavelength_m' not in self.forward_model_params.keys():
                self.forward_model_params['wavelength_m'] = self.dataset.wavelength_m
        return super().build_forward_model()

    def run_post_differentiation_hooks(self, input_data, y_true):
        super().run_post_differentiation_hooks(input_data, y_true)

        # If OPRModeWeights is optimized in the current epoch (i.e., it has gradient) but intensity variation
        # optimization is not enabled, set the gradient of the principal mode's weights to 0. Similar is done
        # for the gradient of the eigenmode weights.
        if self.parameter_group.opr_mode_weights.optimization_enabled(self.current_epoch):
            if not self.parameter_group.opr_mode_weights.intensity_variation_optimization_enabled(
                self.current_epoch
            ):
                self.parameter_group.opr_mode_weights.tensor.grad[:, 0] = 0
            if not self.parameter_group.opr_mode_weights.eigenmode_weight_optimization_enabled(
                self.current_epoch
            ):
                self.parameter_group.opr_mode_weights.tensor.grad[:, 1:] = 0

    def run_post_update_hooks(self) -> None:
        with torch.no_grad():
            if self.parameter_group.object.optimization_enabled(self.current_epoch):
                self.parameter_group.object.post_update_hook()

            if self.parameter_group.probe.optimization_enabled(self.current_epoch):
                self.parameter_group.probe.post_update_hook()

            if self.parameter_group.probe_positions.optimization_enabled(self.current_epoch):
                self.parameter_group.probe_positions.post_update_hook()

    def apply_regularizers(self) -> None:
        """
        Apply Tikonov regularizers, e.g., L1 norm for the object.

        This function calculates the regularization terms and backpropagates them. Since the gradients
        of the parameters haven't been zeroed, the regularizers' gradients will be added to them.
        """
        super().apply_regularizers()

        object_ = self.parameter_group.object
        if object_.l1_norm_constraint_enabled(self.current_epoch):
            # object.data returns a copy, so we directly access the tensor here.
            obj_data = object_.tensor.data[..., 0] + 1j * object_.tensor.data[..., 1]
            reg = object_.l1_norm_constraint_weight * obj_data.abs().sum()
            reg.backward()

    def run_minibatch(self, input_data, y_true, *args, **kwargs):
        y_pred = self.forward_model(*input_data)
        batch_loss = self.loss_function(
            y_pred[:, self.dataset.valid_pixel_mask], y_true[:, self.dataset.valid_pixel_mask]
        )

        batch_loss.backward()
        self.run_post_differentiation_hooks(input_data, y_true)
        self.step_all_optimizers()
        self.forward_model.zero_grad()
        self.run_post_update_hooks()
        self.loss_tracker.update_batch_loss(y_pred=y_pred, y_true=y_true, loss=batch_loss.item())
