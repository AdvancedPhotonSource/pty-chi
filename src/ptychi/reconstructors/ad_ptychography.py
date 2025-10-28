# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Optional, TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from ptychi.reconstructors.ad_general import AutodiffReconstructor
from ptychi.reconstructors.base import IterativePtychographyReconstructor
import ptychi.metrics as metrics

if TYPE_CHECKING:
    import ptychi.data_structures.parameter_group as pg
    import ptychi.api as api


class AutodiffPtychographyReconstructor(AutodiffReconstructor, IterativePtychographyReconstructor):
    def __init__(
        self,
        parameter_group: "pg.PtychographyParameterGroup",
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
    
    def build(self):
        self.update_requires_grad()
        super().build()
        
    def build_forward_model(self):
        self.forward_model_params["wavelength_m"] = self.dataset.wavelength_m
        self.forward_model_params["detector_size"] = tuple(self.dataset.patterns.shape[-2:])
        self.forward_model_params["free_space_propagation_distance_m"] = self.dataset.free_space_propagation_distance_m
        self.forward_model_params["pad_for_shift"] = self.options.forward_model_options.pad_for_shift
        self.forward_model_params["low_memory_mode"] = self.options.forward_model_options.low_memory_mode
        return super().build_forward_model()
    
    def run_pre_epoch_hooks(self) -> None:
        super().run_pre_epoch_hooks()
        self.update_requires_grad()
                    
    def update_requires_grad(self):
        """Set requires_grad of variables that don't need to be optimized at this epoch to False.
        This prevents these variables from holding the old graph and causing errors during
        backward() call.
        """
        for var in self.parameter_group.get_all_parameters():
            if not (var.options.optimization_plan.is_enabled(self.current_epoch) and var.options.optimizable):
                var.tensor.requires_grad = False
            else:
                var.tensor.requires_grad = True
            for subvar in var.sub_modules:
                if not (subvar.options.optimization_plan.is_enabled(self.current_epoch) and subvar.options.optimizable):
                    subvar.tensor.requires_grad = False
                else:
                    subvar.tensor.requires_grad = True

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
        reg = super().apply_regularizers()

        object_ = self.parameter_group.object
        if object_.options.l1_norm_constraint.is_enabled_on_this_epoch(self.current_epoch):
            # object.data returns a copy, so we directly access the tensor here.
            obj_data = object_.tensor.data[..., 0] + 1j * object_.tensor.data[..., 1]
            reg = reg + object_.options.l1_norm_constraint.weight * torch.mean(obj_data.abs())
            
        if object_.options.l2_norm_constraint.is_enabled_on_this_epoch(self.current_epoch):
            obj_data = object_.tensor.data[..., 0] + 1j * object_.tensor.data[..., 1]
            reg = reg + object_.options.l2_norm_constraint.weight * torch.mean(obj_data.abs() ** 2)
            
        if object_.options.total_variation.is_enabled_on_this_epoch(self.current_epoch):
            obj_data = object_.tensor.data[..., 0] + 1j * object_.tensor.data[..., 1]
            reg = reg + object_.options.total_variation.weight * metrics.TotalVariationLoss(reduction="mean")(obj_data)
            
        if reg.requires_grad:
            reg.backward()
        return reg
    
    def get_retain_graph(self) -> bool:
        """Determines whether `loss.backward()` should have `retain_graph=True`.
        Since regularizers' gradients are computed after `loss.backward()`, we need to
        return `True` if any regularizer is enabled.
        """
        if self.parameter_group.object.options.experimental.deep_image_prior_options.enabled:
            return True
        if self.parameter_group.probe.options.experimental.deep_image_prior_options.enabled:
            return True
        if (
            self.parameter_group.object.options.l1_norm_constraint.is_enabled_on_this_epoch(self.current_epoch)
            and self.parameter_group.object.options.l2_norm_constraint.is_enabled_on_this_epoch(self.current_epoch)
            and self.parameter_group.object.options.total_variation.is_enabled_on_this_epoch(self.current_epoch)
        ):
            return True
        return False

    def run_minibatch(self, input_data, y_true, *args, **kwargs):        
        y_pred = self.forward_model(*input_data)
        batch_loss = self.loss_function(
            y_pred[:, self.dataset.valid_pixel_mask], y_true[:, self.dataset.valid_pixel_mask]
        )

        batch_loss.backward(retain_graph=self.get_retain_graph())
        self.run_post_differentiation_hooks(input_data, y_true)
        reg_loss = self.apply_regularizers()
        
        self.step_all_optimizers(forward_model_args=input_data, y_true=y_true)
        self.forward_model.zero_grad()
        self.run_post_update_hooks()
        
        self.loss_tracker.update_batch_loss(y_pred=y_pred, y_true=y_true, loss=batch_loss.item())
        self.loss_tracker.update_batch_regularization_loss(reg_loss.item())
