from typing import Optional

import torch
from torch.utils.data import Dataset
from torch import Tensor

import ptychointerim.ptychotorch.data_structures as ds
from ptychointerim.ptychotorch.reconstructors.base import (
    AnalyticalIterativePtychographyReconstructor,
)
from ptychointerim.image_proc import place_patches_fourier_shift
import ptychointerim.api as api


class PIEReconstructor(AnalyticalIterativePtychographyReconstructor):
    """
    The ptychographic iterative engine (PIE), as described in:

    Andrew Maiden, Daniel Johnson, and Peng Li, "Further improvements to the
    ptychographical iterative engine," Optica 4, 736-745 (2017)

    Object and probe updates are calculated using the formulas in table 1 of
    Maiden (2017).

    The `step_size` parameter is equivalent to gamma in Eq. 22 of Maiden (2017)
    when `optimizer == SGD`.
    """

    def __init__(
        self,
        parameter_group: "ds.Ptychography2DParameterGroup",
        dataset: Dataset,
        options: Optional["api.options.pie.PIEReconstructorOptions"] = None,
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
        if not isinstance(self.parameter_group.object, ds.Object2D):
            raise NotImplementedError("EPIEReconstructor only supports 2D objects.")
        for var in self.parameter_group.get_optimizable_parameters():
            if "lr" not in var.optimizer_params.keys():
                raise ValueError(
                    "Optimizable parameter {} must have 'lr' in optimizer_params.".format(var.name)
                )
        if self.metric_function is not None:
            raise NotImplementedError("EPIEReconstructor does not support metric function yet.")
        if self.parameter_group.probe.has_multiple_opr_modes:
            raise NotImplementedError("EPIEReconstructor does not support multiple OPR modes yet.")

    def run_minibatch(self, input_data, y_true, *args, **kwargs):
        (delta_o, delta_p, delta_pos), batch_loss = self.compute_updates(
            *input_data, y_true, self.dataset.valid_pixel_mask
        )
        self.apply_updates(delta_o, delta_p, delta_pos)
        batch_loss = torch.mean(batch_loss)
        self.loss_tracker.update_batch_loss_with_value(batch_loss.item())

    def compute_updates(
        self, indices: torch.Tensor, y_true: torch.Tensor, valid_pixel_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """
        Calculates the updates of the whole object, the probe, and other parameters.
        This function is called in self.update_step_module.forward.
        """
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        probe_positions = self.parameter_group.probe_positions

        indices = indices.cpu()
        positions = probe_positions.tensor[indices]

        y, obj_patches = self.forward_model.forward(indices, return_object_patches=True)
        psi = self.forward_model.intermediate_variables["psi"]
        psi_far = self.forward_model.intermediate_variables["psi_far"]

        p = probe.get_opr_mode(0)

        psi_prime = (
            psi_far
            / ((psi_far.abs() ** 2).sum(1, keepdims=True).sqrt() + 1e-7)
            * torch.sqrt(y_true + 1e-7)[:, None]
        )
        # Do not swap magnitude for bad pixels.
        psi_prime = torch.where(
            valid_pixel_mask.repeat(psi_prime.shape[0], probe.n_modes, 1, 1), psi_prime, psi_far
        )
        psi_prime = self.forward_model.far_field_propagator.propagate_backward(psi_prime)

        delta_o = None
        if object_.optimization_enabled(self.current_epoch):
            step_weight = self.calculate_object_step_weight(p)
            delta_o_patches = step_weight * (psi_prime - psi)
            delta_o_patches = delta_o_patches.sum(1)
            delta_o = place_patches_fourier_shift(
                torch.zeros_like(object_.data),
                positions + object_.center_pixel,
                delta_o_patches,
                op="add",
            )

        delta_pos = None
        if probe_positions.optimization_enabled(self.current_epoch) and object_.optimizable:
            delta_pos = torch.zeros_like(probe_positions.data)
            delta_pos[indices] = probe_positions.position_correction.get_update(
                psi_prime - psi,
                obj_patches,
                delta_o_patches,
                probe.get_mode_and_opr_mode(0, 0),
                object_.optimizer_params["lr"],
            )

        delta_p_all_modes = None
        if probe.optimization_enabled(self.current_epoch):
            step_weight = self.calculate_probe_step_weight(obj_patches)
            delta_p = step_weight * (psi_prime - psi)
            delta_p = delta_p.mean(0)
            delta_p_all_modes = delta_p[None, :, :]

        batch_loss = torch.mean((torch.sqrt(y) - torch.sqrt(y_true)) ** 2)
        return (delta_o, delta_p_all_modes, delta_pos), torch.atleast_1d(batch_loss)

    def calculate_object_step_weight(self, p: Tensor):
        """Calculate the weight for the object update step."""
        numerator = p.abs() * p.conj()
        denominator = p.abs().sum(0).max() * (
            p.abs() ** 2 + self.parameter_group.object.options.alpha * (p.abs() ** 2).sum(0).max()
        )
        step_weight = numerator / denominator
        return step_weight

    def calculate_probe_step_weight(self, obj_patches: Tensor):
        """Calculate the weight for the probe update step."""
        obj_max = (torch.abs(obj_patches) ** 2).max(-1).values.max(-1).values.view(-1, 1, 1)
        numerator = obj_patches.abs() * obj_patches.conj()
        denominator = obj_max * (
            obj_patches.abs() ** 2 + self.parameter_group.probe.options.alpha * obj_max
        )
        step_weight = numerator / denominator
        return step_weight[:, None]

    def apply_updates(self, delta_o, delta_p, delta_pos, *args, **kwargs):
        """
        Apply updates to optimizable parameters given the updates calculated by self.compute_updates.

        Parameters
        ----------
        delta_o : Tensor
            A (n_replica, h, w, 2) tensor of object update vector.
        delta_p : Tensor
            A (n_replicate, n_opr_modes, n_modes, h, w, 2) tensor of probe update vector.
        delta_pos : Tensor
            A (n_positions, 2) tensor of probe position vectors.
        """
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        probe_positions = self.parameter_group.probe_positions

        if delta_o is not None:
            object_.set_grad(-delta_o)
            object_.optimizer.step()

        if delta_p is not None:
            probe.set_grad(-delta_p)
            probe.optimizer.step()

        if delta_pos is not None:
            probe_positions.set_grad(-delta_pos)
            probe_positions.optimizer.step()


class EPIEReconstructor(PIEReconstructor):
    """
    The extended ptychographic iterative engine (ePIE), as described in:

    Andrew Maiden, Daniel Johnson, and Peng Li, "Further improvements to the
    ptychographical iterative engine," Optica 4, 736-745 (2017)

    Object and probe updates are calculated using the formulas in table 1 of
    Maiden (2017).

    The `step_size` parameter is equivalent to gamma in Eq. 22 of Maiden (2017)
    when `optimizer == SGD`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_object_step_weight(self, p: Tensor):
        p_max = (torch.abs(p) ** 2).sum(0).max()
        step_weight = self.parameter_group.object.options.alpha * p.conj() / p_max
        return step_weight

    def calculate_probe_step_weight(self, obj_patches: Tensor):
        obj_max = (torch.abs(obj_patches) ** 2).max(-1).values.max(-1).values.view(-1, 1, 1)
        step_weight = self.parameter_group.probe.options.alpha * obj_patches.conj() / obj_max
        step_weight = step_weight[:, None]
        return step_weight


class RPIEReconstructor(PIEReconstructor):
    """
    The regularized ptychographic iterative engine (rPIE), as described in:

    Andrew Maiden, Daniel Johnson, and Peng Li, "Further improvements to the
    ptychographical iterative engine," Optica 4, 736-745 (2017)

    Object and probe updates are calculated using the formulas in table 1 of
    Maiden (2017).

    The `step_size` parameter is equivalent to gamma in Eq. 22 of Maiden (2017)
    when `optimizer == SGD`.

    To get the momentum-accelerated PIE (mPIE), use `optimizer == SGD` and use
    the optimizer settings `{'momentum': eta, 'nesterov': True}` where `eta` is
    the constant used in  Eq. 19 of Maiden (2017).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_object_step_weight(self, p: Tensor):
        p_max = (torch.abs(p) ** 2).sum(0).max()
        step_weight = p.conj() / (
            (1 - self.parameter_group.object.options.alpha) * (torch.abs(p) ** 2)
            + self.parameter_group.object.options.alpha * p_max
        )
        return step_weight

    def calculate_probe_step_weight(self, obj_patches: Tensor):
        obj_max = (torch.abs(obj_patches) ** 2).max(-1).values.max(-1).values.view(-1, 1, 1)
        step_weight = obj_patches.conj() / (
            (1 - self.parameter_group.probe.options.alpha) * (torch.abs(obj_patches) ** 2)
            + self.parameter_group.probe.options.alpha * obj_max
        )
        step_weight = step_weight[:, None]
        return step_weight
