from typing import Optional

import torch
import tqdm
from torch.utils.data import Dataset

import ptychointerim.ptychotorch.propagation as prop
from ptychointerim.ptychotorch.data_structures import Ptychography2DVariableGroup
from ptychointerim.ptychotorch.reconstructors.base import AnalyticalIterativeReconstructor
from ptychointerim.image_proc import place_patches_fourier_shift
from ptychointerim.position_correction import compute_positions_cross_correlation_update
from ptychointerim.forward_models import Ptychography2DForwardModel


class EPIEReconstructor(AnalyticalIterativeReconstructor):

    def __init__(self,
                 variable_group: Ptychography2DVariableGroup,
                 dataset: Dataset,
                 batch_size: int = 1,
                 n_epochs: int = 100,
                 *args, **kwargs
    ) -> None:
        super().__init__(
            variable_group=variable_group,
            dataset=dataset,
            batch_size=batch_size,
            n_epochs=n_epochs,
            *args, **kwargs)
        self.forward_model = Ptychography2DForwardModel(variable_group, retain_intermediates=True)

    def check_inputs(self, *args, **kwargs):
        for var in self.variable_group.get_optimizable_variables():
            if 'lr' not in var.optimizer_params.keys():
                raise ValueError("Optimizable variable {} must have 'lr' in optimizer_params.".format(var.name))
        if self.metric_function is not None:
            raise NotImplementedError('EPIEReconstructor does not support metric function yet.')
        if self.variable_group.probe.has_multiple_opr_modes:
            raise NotImplementedError('EPIEReconstructor does not support multiple OPR modes yet.')

    def run_post_update_hooks(self) -> None:
        self.variable_group.probe.post_update_hook()
        
    def run_minibatch(self, input_data, y_true, *args, **kwargs):
        (delta_o, delta_p, delta_pos), batch_loss = self.compute_updates(
            *input_data, y_true, self.dataset.valid_pixel_mask)
        self.apply_updates(delta_o, delta_p, delta_pos)
        batch_loss = torch.mean(batch_loss)
        self.loss_tracker.update_batch_loss_with_value(batch_loss.item())

    def compute_updates(self,
                        indices: torch.Tensor,
                        y_true: torch.Tensor,
                        valid_pixel_mask: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
        """
        Calculates the updates of the whole object, the probe, and other variables.
        This function is called in self.update_step_module.forward. 
        """
        object_ = self.variable_group.object
        probe = self.variable_group.probe
        probe_positions = self.variable_group.probe_positions

        indices = indices.cpu()
        positions = probe_positions.tensor[indices]

        y, obj_patches = self.forward_model.forward(indices, return_object_patches=True)
        psi = self.forward_model.intermediate_variables['psi']
        psi_far = self.forward_model.intermediate_variables['psi_far']

        # Get psi_prime
        p = probe.get_opr_mode(0)
        I_total = (torch.abs(probe.get_opr_mode(0)) ** 2).sum()
        A = ( (torch.abs(p) ** 2).sum((1, 2)) / I_total ) ** 0.5
        psi_prime = psi_far / torch.abs(psi_far) * torch.sqrt(y_true + 1e-7)[:, None] * A[None, :, None, None]
        # Do not swap magnitude for bad pixels.
        psi_prime = torch.where(valid_pixel_mask.repeat(psi_prime.shape[0], probe.n_modes, 1, 1), psi_prime, psi_far)
        psi_prime = prop.back_propagate_far_field(psi_prime)

        delta_o = None
        if object_.optimizable:
            delta_o_patches = p.conj() / (torch.abs(p) ** 2).sum(0).max()
            delta_o_patches = delta_o_patches * (psi_prime - psi)
            delta_o_patches = delta_o_patches.sum(axis=1)
            delta_o = place_patches_fourier_shift(torch.zeros_like(object_.data), positions + object_.center_pixel, delta_o_patches, op='add')

        delta_pos = None
        if probe_positions.optimizable and object_.optimizable:
            updated_obj_patches = obj_patches + delta_o_patches * object_.optimizer_params['lr']
            delta_pos = compute_positions_cross_correlation_update(
                obj_patches, updated_obj_patches, indices, probe_positions.data, probe.data[0, 0])

        delta_p_all_modes = None
        if probe.optimizable:
            delta_p = obj_patches.conj() / (torch.abs(obj_patches) ** 2).max(-1).values.max(-1).values.view(-1, 1, 1)
            delta_p = delta_p[:, None] * (psi_prime - psi)
            delta_p = delta_p.mean(0)
            delta_p_all_modes = delta_p[None, :, :]

        batch_loss = torch.mean((torch.sqrt(y) - torch.sqrt(y_true)) ** 2)
        return (delta_o, delta_p_all_modes, delta_pos), torch.atleast_1d(batch_loss)

    def apply_updates(self, delta_o, delta_p, delta_pos, *args, **kwargs):
        """
        Apply updates to optimizable parameters given the updates calculated by self.compute_updates.

        :param delta_o: A (n_replica, h, w, 2) tensor of object update vector.
        :param delta_p: A (n_replicate, n_opr_modes, n_modes, h, w, 2) tensor of probe update vector.
        :param delta_pos: A (n_positions, 2) tensor of probe position vectors.
        """
        object_ = self.variable_group.object
        probe = self.variable_group.probe
        probe_positions = self.variable_group.probe_positions

        if delta_o is not None:
            object_.set_grad(-delta_o)
            object_.optimizer.step()
            
        if delta_p is not None:
            probe.set_grad(-delta_p)
            probe.optimizer.step()

        if delta_pos is not None:
            probe_positions.set_grad(-delta_pos)
            probe_positions.optimizer.step()