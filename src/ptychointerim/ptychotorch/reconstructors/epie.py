from typing import Optional

import torch
import tqdm
from torch.utils.data import Dataset

import ptychointerim.ptychotorch.propagation as prop
from ptychointerim.ptychotorch.data_structures import Ptychography2DVariableGroup
from ptychointerim.ptychotorch.reconstructors.base import AnalyticalIterativeReconstructor
from ptychointerim.image_proc import place_patches_fourier_shift
from ptychointerim.position_correction import compute_positions_cross_correlation_update


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

    def check_inputs(self, *args, **kwargs):
        for var in self.variable_group.get_optimizable_variables():
            assert 'lr' in var.optimizer_params.keys(), \
                "Optimizable variable {} must have 'lr' in optimizer_params.".format(var.name)
        if self.metric_function is not None:
            raise NotImplementedError('EPIEReconstructor does not support metric function yet.')
        if self.variable_group.probe.has_multiple_opr_modes:
            raise NotImplementedError('EPIEReconstructor does not support multiple OPR modes yet.')
        
    def run_minibatch(self, input_data, y_true, *args, **kwargs):
        (delta_o, delta_p, delta_pos), batch_loss = self.compute_updates(*input_data, y_true, self.dataset.valid_pixel_mask)
        self.apply_updates(delta_o, delta_p, delta_pos)
        batch_loss = torch.mean(batch_loss)
        self.loss_tracker.update_batch_loss_with_value(batch_loss.item())

    @staticmethod
    def compute_updates(update_step_module: torch.nn.Module,
                        indices: torch.Tensor,
                        y_true: torch.Tensor,
                        valid_pixel_mask: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
        """
        Calculates the updates of the whole object, the probe, and other variables.
        This function is called in self.update_step_module.forward. 
        """
        object_ = update_step_module.variable_module_dict['object']
        probe = update_step_module.variable_module_dict['probe']
        probe_positions = update_step_module.variable_module_dict['probe_positions']

        indices = indices.cpu()
        positions = probe_positions.tensor[indices]

        y = 0.0
        obj_patches = object_.extract_patches(
            positions, probe.get_spatial_shape()
        )

        p = torch.zeros_like(probe.get_opr_mode(0))
        psi_array_shape = (len(obj_patches), probe.n_modes, *obj_patches.shape[1:])
        psi = torch.zeros(psi_array_shape, dtype=obj_patches.dtype)
        psi_far = torch.zeros(psi_array_shape, dtype=obj_patches.dtype)
        psi_prime = torch.zeros(psi_array_shape, dtype=obj_patches.dtype)

        I_total = (torch.abs(probe.get_opr_mode(0)) ** 2).sum()
        def compute_psi_variables(p, y):
            psi = obj_patches * p
            psi_far = prop.propagate_far_field(psi)
            y = y + torch.abs(psi_far) ** 2

            # Scaling factor to account for distribution of power among probe modes.
            A = ( (torch.abs(p) ** 2).sum() / I_total ) ** 0.5
            psi_prime = psi_far / torch.abs(psi_far) * torch.sqrt(y_true + 1e-7) * A
            # Do not swap magnitude for bad pixels.
            psi_prime = torch.where(valid_pixel_mask.repeat(psi_prime.shape[0], 1, 1), psi_prime, psi_far)
            psi_prime = prop.back_propagate_far_field(psi_prime)

            return psi, psi_far, psi_prime, y
        
        for mode in range(probe.n_modes):
            p[mode] = probe.get_mode_and_opr_mode(mode, 0)
            psi[:, mode], psi_far[:, mode], psi_prime[:, mode], y = compute_psi_variables(p[mode], y)

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
            
        # DataParallel would split the real and imaginary parts of delta_o
        # and store them in an additional dimension at the end. To keep things consistent,
        # we do the splitting manually for cases without DataParallel. 
        # Also, add a new dimension in the front for DataParallel to concantenate multi-device outputs.
        delta_o, delta_p_all_modes = update_step_module.process_updates(delta_o, delta_p_all_modes)

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
            delta_o = delta_o[..., 0] + 1j * delta_o[..., 1]
            delta_o = delta_o.sum(0)
            object_.set_grad(-delta_o)
            object_.optimizer.step()
            
        if delta_p is not None:
            delta_p = delta_p[..., 0] + 1j * delta_p[..., 1]
            delta_p = delta_p.mean(0)
            probe.set_grad(-delta_p)
            probe.optimizer.step()

        if delta_pos is not None:
            probe_positions.set_grad(-delta_pos)
            probe_positions.optimizer.step()