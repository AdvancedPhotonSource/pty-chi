import torch
import tqdm
from torch.utils.data import Dataset

import ptychointerim.ptychotorch.propagation as prop
from ptychointerim.ptychotorch.data_structures import Ptychography2DVariableGroup
from ptychointerim.ptychotorch.reconstructors.base import AnalyticalIterativeReconstructor


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
        self.check_inputs()

    def check_inputs(self, *args, **kwargs):
        for var in self.variable_group.get_optimizable_variables():
            assert 'lr' in var.optimizer_params.keys(), \
                "Optimizable variable {} must have 'lr' in optimizer_params.".format(var.name)
        if self.metric_function is not None:
            raise NotImplementedError('EPIEReconstructor does not support metric function yet.')

    def run(self, *args, **kwargs):
        torch.no_grad()
        for i_epoch in tqdm.trange(self.n_epochs):
            for batch_data in self.dataloader:
                input_data = [x.to(torch.get_default_device()) for x in batch_data[:-1]]
                y_true = batch_data[-1].to(torch.get_default_device())

                (delta_o, delta_p), batch_loss = self.update_step_module(*input_data, y_true)
                self.apply_updates(input_data[0], delta_o, delta_p)
                batch_loss = torch.mean(batch_loss)

                self.loss_tracker.update_batch_loss_with_value(batch_loss.item())
            self.loss_tracker.conclude_epoch(epoch=i_epoch)
            self.loss_tracker.print_latest()

    @staticmethod
    def compute_updates(update_step_module: torch.nn.Module,
                        indices: torch.Tensor,
                        y_true: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
        object_ = update_step_module.variable_module_dict['object']
        probe = update_step_module.variable_module_dict['probe']
        probe_positions = update_step_module.variable_module_dict['probe_positions']

        assert probe.n_modes == 1, "This ePIE implementation only works with a single probe mode."

        indices = indices.cpu()
        positions = probe_positions.tensor[indices]

        y = 0.0
        obj_patches = object_.extract_patches(
            positions, probe.get_spatial_shape()
        )
        p = probe.get_mode(0)
        psi = obj_patches * p
        psi_far = prop.propagate_far_field(psi)
        y = y + torch.abs(psi_far) ** 2

        psi_prime = psi_far / torch.abs(psi_far) * torch.sqrt(y_true + 1e-7)
        psi_prime = prop.back_propagate_far_field(psi_prime)

        delta_o = None
        if object_.optimizable:
            delta_o = p.conj() / (torch.abs(p) ** 2).max()
            delta_o = delta_o * (psi_prime - psi)
            # Add a new dimension in the front for DataParallel to concantenate them
            delta_o = delta_o[None, ...]
            # DataParallel would split the real and imaginary parts of delta_o
            # and store them in an additional dimension at the end. To keep things consistent,
            # we do the splitting manually for cases without DataParallel. 
            delta_o = torch.stack([delta_o.real, delta_o.imag], dim=-1)

        delta_p_all_modes = None
        if probe.optimizable:
            delta_p_all_modes = []
            delta_p = obj_patches.conj() / (torch.abs(obj_patches) ** 2).max(-1).values.max(-1).values.view(-1, 1, 1)
            delta_p = delta_p * (psi_prime - psi)
            delta_p = delta_p.mean(0)
            delta_p_all_modes.append(delta_p)
            delta_p_all_modes = torch.cat(delta_p_all_modes, dim=0)
            # Add a new dimension in the front for DataParallel to concantenate them
            delta_p_all_modes = delta_p_all_modes[None, ...]
            delta_p_all_modes = torch.stack([delta_p_all_modes.real, delta_p_all_modes.imag], dim=-1)

        batch_loss = torch.mean((torch.sqrt(y) - torch.sqrt(y_true)) ** 2)
        return (delta_o, delta_p_all_modes), torch.atleast_1d(batch_loss)

    def apply_updates(self, indices, delta_o, delta_p, *args, **kwargs):
        object_ = self.variable_group.object
        probe = self.variable_group.probe
        probe_positions = self.variable_group.probe_positions

        if delta_o is not None:
            positions = probe_positions.tensor[indices]
            delta_o = delta_o[..., 0] + 1j * delta_o[..., 1]
            delta_o = delta_o.reshape(-1, delta_o.shape[-2], delta_o.shape[-1])
            delta_o = object_.optimizer_params['lr'] * delta_o
            self.variable_group.object.place_patches(positions, delta_o, op='add')

        if delta_p is not None:
            delta_p = delta_p[..., 0] + 1j * delta_p[..., 1]
            delta_p = torch.mean(delta_p, dim=0)
            delta_p = probe.optimizer_params['lr'] * delta_p
            p_new = self.variable_group.probe.get_mode(0) + delta_p
            self.variable_group.probe.tensor.set_data(p_new[None, :, :])