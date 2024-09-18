from typing import Type, Optional

import torch
import tqdm
from torch.utils.data import Dataset

from ptychointerim.ptychotorch.reconstructors import AutodiffReconstructor
from ptychointerim.ptychotorch.data_structures import Ptychography2DVariableGroup
from ptychointerim.forward_models import Ptychography2DForwardModel
from ptychointerim.metrics import MSELossOfSqrt
import ptychointerim.ptychotorch.propagation as prop


class ADLSQMLReconstructor(AutodiffReconstructor):
    """
    The least square maximum likelihood (LSQ-ML) algorithm described in
    
    Odstrčil, M., Menzel, A., & Guizar-Sicairos, M. (2018). Iterative 
    least-squares solver for generalized maximum-likelihood ptychography. 
    Optics Express, 26(3), 3108–3123. doi:10.1364/oe.26.003108
    
    This implementation uses automatic differentiation to get necessary gradients,
    but other steps, including the solving of the step size, are done analytically.
    """

    def __init__(self,
                 variable_group: Ptychography2DVariableGroup,
                 dataset: Dataset,
                 batch_size: int = 1,
                 loss_function: torch.nn.Module = None,
                 n_epochs: int = 100,
                 *args, **kwargs
    ) -> None:
        super().__init__(
            variable_group=variable_group,
            dataset=dataset,
            forward_model_class=Ptychography2DForwardModel,
            forward_model_params={'retain_intermediates': True},
            batch_size=batch_size,
            loss_function=loss_function,
            n_epochs=n_epochs,
            *args, **kwargs)
        self.alpha_psi_far = 0.5
        
    def get_psi_far_step_size(self, y_pred, y_true):
        supported_losses = (MSELossOfSqrt, torch.nn.PoissonNLLLoss)
        if not isinstance(self.loss_function, supported_losses):
            raise ValueError('{} is an unsupported loss function. Legal loss functions '
                             'are {}'.format(
                                 self.loss_function, supported_losses))
        if isinstance(self.loss_function, MSELossOfSqrt):
            alpha = 0.5
            return alpha  # Eq. 16
        elif isinstance(self.loss_function, torch.nn.PoissonNLLLoss):
            xi = 1 - y_true / y_pred  # Eq. 17
            batchsize = y_true.shape[0]
            alpha = torch.sum(y_pred - xi * (y_true / (1 - self.alpha_psi_far * xi))) / batchsize
            alpha = alpha / (torch.sum(xi ** 2 * y_pred) / batchsize)
            return alpha
        
    def run_reciprocal_space_step(self, y_pred, y_true):
        """
        Run step 1 of LSQ-ML, which updates psi. 
        
        :return: (batch_size, n_probe_modes, h, w) complex tensor
        """
        # gradient as in Eq. 12a/b
        dl_dpsi_far = self.get_forward_model().psi_far.grad
        with torch.no_grad():
            self.alpha_psi_far = self.get_psi_far_step_size(y_pred, y_true)
            psi_far = self.get_forward_model().psi_far
            psi_far = psi_far - self.alpha_psi_far * dl_dpsi_far  # Eq. 14
            psi_opt = prop.back_propagate_far_field(psi_far)
        return psi_opt
    
    def run_real_space_step(self, psi_opt):
        """
        Run step 2 of LSQ-ML, which updates the object, probe, and other variables
        using psi updated in step 1. 
        
        This step is independent of the likelihood function chosen through `loss_function`.
        
        :param psi_opt: a (batch_size, n_probe_modes, h, w) complex tensor. Should be 
            psi updated in step 1.
        """
        positions = self.get_forward_model().positions
        psi_upd = self.get_forward_model().psi
        # chi.shape == (batch_size, n_probe_modes, h, w)
        chi = psi_opt - psi_upd  # Eq, 19
        obj_patches = self.get_forward_model().obj_patches
        
        delta_p = self._calculate_probe_update_direction(chi, obj_patches)
        
    def update_probe(self, chi, obj_patches, delta=1e-5):
        delta_p_i = self._calculate_probe_gradient(chi, obj_patches)  # Eq. 24a
        alpha_p_i = self._calculate_probe_update_step_size(delta_p_i, chi, obj_patches)  # Eq. 23a
        delta_p_hat = self._calculate_probe_update_direction(delta_p_i)  # Eq. 25a
        
        
        return delta_p_hat
        
    def _calculate_probe_gradient(self, chi, obj_patches):
        # Shape of chi:          (batch_size, n_probe_modes, h, w)
        # Shape of obj_patches:  (batch_size, h, w)
        # Shape of delta_p:      (batch_size, n_probe_modes, h, w)
        delta_p = chi * obj_patches.conj()[:, None, :, :]  # Eq. 24a
        return delta_p
        
    def _calculate_probe_update_step_size(self, delta_p, chi, obj_patches, gamma=1e-5):
        # Shape of delta_p:      (batch_size, n_probe_modes, h, w)
        # Shape of chi:          (batch_size, n_probe_modes, h, w)
        # Shape of obj_patches:  (batch_size, h, w)
        # Shape of alpha:        (batch_size, n_probe_modes, h, w)
        delta_p_o = delta_p * obj_patches[:, None, :, :]
        alpha_p_i = torch.real(delta_p_o.conj() * chi)
        # Shape of alpha:        (batch_size,)
        alpha_p_i = alpha_p_i.sum(-1).sum(-1).sum(-1)
        alpha_denom = (delta_p_o.abs() ** 2).sum(-1).sum(-1).sum(-1)
        alpha_p_i = alpha_p_i / (alpha_denom + gamma)  # Eq. 23a
        return alpha_p_i
        
    def _calculate_probe_update_direction(self, delta_p, delta=1e-5):
        object_ = self.variable_group.object.tensor.complex()
        # Sum over batch dimension
        # Shape of delta_p_hat:  (n_probe_modes, h, w)
        delta_p_hat = torch.sum(delta_p, dim=0)  # Eq. 25a
        delta_p_hat = delta_p_hat / ((object_.abs() ** 2).sum() + delta)
        return delta_p_hat
    
    def _apply_probe_update(self, alpha_p_i, delta_p_hat, obj_patches):
        # Shape of alpha_p_i:        (batch_size,)
        # Shape of delta_p_hat:      (n_probe_modes, h, w)
        # Shape of obj_patches:      (batch_size, h, w)
        # Shape of update_vec:       (batch_size, n_probe_modes, h, w)
        update_vec = delta_p_hat * obj_patches[:, None, :, :].abs() ** 2
        update_vec = update_vec * alpha_p_i[:, None, None, None]
        # Shape of update_vec:       (n_probe_modes, h, w)
        update_vec = update_vec.sum(0)
        # Shape of update_vec_denom: (h, w)
        update_vec_denom = (obj_patches.abs() ** 2).sum(0)
        update_vec = update_vec / update_vec_denom
        self.variable_group.probe.tensor.set_complex(
            self.variable_group.probe.tensor.complex() + update_vec
        )
    
    def update_object(self, chi, obj_patches, positions, delta=1e-5):
        delta_o_i = self._calculate_object_patch_gradient(chi)
        alpha_o_i = self._calculate_object_update_step_size(delta_o_i, chi)
        delta_o_hat = self._calculate_object_update_direction(delta_o_i, positions)
        return delta_o_hat
    
    def _calculate_object_patch_gradient(self, chi):
        # Shape of probe:        (n_probe_modes, h, w)
        probe = self.variable_group.probe.tensor.complex()
        # Shape of chi:          (batch_size, n_probe_modes, h, w)
        # Shape delta_o_patches: (batch_size, h, w)
        # Multiply and sum over probe mode dimension
        delta_o_patches = torch.sum(chi * probe.conj(), dim=1)  # Eq. 24b
        return delta_o_patches
    
    def _calculate_object_update_step_size(self, delta_o_patches, chi, gamma=1e-5):
        probe = self.variable_group.probe.tensor.complex()
        # Shape of delta_o_patches: (batch_size, h, w)
        # Shape of probe:           (batch_size, n_probe_modes, h, w)
        delta_o_patches_p = delta_o_patches[:, None, :, :] * probe
        # Shape of chi:             (batch_size, n_probe_modes, h, w)
        alpha_o_i = torch.real(delta_o_patches_p.conj() * chi)
        # Shape of alpha_o_i:       (batch_size,)
        alpha_o_i = alpha_o_i.sum(-1).sum(-1).sum(-1)
        alpha_denom = (delta_o_patches_p.abs() ** 2).sum(-1).sum(-1).sum(-1)
        alpha_o_i = alpha_o_i / (alpha_denom + gamma)  # Eq. 23b
        return delta_o_patches
    
    def _calculate_object_update_direction(self, delta_o_patches, positions, delta=1e-5):
        # Shape of probe:        (n_probe_modes, h, w)
        probe = self.variable_group.probe.tensor.complex()
        object_ = self.variable_group.object.tensor.complex()
        # Shape of chi:          (batch_size, n_probe_modes, h, w)
        # Shape delta_o_patches: (batch_size, h, w)
        # Multiply and sum over probe mode dimension
        delta_o_hat = object_.place_patches_on_empty_buffer(delta_o_patches, positions)  # Eq. 25b
        # Shape of delta_o_hat:  (h_whole, w_whole)
        delta_o_hat = delta_o_hat / ((probe.abs() ** 2).sum() + delta)
        return delta_o_hat
        
    def run(self, *args, **kwargs):
        for i_epoch in tqdm.trange(self.n_epochs):
            epoch_loss = 0.0
            for batch_data in self.dataloader:
                input_data = [x.to(torch.get_default_device()) for x in batch_data[:-1]]
                y_true = batch_data[-1].to(torch.get_default_device())

                # Run forward model and calculate gradients
                y_pred = self.forward_model(*input_data)
                batch_loss = self.loss_function(y_pred, y_true)
                batch_loss.backward()
                
                psi_opt = self.run_reciprocal_space_step(y_pred, y_true)
                
                self.step_all_optimizers()
                self.forward_model.zero_grad()

                epoch_loss = epoch_loss + batch_loss.item()
            epoch_loss = epoch_loss / len(self.dataloader)
            self.loss_tracker.conclude_epoch(epoch=i_epoch, loss=epoch_loss)
            self.loss_tracker.print_latest()