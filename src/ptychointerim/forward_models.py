from typing import Optional
import torch
from torch import Tensor
from torch.nn import ModuleList

from ptychointerim.ptychotorch.data_structures import (Variable, VariableGroup, Ptychography2DVariableGroup)
import ptychointerim.ptychotorch.propagation as prop
from ptychointerim.metrics import MSELossOfSqrt


class ForwardModel(torch.nn.Module):
    
    def __init__(self, variable_group: VariableGroup, *args, **kwargs) -> None:
        super().__init__()
        
        assert isinstance(variable_group, VariableGroup)
        
        self.variable_group = variable_group
        self.optimizable_variables: ModuleList[Variable] = ModuleList()
        
    def register_optimizable_parameters(self):
        for var in self.variable_group.__dict__.values():
            if var.optimizable:
                self.optimizable_variables.append(var)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def post_differentiation_hook(self, *args, **kwargs):
        pass
    
    
class Ptychography2DForwardModel(ForwardModel):
    
    def __init__(
            self, 
            variable_group: Ptychography2DVariableGroup,
            retain_intermediates: bool = False,
            *args, **kwargs) -> None:
        super().__init__(variable_group, *args, **kwargs)
        self.retain_intermediates = retain_intermediates
        
        self.object = variable_group.object
        self.probe = variable_group.probe
        self.probe_positions = variable_group.probe_positions
        self.opr_mode_weights = variable_group.opr_mode_weights
        
        # Intermediate variables. Only used if retain_intermediate is True.
        self.intermediate_variables = {
            'positions': None,
            'obj_patches': None,
            'psi': None,
            'psi_far': None
        }
        
    def check_inputs(self):
        if self.probe.has_multiple_opr_modes:
            assert self.opr_mode_weights is not None, \
                'OPRModeWeights must be given when the probe has multiple OPR modes.'
        
    def forward_real_space(self, indices: Tensor, obj_patches: Tensor) -> Tensor:
        if self.probe.has_multiple_opr_modes:
            # Shape of probe:  (batch_size, n_modes, h, w)
            probe = self.probe.get_unique_probes(
                self.opr_mode_weights.get_weights(indices), mode_to_apply=0
            )
        else:
            # Shape of probe:  (n_modes, h, w)
            probe = self.probe.data
        # Shape of psi:        (batch_size, n_probe_modes, h, w)
        psi = obj_patches[:, None, :, :] * probe
        return psi
    
    def forward_far_field(self, psi: Tensor) -> Tensor:
        """
        Propagate exit waves to far field. 

        :param psi: a (n_patches, n_probe_modes, h, w) tensor of exit waves.
        :return: (n_patches, h, w) tensor of detected intensities.
        """
        psi_far = prop.propagate_far_field(psi)
        return psi_far

    def forward(self, indices: Tensor, return_object_patches: bool = False) -> Tensor:
        """Run ptychographic forward simulation and calculate the measured intensities.

        :param indices: A (N,) tensor of diffraction pattern indices in the batch.
        :param positions: A (N, 2) tensor of probe positions in pixels.
        :return: measured intensities (squared magnitudes).
        """
        positions = self.probe_positions.tensor[indices]
        obj_patches = self.object.extract_patches(positions, self.probe.get_spatial_shape())
        
        if self.retain_intermediates:
            self.intermediate_variables['positions'] = positions
            self.intermediate_variables['obj_patches'] = obj_patches
            psi = self.forward_real_space(indices, obj_patches)
            psi_far = self.forward_far_field(psi)
            
            self.intermediate_variables['psi'] = psi
            self.intermediate_variables['psi_far'] = psi_far
            self.intermediate_variables['psi_far'].requires_grad_(True)
            self.intermediate_variables['psi_far'].retain_grad()
            
            y = torch.abs(self.intermediate_variables['psi_far']) ** 2
            # Sum along probe modes
            y = y.sum(1)
        else:
            y = 0.0
            if self.probe.has_multiple_opr_modes:
                # Use OPR modes only for the first incoherent mode.
                # Shape of p:       (batch_size, n_modes, h, w)
                p = self.probe.get_unique_probes(self.opr_mode_weights.get_weights(indices), mode_to_apply=0)
            else:
                # Shape of p:       (n_modes, h, w)
                p = self.probe.get_opr_mode(0)
            for i_probe_mode in range(self.probe.n_modes):
                p_ith_mode = p[i_probe_mode] if p.ndim == 3 else p[:, i_probe_mode]
                psi = obj_patches * p_ith_mode
                psi_far = prop.propagate_far_field(psi)
                y = y + torch.abs(psi_far) ** 2
                    
        returns = [y]
        if return_object_patches:
            returns.append(obj_patches)
        if len(returns) == 1:
            return returns[0]
        else:
            return returns
        
    def post_differentiation_hook(self, *data_and_label, **kwargs):
        patterns = data_and_label[-1]
        self.scale_gradients(patterns)
    
    def scale_gradients(self, patterns):
        """
        Scale the gradients of object and probe so that they are identical to the
        update functions of ePIE. 
        
        For object, the ePIE update function is
        
            o = o + alpha * p.conj() / (abs(p) ** 2).max() * (psi_prime - psi)
            
        while the gradient given by AD when using MSELoss(reduction="mean") is 
        
            -(1 / (batch_size * h * w)) * alpha * p.conj() * (psi_prime - psi)
            
        To scale the AD gradient to match ePIE, we should
        (1) multiply it by batch_size * h * w;
        (2) divide it by (abs(p) ** 2).max() to make up the ePIE scaling factor.
        
        For probe, the ePIE update function is
        
            p = p + alpha * mean(o.conj() / (abs(o) ** 2).max() * (psi_prime - psi), axis=0)
            
        while the gradient given by AD when using MSELoss(reduction="mean") is 
        
            -(1 / (batch_size * h * w)) * alpha * sum(o.conj() * (psi_prime - psi), axis=0)
            
        To scale the AD gradient to match ePIE, we should
        (1) multiply it by batch_size * h * w;
        (2) divide it by (abs(o) ** 2).max() to make up the ePIE scaling factor 
            (but we can assume this is 1.0);
        (3) divide it by batch_size to make up the mean over the batch dimension.
        """
        # Directly modify the gradients here. Tensor.register_hook has memory leak issue.
        if self.object.optimizable:
            self.object.tensor.data.grad = \
                self.object.tensor.data.grad / self.probe.get_all_mode_intensity().max() \
                    * patterns.numel()
        # Assuming (obj_patches.abs() ** 2).max() == 1.0
        if self.probe.optimizable:
            self.probe.tensor.data.grad = \
                self.probe.tensor.data.grad * (patterns.numel() / len(patterns))


class NoiseModel(torch.nn.Module):
    def __init__(self, eps=1e-6, valid_pixel_mask: Optional[Tensor] = None) -> None:
        super().__init__()
        self.eps = eps
        self.noise_statistics = None
        self.valid_pixel_mask = valid_pixel_mask
        
    def nll(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Calculate the negative log-likelihood.
        
        :param y_true: _description_
        :param y_pred: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        """
        raise NotImplementedError
    
    def backward(self, *args, **kwargs):
        raise NotImplementedError
        
class GaussianNoiseModel(NoiseModel):
    def __init__(self, sigma: float = 0.5, eps: float = 1e-6, *args, **kwargs) -> None:
        super().__init__(eps=eps, *args, **kwargs)
        self.noise_statistics = 'gaussian'
        self.sigma = sigma
        self.loss_function = MSELossOfSqrt()
        
    def nll(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # This is averaged over all pixels, so it differs from Eq. 11a in Odstrcil (2018)
        # by a factor of 1 / y_pred.numel().
        l = self.loss_function(y_pred, y_true) / self.sigma ** 2 * 0.5
        return l
    
    
class PtychographyGaussianNoiseModel(GaussianNoiseModel):
    def __init__(self, sigma: float = 0.5, eps: float = 1e-6, *args, **kwargs) -> None:
        super().__init__(sigma=sigma, eps=eps, *args, **kwargs)
    
    def backward_to_psi_far(self, y_pred, y_true, psi_far):
        """
        Compute the gradient of the NLL with respect to far field wavefront.
        """
        # Shape of g:       (batch_size, h, w)
        # Shape of psi_far: (batch_size, n_probe_modes, h, w)
        g = (1 - torch.sqrt(y_true / (y_pred + self.eps) + self.eps)) # Eq. 12b
        if self.valid_pixel_mask is not None:
            g[:, torch.logical_not(self.valid_pixel_mask)] = 0
        w = 1 / (2 * self.sigma) ** 2
        g = 2 * w * g[:, None, :, :] * psi_far
        return g
        
        
class PoissonNoiseModel(NoiseModel):
    def __init__(self, eps: float = 1e-6, *args, **kwargs) -> None:
        super().__init__(eps=eps, *args, **kwargs)
        self.noise_statistics = 'poisson'
        self.loss_function = torch.nn.PoissonNLLLoss(log_input=False)

    def nll(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # This is averaged over all pixels, so it differs from Eq. 11a in Odstrcil (2018)
        # by a factor of 1 / y_pred.numel().
        l = self.loss_function(y_pred, y_true)
        return l
    

class PtychographyPoissonNoiseModel(PoissonNoiseModel):
    def __init__(self, eps: float = 1e-6, *args, **kwargs) -> None:
        super().__init__(eps=eps, *args, **kwargs)
        
    def backward_to_psi_far(self, y_pred: Tensor, y_true: Tensor, psi_far: Tensor):
        """
        Compute the gradient of the NLL with respect to far field wavefront.
        """
        g = 1 - y_true / (y_pred + self.eps)  # Eq. 12b
        if self.valid_pixel_mask is not None:
            g[:, torch.logical_not(self.valid_pixel_mask)] = 0
        g = g[:, None, :, :] * psi_far
        return g
    