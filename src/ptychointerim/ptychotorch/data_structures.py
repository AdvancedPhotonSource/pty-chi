from typing import Optional, Union, Tuple, Type, Literal
import dataclasses

import torch
from torch import Tensor
from torch.nn import Module, Parameter
import numpy as np
from numpy import ndarray

import ptychointerim.image_proc as ip
from ptychointerim.ptychotorch.utils import to_tensor, get_default_complex_dtype
import ptychointerim.maths as pmath


class ComplexTensor(Module):
    """
    A module that stores the real and imaginary parts of a complex tensor
    as real tensors. 
    
    The support of PyTorch DataParallel on complex parameters is flawed. To
    avoid the issue, complex parameters are stored as two real tensors.
    """
    
    def __init__(self, 
                 data: Union[Tensor, ndarray], 
                 requires_grad: bool = True, 
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        data = to_tensor(data)
        data = torch.stack([data.real, data.imag], dim=-1).requires_grad_(requires_grad)
        data = data.type(torch.get_default_dtype())
        
        self.register_parameter(name='data', param=Parameter(data))
        
    def mag(self) -> Tensor:
        return torch.sqrt(self.data[..., 0] ** 2 + self.data[..., 1] ** 2)
    
    def magsq(self) -> Tensor:
        return self.data[..., 0] ** 2 + self.data[..., 1] ** 2
    
    def phase(self) -> Tensor:
        return torch.atan2(self.data[..., 1], self.data[..., 0])
    
    def real(self) -> Tensor:
        return self.data[..., 0]
    
    def imag(self) -> Tensor:
        return self.data[..., 1]
    
    def complex(self) -> Tensor:
        return self.real() + 1j * self.imag()
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape[:-1]
    
    def set_data(self, data: Union[Tensor, ndarray]):
        data = to_tensor(data)
        data = torch.stack([data.real, data.imag], dim=-1)
        data = data.type(torch.get_default_dtype())
        self.data.copy_(to_tensor(data))


class Variable(Module):
    
    name = None
    optimizable: bool = True
    optimizer = None
    
    def __init__(self, 
                 shape: Optional[Tuple[int, ...]] = None, 
                 data: Optional[Union[Tensor, ndarray]] = None,
                 is_complex: bool = False,
                 name: Optional[str] = None, 
                 optimizable: bool = True,
                 optimizer_class: Optional[Type[torch.optim.Optimizer]] = None,
                 optimizer_params: Optional[dict] = None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert shape is not None or data is not None
        self.optimizable = optimizable
        self.name = name
        self.optimizer_class = optimizer_class
        self.optimizer_params = {} if optimizer_params is None else optimizer_params
        self.optimizer = None
        self.is_complex = is_complex
        self.preconditioner = None
        
        if is_complex:
            if data is not None:
                self.tensor = ComplexTensor(data).requires_grad_(optimizable)
            else:
                self.tensor = ComplexTensor(torch.zeros(shape), requires_grad=optimizable)
        else:
            if data is not None:
                tensor = to_tensor(data).requires_grad_(optimizable)
            else:
                tensor = torch.zeros(shape).requires_grad_(optimizable)
            # Register the tensor as a parameter. In subclasses, do the same for any
            # additional differentiable variables. If you have a buffer that does not
            # need gradients, use register_buffer instead.
            self.register_parameter('tensor', Parameter(tensor))
                
        self.build_optimizer()
        
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.tensor.shape
    
    @property
    def data(self) -> Tensor:
        if self.is_complex:
            return self.tensor.complex()
        else:
            return self.tensor.clone()
            
    def build_optimizer(self):
        if self.optimizable and self.optimizer_class is None:
            raise ValueError("optimizer_class must be specified if optimizable is True.")
        if self.optimizable:
            if isinstance(self.tensor, ComplexTensor):
                self.optimizer = self.optimizer_class([self.tensor.data], **self.optimizer_params)
            else:
                self.optimizer = self.optimizer_class([self.tensor], **self.optimizer_params)
            
            
    def set_optimizable(self, optimizable):
        self.optimizable = optimizable
        self.tensor.requires_grad_(optimizable)
        
    def get_tensor(self, name):
        """Get a member tensor in this object.
        
        It is necessary to use this method to access memebers when 
        # (1) the forward model is wrapped in DataParallel,
        # (2) multiple deivces are used,
        # (3) the model has complex parameters. 
        # DataParallel adds an additional dimension at the end of each registered 
        # complex parameter (not an issue for real parameters).
        This method selects the right index along that dimension by checking
        the device ID. 
        """
        var = getattr(self, name)
        # If the current shape has one more dimension than the original shape,
        # it means that the DataParallel wrapper has added an additional
        # dimension. Select the right index from the last dimension.
        if len(var.shape) > len(self.shape):
            dev_id = var.device.index
            if dev_id is None:
                raise RuntimeError("Expecting multi-GPU, but unable to find device ID.")
            var = var[..., dev_id]
        return var
    
    def get_config_dict(self):
        return {'name': self.name, 
                'optimizer_class': str(self.optimizer_class), 
                'optimizer_params': self.optimizer_params,
                'optimizable': self.optimizable}
        
    def set_data(self, data):
        if isinstance(self.tensor, ComplexTensor):
            self.tensor.set_data(data)
        else:
            self.tensor.copy_(to_tensor(data))
            
    def get_grad(self):
        if isinstance(self.tensor, ComplexTensor):
            return self.tensor.data.grad[..., 0] + 1j * self.tensor.data.grad[..., 1]
        else:
            return self.tensor.grad
            
    def set_grad(self, grad):
        """
        Populate the `grad` field of the contained tensor, so that it can optimized
        by PyTorch optimizers. You should not need this for AutodiffReconstructor.
        However, method without automatic differentiation needs this to fill in the gradients
        manually.

        :param grad: tensor of gradient. 
        """
        if isinstance(self.tensor, ComplexTensor):
            grad = torch.stack([grad.real, grad.imag], dim=-1)
            self.tensor.data.grad = grad
        else:
            self.tensor.grad = grad
            
    def post_update_hook(self, *args, **kwargs):
        pass
    

class DummyVariable(Variable):
    def __init__(self, *args, **kwargs):
        super().__init__(shape=(1,), optimizable=False, *args, **kwargs)


class Object(Variable):
    
    pixel_size_m: float = 1.0
    
    def __init__(self, *args, pixel_size_m: float = 1.0, name='object', **kwargs):
        super().__init__(*args, name=name, is_complex=True, **kwargs)
        self.pixel_size_m = pixel_size_m
        center_pixel = torch.tensor(self.shape, device=torch.get_default_device()) / 2.0
        
        self.register_buffer('center_pixel', center_pixel)

    def extract_patches(self, positions, patch_shape, *args, **kwargs):
        raise NotImplementedError
    
    def place_patches(self, positions, patches, *args, **kwargs):
        raise NotImplementedError
        

class Object2D(Object):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def extract_patches(self, positions: Tensor, patch_shape: Tuple[int, int]):
        """Extract patches from 2D object.

        :param positions: a tensor of shape (N, 2) giving the center positions of the patches in pixels.
            The origin of the given positions are assumed to be `self.center_pixel`.
        :param patch_shape: a tuple giving the patch shape in pixels.
        """
        # Positions are provided with the origin in the center of the object support. 
        # We shift the positions so that the origin is in the upper left corner.
        positions = positions + self.center_pixel
        patches = ip.extract_patches_fourier_shift(self.tensor.complex(), positions, patch_shape)
        return patches
    
    def place_patches(self, positions: Tensor, patches: Tensor, *args, **kwargs):
        """Place patches into a 2D object.
        
        :param positions: a tensor of shape (N, 2) giving the center positions of the patches in pixels.
            The origin of the given positions are assumed to be `self.center_pixel`.
        :param patches: (N, H, W) tensor ofimage patches.
        """
        positions = positions + self.center_pixel
        image = ip.place_patches_fourier_shift(self.tensor.complex(), positions, patches)
        self.tensor.set_data(image)
        
    def place_patches_on_empty_buffer(self, positions: Tensor, patches: Tensor, *args, **kwargs):
        """Place patches into a zero array with the same shape as the object.
        
        :param positions: a tensor of shape (N, 2) giving the center positions of the patches in pixels.
        :param patches: (N, H, W) tensor ofimage patches.
        :return: a tensor with the same shape as the object with patches added onto it.
        """
        positions = positions + self.center_pixel
        image = torch.zeros(self.shape, dtype=get_default_complex_dtype(), device=self.tensor.data.device)
        image = ip.place_patches_fourier_shift(image, positions, patches, op='add')
        return image
        
        
class Probe(Variable):
    
    n_modes = 1
    
    def __init__(self, *args, name='probe', **kwargs):
        """
        Represents the probe function in a tensor of shape 
            `(n_opr_modes, n_modes, h, w)`
        where:
        - n_opr_modes is the number of mutually coherent probe modes used in orthogonal
          probe relaxation (OPR). 
        - n_modes is the number of mutually incoherent probe modes.

        :param name: name of the variable, defaults to 'probe'
        """
        super().__init__(*args, name=name, is_complex=True, **kwargs)
        assert len(self.shape) == 4, 'Probe tensor must be of shape (n_opr_modes, n_modes, h, w).'
        
    def shift(self, shifts: Tensor):
        """
        Generate shifted probe. 

        :param shifts: A tensor of shape (2,) or (N, 2) giving the shifts in pixels.
            If a (N, 2)-shaped tensor is given, a batch of shifted probes are generated.
        """
        if shifts.ndim == 1:
            probe_straightened = self.tensor.complex().view(-1, *self.shape[-2:])
            shifted_probe = ip.fourier_shift(
                probe_straightened, 
                shifts[None, :].repeat([[probe_straightened.shape[0], 1, 1]])
            )
            shifted_probe = shifted_probe.view(*self.shape)
        else:
            n_shifts = shifts.shape[0]
            n_images_each_probe = self.shape[0] * self.shape[1]
            probe_straightened = self.tensor.complex().view(n_images_each_probe, *self.shape[-2:])
            probe_straightened = probe_straightened.repeat(n_shifts, 1, 1)
            shifts = shifts.repeat_interleave(n_images_each_probe, dim=0)
            shifted_probe = ip.fourier_shift(probe_straightened, shifts)
            shifted_probe = shifted_probe.reshape(n_shifts, *self.shape)
        return shifted_probe
    
    @property
    def n_modes(self):
        return self.tensor.shape[1]
    
    @property
    def n_opr_modes(self):
        return self.tensor.shape[0]
    
    @property
    def has_multiple_opr_modes(self):
        return self.n_opr_modes > 1

    def get_mode(self, mode: int):
        return self.tensor.complex()[:, mode]
    
    def get_opr_mode(self, mode: int):
        return self.tensor.complex()[mode]
    
    def get_mode_and_opr_mode(self, mode: int, opr_mode: int):
        return self.tensor.complex()[opr_mode, mode]
    
    def get_spatial_shape(self):
        return self.shape[-2:]
    
    def get_all_mode_intensity(
            self, 
            opr_mode: Optional[int] = 0, 
            weights: Optional[Union[Tensor, Variable]] = None,
        ) -> Tensor:
        """
        Get the intensity of all probe modes.

        :param opr_mode: the OPR mode. If this is not None, only the intensity of the chosen
            OPR mode is calculated. Otherwise, it calculates the intensity of the weighted sum
            of all OPR modes. In that case, `weights` must be given.
        :param weights: a (n_opr_modes,) tensor giving the weights of OPR modes.
        :return: _description_
        """
        if isinstance(weights, OPRModeWeights):
            weights = weights.data
        if opr_mode is not None:
            p = self.data[opr_mode]
        else:
            p = (self.data * weights[None, :, :, :]).sum(0)
        return torch.sum((p.abs()) ** 2, dim=0)
    
    def get_unique_probes(self, weights: Union[Tensor, Variable], mode_to_apply: Optional[int] = None) -> Tensor:
        """
        Creates the unique probe for one or more scan points given the weights of eigenmodes.
        
        :param weights: A (n_points, n_opr_modes) or (n_opr_modes,) tensor giving the weights 
            of the eigenmodes. 
        :param mode_to_apply: The incoherent mode for which OPR modes should be applied. The data
            for other modes will be set to the value of the first OPR mode. If None,
            OPR correction will be done to all incoherent modes. 
        :return: A (n_points, n_modes, h, w) tensor of unique probes if weights.ndim == 2, 
            or a (n_modes, h, w) tensor if weights.ndim == 1.
        """
        if isinstance(weights, OPRModeWeights):
            weights = weights.data
        
        p_orig = None
        if mode_to_apply is not None:
            p_orig = self.data.clone()
            p = p_orig[:, [mode_to_apply], :, :]
        else:
            p = self.data.clone()
        if weights.ndim == 1:
            unique_probe = p * weights[:, None, None, None]
            unique_probe = unique_probe.sum(0)
        else:
            unique_probe = p[None, ...] * weights[:, :, None, None, None]
            unique_probe = unique_probe.sum(1)
            
        # If OPR is only applied on one incoherent mode, add in the rest of the modes.
        if mode_to_apply is not None:
            if weights.ndim == 1:
                # Shape of unique_probe:     (1, h, w)
                p_orig[0, [mode_to_apply], :, :] = unique_probe
                unique_probe = p_orig[0, ...]
            else:
                # Shape of unique_probe:     (n_points, 1, h, w)
                p_orig = p_orig[None, ...].repeat(weights.shape[0], 1, 1, 1, 1)
                p_orig[:, 0, [mode_to_apply], :, :] = unique_probe
                unique_probe = p_orig[:, 0, ...]
        return unique_probe
    
    def constrain_opr_mode_orthogonality(self, weights: Union[Tensor, Variable], eps=1e-5):
        """Add the following constraints to variable probe weights

        1. Remove outliars from weights
        2. Enforce orthogonality once per epoch
        3. Sort the variable probes by their total energy
        4. Normalize the variable probes so the energy is contained in the weight
        
        Adapted from Tike (https://github.com/AdvancedPhotonSource/tike). The implementation
        in Tike assumes a separately stored variable probe eigenmodes; here we use the
        PtychoShelves convention and regard the second and following OPR modes as eigenmodes.
        
        Also, this function assumes that OPR correction is only applied to the first
        incoherent mode when mixed state probe is used, as this is what PtychoShelves does. 
        OPR modes of other incoherent modes are ignored, for now. 
        
        :param weights: a (n_points, n_opr_modes) tensor of weights.
        :return: normalized and sorted OPR mode weights.
        """
        if isinstance(weights, OPRModeWeights):
            weights = weights.data
        
        # The main mode of the probe is the first OPR mode, while the
        # variable part of the probe is the second and following OPR modes.
        # The main mode should not change during orthogonalization, but the
        # variable OPR modes should all be orthogonal to it. 
        probe = self.data
        
        if False:
            # Normalize variable probes
            vnorm = pmath.mnorm(probe, dim=(-2, -1), keepdims=True)
            probe /= (vnorm + eps)
            # Shape of weights:      (n_points, n_opr_modes). 
            # Currently, only the first incoherent mode has OPR modes, and the
            # stored weights are for that mode.
            weights = weights * vnorm[:, 0, 0, 0]

        # Orthogonalize variable probes. With Gram-Schmidt, the first
        # OPR mode (i.e., the main mode) should not change during orthogonalization.
        probe = pmath.orthogonalize_gs(
            probe,
            dim=(-2, -1),
            group_dim=-4,
        )

        # Compute the energies of variable OPR modes (i.e., the second and following) 
        # in order to sort probes by energy.
        # Shape of power:         (n_opr_modes - 1,).
        power = pmath.norm(weights[..., 1:], dim=0) ** 2
        
        # Sort the probes by energy
        sorted = torch.argsort(-power)
        weights[:, 1:] = weights[:, sorted + 1]
        # Apply only to the first incoherent mode.
        probe[1:, 0, :, :] = probe[sorted + 1, 0, :, :]

        # Remove outliars from variable probe weights.
        aevol = torch.abs(weights)
        weights = torch.minimum(
            aevol,
            1.5 * torch.quantile(
                aevol,
                0.95,
                dim=0,
                keepdims=True,
            ).type(weights.dtype),
        ) * torch.sign(weights)
            
        # Update stored data.
        self.set_data(probe)
        return weights
    
    def post_update_hook(self, weights: Union[Tensor, Variable]) -> Tensor:
        super().post_update_hook()
        if self.has_multiple_opr_modes:
            weights = self.constrain_opr_mode_orthogonality(weights)
        return weights
    
    
class OPRModeWeights(Variable):
    def __init__(self, *args, name='opr_weights', **kwargs):
        """
        Weights of OPR modes for each scan point.

        :param name: name of the variable, defaults to 'opr_weights'
        """
        super().__init__(*args, name=name, is_complex=False, **kwargs)
        assert len(self.shape) == 2, 'OPR weights must be of shape (n_scan_points, n_opr_modes).'
        
        self.n_opr_modes = self.tensor.shape[1]
        
    def get_weights(self, indices: Union[tuple[int, ...], slice]) -> Tensor:
        return self.data[indices]
    

class ProbePositions(Variable):
    
    pixel_size_m: float = 1.0
    conversion_factor_dict = {'nm': 1e9, 'um': 1e6, 'm': 1.0}
        
    def __init__(self, *args, pixel_size_m: float = 1.0, name: str = 'probe_positions', 
                 update_magnitude_limit=0, **kwargs):
        """Probe positions. 

        :param data: a tensor of shape (N, 2) giving the probe positions in pixels. 
            Input positions should be in row-major order, i.e., y-posiitons come first.
        """
        super().__init__(*args, name=name, is_complex=False, **kwargs)
        self.pixel_size_m = pixel_size_m
        self.update_magnitude_limit = update_magnitude_limit
        
    def get_positions_in_physical_unit(self, unit: str = 'm'):
        return self.tensor * self.pixel_size_m * self.conversion_factor_dict[unit]


@dataclasses.dataclass
class VariableGroup:

    def get_all_variables(self) -> list[Variable]:
        return list(self.__dict__.values())

    def get_optimizable_variables(self) -> list[Variable]:
        ovs = []
        for var in self.get_all_variables():
            if var.optimizable:
                ovs.append(var)
        return ovs
    
    def get_config_dict(self):
        return {var.name: var.get_config_dict() for var in self.get_all_variables()}
    

@dataclasses.dataclass
class PtychographyVariableGroup(VariableGroup):
    
    object: Object

    probe: Probe

    probe_positions: ProbePositions
    
    opr_mode_weights: Optional[OPRModeWeights] = dataclasses.field(default_factory=DummyVariable)


@dataclasses.dataclass
class Ptychography2DVariableGroup(PtychographyVariableGroup):

    object: Object2D
    