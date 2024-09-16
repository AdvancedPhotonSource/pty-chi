from typing import Optional, Union, Tuple, Type
import dataclasses

import torch
from torch import Tensor
from torch.nn import Module, Parameter
import numpy as np
from numpy import ndarray

import ptychointerim.ptychotorch.image_proc as ip
from ptychointerim.ptychotorch.utils import to_tensor, get_default_complex_dtype


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
        requires_grad = self.data.requires_grad
        data = to_tensor(data)
        data = torch.stack([data.real, data.imag], dim=-1)
        data = data.type(torch.get_default_dtype())
        self.data = Parameter(data)
        self.data.requires_grad_(requires_grad)


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
            return self.tensor
            
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
            self.tensor = to_tensor(data)
            
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
        super().__init__(*args, name=name, is_complex=True, **kwargs)
        self.n_modes = self.tensor.shape[0]
        
    def shift(self, shifts: Tensor):
        """Generate shifted probe. 

        :param shifts: A tensor of shape (2,) or (N, 2) giving the shifts in pixels.
            If a (N, 2)-shaped tensor is given, a batch of shifted probes are generated.
        """
        if shifts.ndim == 1:
            shifted_probe = ip.fourier_shift(self.tensor.complex()[None, ...], shifts[None, :])[0]
        else:
            shifted_probe = ip.fourier_shift(self.tensor.complex()[None, ...].repeat(shifts.shape[0], 1, 1), shifts)
        return shifted_probe

    def get_mode(self, mode: int):
        return self.tensor.complex()[mode]
    
    def get_spatial_shape(self):
        return self.shape[-2:]
    
    def get_all_mode_intensity(self):
        return torch.sum((self.tensor.complex().abs()) ** 2, dim=0)


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


@dataclasses.dataclass
class Ptychography2DVariableGroup(PtychographyVariableGroup):

    object: Object2D

    