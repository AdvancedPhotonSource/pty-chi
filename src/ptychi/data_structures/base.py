# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Optional, Union, Tuple, Sequence, TYPE_CHECKING
import logging

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from numpy import ndarray

from ptychi.utils import to_tensor
import ptychi.maps as maps
import ptychi.api.options.base as obase

if TYPE_CHECKING:
    import ptychi.api as api

logger = logging.getLogger(__name__)


class ComplexTensor(Module):
    """
    A module that stores the real and imaginary parts of a complex tensor
    as real tensors.

    The support of PyTorch DataParallel on complex parameters is flawed. To
    avoid the issue, complex parameters are stored as two real tensors.
    """

    def __init__(
        self, 
        data: Union[Tensor, ndarray], 
        requires_grad: bool = True, 
        data_as_parameter: bool = True, 
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        data = to_tensor(data)
        data = torch.stack([data.real, data.imag], dim=-1).requires_grad_(requires_grad)
        data = data.type(torch.get_default_dtype())

        if data_as_parameter:
            self.register_parameter(name="data", param=Parameter(data))
        else:
            self.register_buffer("data", data)

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

    def set_data(self, data: Union[Tensor, ndarray], slicer=None):
        if slicer is None:
            slicer = (slice(None),)
        elif not isinstance(slicer, Sequence):
            slicer = (slicer,)
        data = to_tensor(data)
        data = torch.stack([data.real, data.imag], dim=-1)
        data = data.type(torch.get_default_dtype())
        self.data[*slicer].copy_(to_tensor(data))


class ReconstructParameter(Module):
    name = None
    optimizable: bool = True
    optimization_plan: "api.OptimizationPlan" = None
    optimizer = None
    is_dummy = False

    def __init__(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        data: Optional[Union[Tensor, ndarray]] = None,
        is_complex: bool = False,
        name: Optional[str] = None,
        options: "api.options.base.ParameterOptions" = None,
        data_as_parameter: bool = True,
        build_optimizer: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """The base reconstructor parameter class.

        Parameters
        ----------
        shape : Optional[Tuple[int, ...]], optional
            The shape of the parameter.
        data : Optional[Union[Tensor, ndarray]], optional
            The data of the parameter.
        is_complex : bool, optional
            Whether the parameter is complex.
        name : Optional[str], optional
            The name of the parameter.
        options : api.options.base.ParameterOptions, optional
            Options of the parameter.
        data_as_parameter : bool, optional
            Whether the data is stored as a torch.Parameter. In most cases this should be True,
            but for DIPObject, the data is not directly optimized so it should just be a buffer.
        """
        super().__init__(*args, **kwargs)
        if shape is None and data is None:
            raise ValueError("Either shape or data must be specified.")
        if options is None:
            if isinstance(self, DummyParameter):
                options = obase.ParameterOptions(optimizable=False)
            else:
                raise ValueError("Parameter options of {} must be specified.".format(self.name))

        self.name = name
        self.options = options
        self.optimizable = self.options.optimizable
        self.optimization_plan = self.options.optimization_plan
        if self.optimization_plan is None:
            raise ValueError("Optimization plan of {} is not specified.".format(self.name))
        self.optimizer_class = maps.get_optimizer_by_enum(self.options.optimizer)

        self.optimizer_params = (
            {} if self.options.optimizer_params is None else self.options.optimizer_params
        )
        # If optimizer_params has 'lr', it will overwrite the step_size.
        self.optimizer_params = dict(
            {"lr": self.options.step_size}, **self.options.optimizer_params
        )
        self.optimizer = None
        
        self.sub_modules = []
        self.optimizable_sub_modules = []
        self.is_complex = is_complex
        self.preconditioner = None
        self.update_buffer = None

        if is_complex:
            if data is not None:
                self.tensor = ComplexTensor(data, data_as_parameter=data_as_parameter).requires_grad_(self.optimizable)
            else:
                self.tensor = ComplexTensor(torch.zeros(shape), data_as_parameter=data_as_parameter, requires_grad=self.optimizable)
        else:
            if data is not None:
                tensor = to_tensor(data).requires_grad_(self.optimizable)
            else:
                tensor = torch.zeros(shape).requires_grad_(self.optimizable)
            # Register the tensor as a parameter. In subclasses, do the same for any
            # additional differentiable parameters. If you have a buffer that does not
            # need gradients, use register_buffer instead.
            if data_as_parameter:
                self.register_parameter("tensor", Parameter(tensor))
            else:
                self.register_buffer("tensor", tensor)

        if build_optimizer:
            self.build_optimizer()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.tensor.shape

    @property
    def data(self) -> Tensor:
        """Get a copy of the parameter data. For complex parameters,
        it returns a complex tensor rather than real tensors giving the
        real and imaginary parts as in the internal representation
        of ComplexTensor.

        Returns
        -------
        Tensor
            A copy of the parameter data.
        """
        if self.is_complex:
            return self.tensor.complex()
        else:
            return self.tensor.clone()
        
    def register_optimizable_sub_module(self, sub_module):
        if sub_module.optimizable and sub_module not in self.optimizable_sub_modules:
            self.optimizable_sub_modules.append(sub_module)
        if sub_module not in self.sub_modules:
            self.sub_modules.append(sub_module)

    def build_optimizer(self):
        if self.optimizable and self.optimizer_class is None:
            raise ValueError(
                "Parameter {} is optimizable but no optimizer is specified.".format(self.name)
            )
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

    def set_data(
        self, data, slicer: Optional[Union[slice, int] | tuple[Union[slice, int], ...]] = None
    ):
        if slicer is None:
            slicer = (slice(None),)
        elif not isinstance(slicer, Sequence):
            slicer = (slicer,)
        if isinstance(self.tensor, ComplexTensor):
            self.tensor.set_data(data, slicer=slicer)
        else:
            self.tensor[*slicer].copy_(to_tensor(data))

    def get_grad(self):
        if isinstance(self.tensor, ComplexTensor):
            return self.tensor.data.grad[..., 0] + 1j * self.tensor.data.grad[..., 1]
        else:
            return self.tensor.grad

    def set_grad(
        self,
        grad: Tensor,
        slicer: Optional[Union[slice, int] | tuple[Union[slice, int], ...]] = None,
    ):
        """
        Populate the `grad` field of the contained tensor, so that it can optimized
        by PyTorch optimizers. You should not need this for AutodiffReconstructor.
        However, method without automatic differentiation needs this to fill in the gradients
        manually.

        Parameters
        ----------
        grad : Tensor
            A tensor giving the gradient. If the gradient is complex, give it as it is.
            This routine will separate the real and imaginary parts and write them into
            the tensor.grad inside the ComplexTensor object.
        slicer : Optional[Union[slice, int] | tuple[Union[slice, int], ...]]
            A tuple of, or a single slice object or integer, that defines the region of
            the region of the gradient to update. The shape of `grad` should match
            the region given by `slicer`, if given. If None, the whole gradient is updated.
        """
        if self.tensor.data.grad is None and slicer is not None:
            raise ValueError("Setting gradient with slicing is not allowed when gradient is None.")
        if slicer is None:
            slicer = (slice(None),)
        elif not isinstance(slicer, Sequence):
            slicer = (slicer,)
        if len(slicer) > len(self.shape):
            raise ValueError("The number of slices should not exceed the number of dimensions.")
        if isinstance(self.tensor, ComplexTensor):
            grad = torch.stack([grad.real, grad.imag], dim=-1)
            if self.tensor.data.grad is None:
                self.tensor.data.grad = grad
            else:
                self.tensor.data.grad[*slicer, ..., :] = grad
        else:
            if self.tensor.grad is None:
                self.tensor.grad = grad
            else:
                self.tensor.grad[*slicer] = grad

    def initialize_grad(self):
        """
        Initialize the gradient with zeros.
        """
        if isinstance(self.tensor, ComplexTensor):
            self.tensor.data.grad = torch.zeros_like(self.tensor.data)
        else:
            self.tensor.grad = torch.zeros_like(self.tensor)

    def post_update_hook(self, *args, **kwargs):
        pass

    def optimization_enabled(self, epoch: int):
        if self.optimizable and self.optimization_plan.is_enabled(epoch):
            enabled = True
        else:
            enabled = False
        logger.debug(f"{self.name} optimization enabled at epoch {epoch}: {enabled}")
        return enabled
    
    def step_optimizer(self, limit: float = None):
        """Step the optimizer with gradient filled in. This function
        can optionally impose a limit on the magnitude of the update.

        Parameters
        ----------
        limit : float, optional
            The maximum allowed magnitude of the update. Set to None to disable the limit.
        """
        if limit is not None and limit <= 0:
            raise ValueError("`limit` should either be None or a positive number.")
        if limit == torch.inf:
            limit = None
        if limit is not None:
            data0 = self.data
        self.optimizer.step()
        if limit is not None:
            data = self.data
            dx = data - data0
            update_mag = dx.abs()
            exceed_mask = update_mag > limit
            dx[exceed_mask] = dx[exceed_mask] * limit / update_mag[exceed_mask]
            self.set_data(data0 + dx)


class DummyParameter(ReconstructParameter):
    is_dummy = True

    def __init__(self, *args, **kwargs):
        super().__init__(shape=(1,), *args, **kwargs)

    def optimization_enabled(self, *args, **kwargs):
        return False


class BoundingBox(torch.nn.Module):
    def __init__(self, sy, ey, sx, ex, origin=(0, 0)):
        super().__init__()
        tensor = to_tensor([sy, ey, sx, ex])
        origin = to_tensor(origin)
        self.register_buffer("tensor", tensor)
        self.register_buffer("origin", origin)

    def __repr__(self):
        return __class__.__name__ + "(sy={}, ey={}, sx={}, ex={}, origin={})".format(
            self.sy, self.ey, self.sx, self.ex, self.origin
        )

    @property
    def sy(self):
        return self.tensor[0]

    @property
    def ey(self):
        return self.tensor[1]

    @property
    def sx(self):
        return self.tensor[2]

    @property
    def ex(self):
        return self.tensor[3]

    def get_bbox_with_top_left_origin(self) -> "BoundingBox":
        """
        Get a new bounding box with the top left origin.
        """
        bbox = BoundingBox(*(self.origin.repeat_interleave(2) + self.tensor), origin=(0, 0))
        return bbox

    def get_slicer(self) -> slice:
        return (slice(int(self.sy), int(self.ey)), slice(int(self.sx), int(self.ex)))
