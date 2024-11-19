from typing import Optional, Union, Tuple, Sequence, TYPE_CHECKING
import logging
import dataclasses

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
        self, data: Union[Tensor, ndarray], requires_grad: bool = True, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        data = to_tensor(data)
        data = torch.stack([data.real, data.imag], dim=-1).requires_grad_(requires_grad)
        data = data.type(torch.get_default_dtype())

        self.register_parameter(name="data", param=Parameter(data))

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
        *args,
        **kwargs,
    ) -> None:
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

        self.is_complex = is_complex
        self.preconditioner = None
        self.update_buffer = None

        if is_complex:
            if data is not None:
                self.tensor = ComplexTensor(data).requires_grad_(self.optimizable)
            else:
                self.tensor = ComplexTensor(torch.zeros(shape), requires_grad=self.optimizable)
        else:
            if data is not None:
                tensor = to_tensor(data).requires_grad_(self.optimizable)
            else:
                tensor = torch.zeros(shape).requires_grad_(self.optimizable)
            # Register the tensor as a parameter. In subclasses, do the same for any
            # additional differentiable parameters. If you have a buffer that does not
            # need gradients, use register_buffer instead.
            self.register_parameter("tensor", Parameter(tensor))

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

    def set_data(self, data, slicer: Optional[Union[slice, int] | tuple[Union[slice, int], ...]] = None):
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

    def get_config_dict(self):
        return self.options.get_non_data_fields()


class DummyParameter(ReconstructParameter):
    is_dummy = True

    def __init__(self, *args, **kwargs):
        super().__init__(shape=(1,), *args, **kwargs)

    def optimization_enabled(self, *args, **kwargs):
        return False
