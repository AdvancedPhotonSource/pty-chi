from typing import Optional, Union, Tuple
import dataclasses
import os
import logging

import torch
from torch import Tensor
from torch.nn import Module, Parameter
import numpy as np
from numpy import ndarray
import tifffile

import ptychointerim.image_proc as ip
from ptychointerim.ptychotorch.utils import to_tensor, get_default_complex_dtype
import ptychointerim.maths as pmath
import ptychointerim.api as api
import ptychointerim.maps as maps
from ptychointerim.propagate import WavefieldPropagator, FourierPropagator


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

    def set_data(self, data: Union[Tensor, ndarray]):
        data = to_tensor(data)
        data = torch.stack([data.real, data.imag], dim=-1)
        data = data.type(torch.get_default_dtype())
        self.data.copy_(to_tensor(data))


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
            options = self.get_option_class()()

        self.name = name
        self.options = options
        self.optimizable = self.options.optimizable
        self.optimization_plan = self.options.optimization_plan
        if self.optimization_plan is None:
            self.optimization_plan = api.OptimizationPlan()
        self.optimizer_class = maps.get_optimizer_by_enum(self.options.optimizer)
        
        self.optimizer_params = (
            {} if self.options.optimizer_params is None else self.options.optimizer_params
        )
        # If optimizer_params has 'lr', it will overwrite the step_size.
        self.optimizer_params = dict({'lr': self.options.step_size}, **self.options.optimizer_params)
        self.optimizer = None
        
        self.is_complex = is_complex
        self.preconditioner = None

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
                self.optimizer = self.optimizer_class(
                    [self.tensor.data], **self.optimizer_params
                )
            else:
                self.optimizer = self.optimizer_class(
                    [self.tensor], **self.optimizer_params
                )

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

    def get_option_class(self):
        if isinstance(self, DummyParameter):
            return api.options.base.ParameterOptions
        try:
            return self.__class__.__init__.__annotations__["options"]
        except KeyError:
            return api.options.base.ParameterOptions

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

    def optimization_enabled(self, epoch: int):
        if self.optimizable and self.optimization_plan.is_enabled(epoch):
            enabled = True
        else:
            enabled = False
        logging.debug(f"{self.name} optimization enabled at epoch {epoch}: {enabled}")
        return enabled

    def get_config_dict(self):
        return self.options.get_non_data_fields()


class DummyParameter(ReconstructParameter):
    is_dummy = True

    def __init__(self, *args, **kwargs):
        super().__init__(shape=(1,), *args, **kwargs)

    def optimization_enabled(self, *args, **kwargs):
        return False


class Object(ReconstructParameter):
    pixel_size_m: float = 1.0

    def __init__(
        self,
        name: str = "object",
        options: "api.options.base.ObjectOptions" = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, name=name, options=options, is_complex=True, **kwargs)
        self.pixel_size_m = options.pixel_size_m
        self.l1_norm_constraint_weight = options.l1_norm_constraint_weight
        self.l1_norm_constraint_stride = options.l1_norm_constraint_stride
        self.smoothness_constraint_alpha = options.smoothness_constraint_alpha
        self.smoothness_constraint_stride = options.smoothness_constraint_stride
        self.total_variation_weight = options.total_variation_weight
        self.total_variation_stride = options.total_variation_stride
        center_pixel = torch.tensor(self.shape, device=torch.get_default_device()) / 2.0

        self.register_buffer("center_pixel", center_pixel)

    def extract_patches(self, positions, patch_shape, *args, **kwargs):
        raise NotImplementedError

    def place_patches(self, positions, patches, *args, **kwargs):
        raise NotImplementedError

    def l1_norm_constraint_enabled(self, current_epoch: int):
        if (
            self.l1_norm_constraint_weight > 0
            and self.optimization_enabled(current_epoch)
            and (current_epoch - self.optimization_plan.start) % self.l1_norm_constraint_stride == 0
        ):
            return True
        else:
            return False

    def constrain_l1_norm(self):
        data = self.data
        l1_grad = torch.sgn(data)
        data = data - self.l1_norm_constraint_weight * l1_grad
        self.set_data(data)
        logging.debug("L1 norm constraint applied to object.")

    def smoothness_constraint_enabled(self, current_epoch: int):
        if (
            self.smoothness_constraint_alpha > 0
            and self.optimization_enabled(current_epoch)
            and (current_epoch - self.optimization_plan.start) % self.smoothness_constraint_stride
            == 0
        ):
            return True
        else:
            return False

    def constrain_smoothness(self) -> None:
        """
        Smooth the magnitude of the object.
        """
        if self.smoothness_constraint_alpha > 1.0 / 8:
            logging.warning(
                f"Alpha = {self.smoothness_constraint_alpha} in smoothness constraint should be less than 1/8."
            )
        psf = torch.ones(3, 3, device=self.device) * self.smoothness_constraint_alpha
        psf[2, 2] = 1 - 8 * self.smoothness_constraint_alpha

        data = self.data
        mag = data.abs()
        mag = ip.convolve2d(mag, psf, "same")
        data = data / data.abs() * mag
        self.set_data(data)

    def total_variation_enabled(self, current_epoch: int):
        if (
            self.total_variation_weight > 0
            and self.optimization_enabled(current_epoch)
            and (current_epoch - self.optimization_plan.start) % self.total_variation_stride == 0
        ):
            return True
        else:
            return False

    def constrain_total_variation(self) -> None:
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
        image = torch.zeros(
            self.shape, dtype=get_default_complex_dtype(), device=self.tensor.data.device
        )
        image = ip.place_patches_fourier_shift(image, positions, patches, op="add")
        return image

    def constrain_total_variation(self) -> None:
        data = self.data
        data = ip.total_variation_2d_chambolle(data, lmbda=self.total_variation_weight, niter=2)
        self.set_data(data)


class MultisliceObject(Object2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if len(self.shape) != 3:
            raise ValueError("MultisliceObject should have a shape of (n_slices, h, w).")
        if self.options.slice_spacings_m is None or len(self.options.slice_spacings_m) != self.n_slices - 1:
            raise ValueError("The number of slice spacings must be n_slices - 1.")

        self.register_buffer("slice_spacings_m", to_tensor(self.options.slice_spacings_m))

        center_pixel = torch.tensor(self.shape[1:], device=torch.get_default_device()) / 2.0
        self.register_buffer("center_pixel", center_pixel)

    @property
    def n_slices(self):
        return self.shape[0]

    @property
    def lateral_shape(self):
        return self.shape[1:]

    def get_slice(self, index):
        return self.data[index, ...]

    def extract_patches(self, positions: Tensor, patch_shape: Tuple[int, int]):
        """Extract (n_patches, n_slices, h', w') patches from the multislice object.

        :param positions: a tensor of shape (N, 2) giving the center positions of the patches in pixels.
            The origin of the given positions are assumed to be `self.center_pixel`.
        :param patch_shape: a tuple giving the lateral patch shape in pixels.
        """
        # Positions are provided with the origin in the center of the object support.
        # We shift the positions so that the origin is in the upper left corner.
        positions = positions + self.center_pixel
        patches_all_slices = []
        for i_slice in range(self.n_slices):
            patches = ip.extract_patches_fourier_shift(
                self.get_slice(i_slice), positions, patch_shape
            )
            patches_all_slices.append(patches)
        patches_all_slices = torch.stack(patches_all_slices, dim=1)
        return patches_all_slices

    def place_patches(self, positions: Tensor, patches: Tensor, *args, **kwargs):
        """Place patches into a 2D object.

        :param positions: a tensor of shape (N, 2) giving the center positions of the patches in pixels.
            The origin of the given positions are assumed to be `self.center_pixel`.
        :param patches: (n_patches, n_slices, H, W) tensor ofimage patches.
        """
        positions = positions + self.center_pixel
        updated_slices = []
        for i_slice in range(self.n_slices):
            image = ip.place_patches_fourier_shift(
                self.get_slice(i_slice), positions, patches[:, i_slice, ...]
            )
            updated_slices.append(image)
        updated_slices = torch.stack(updated_slices, dim=0)
        self.tensor.set_data(updated_slices)

    def place_patches_on_empty_buffer(self, positions: Tensor, patches: Tensor, *args, **kwargs):
        """Place patches into a zero array with the lateral shape of the object.

        :param positions: a tensor of shape (N, 2) giving the center positions of the patches in pixels.
        :param patches: (N, H, W) tensor ofimage patches.
        :return: a tensor with the lateral shape of the object with patches added onto it.
        """
        positions = positions + self.center_pixel
        image = torch.zeros(
            self.lateral_shape, dtype=get_default_complex_dtype(), device=self.tensor.data.device
        )
        image = ip.place_patches_fourier_shift(image, positions, patches, op="add")
        return image

    def constrain_total_variation(self) -> None:
        data = self.data
        for i_slice in range(self.n_slices):
            data[i_slice] = ip.total_variation_2d_chambolle(
                data[i_slice], lmbda=self.total_variation_weight, niter=2
            )
        self.set_data(data)


class Probe(ReconstructParameter):
    # TODO: eigenmode_update_relaxation is only used for LSQML. We should create dataclasses
    # to contain additional options for ReconstructParameter classes, and subclass them for specific
    # reconstruction algorithms - for example, ProbeOptions -> LSQMLProbeOptions.
    def __init__(
        self,
        name: str = "probe",
        options: "api.options.base.ProbeOptions" = None,
        *args,
        **kwargs,
    ):
        """
        Represents the probe function in a tensor of shape
            `(n_opr_modes, n_modes, h, w)`
        where:
        - n_opr_modes is the number of mutually coherent probe modes used in orthogonal
          probe relaxation (OPR).
        - n_modes is the number of mutually incoherent probe modes.

        :param name: name of the parameter, defaults to 'probe'.
        :param eigenmode_update_relaxation: relaxation factor, or effectively the step size,
            for eigenmode update in LSQML.
        :param probe_power: the target probe power. If greater than 0, probe power constraint
            is run every `probe_power_constraint_stride` epochs, where it scales the probe
            and object intensity such that the power of the far-field probe is `probe_power`.
        :param probe_power_constraint_stride: the number of epochs between probe power constraint
            updates.
        :param orthogonalize_incoherent_modes: whether to orthogonalize incoherent probe modes. If
            True, the incoherent probe modes are orthogonalized every
            `orthogonalize_incoherent_modes_stride` epochs.
        :param orthogonalize_incoherent_modes_stride: the number of epochs between orthogonalizing
            the incoherent probe modes.
        """
        super().__init__(*args, name=name, options=options, is_complex=True, **kwargs)
        if len(self.shape) != 4:
            raise ValueError("Probe tensor must be of shape (n_opr_modes, n_modes, h, w).")

        self.probe_power = options.probe_power
        self.probe_power_constraint_stride = options.probe_power_constraint_stride
        self.orthogonalize_incoherent_modes = options.orthogonalize_incoherent_modes
        self.orthogonalize_incoherent_modes_stride = options.orthogonalize_incoherent_modes_stride
        self.orthogonalize_incoherent_modes_method = options.orthogonalize_incoherent_modes_method
        self.orthogonalize_opr_modes = options.orthogonalize_opr_modes
        self.orthogonalize_opr_modes_stride = options.orthogonalize_opr_modes_stride

    def shift(self, shifts: Tensor):
        """
        Generate shifted probe.

        :param shifts: A tensor of shape (2,) or (N, 2) giving the shifts in pixels.
            If a (N, 2)-shaped tensor is given, a batch of shifted probes are generated.
        """
        if shifts.ndim == 1:
            probe_straightened = self.tensor.complex().view(-1, *self.shape[-2:])
            shifted_probe = ip.fourier_shift(
                probe_straightened, shifts[None, :].repeat([[probe_straightened.shape[0], 1, 1]])
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

    @property
    def has_multiple_incoherent_modes(self):
        return self.n_modes > 1

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
        weights: Optional[Union[Tensor, ReconstructParameter]] = None,
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

    def get_unique_probes(
        self, weights: Union[Tensor, ReconstructParameter], mode_to_apply: Optional[int] = None
    ) -> Tensor:
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

    def constrain_incoherent_modes_orthogonality(self):
        """Orthogonalize the incoherent probe modes for the first OPR mode."""
        probe = self.data

        norm_first_mode_orig = pmath.norm(probe[0, 0], dim=(-2, -1))

        if self.orthogonalize_incoherent_modes_method == api.enums.OrthogonalizationMethods.GS:
            func = pmath.orthogonalize_gs
        elif self.orthogonalize_incoherent_modes_method == api.enums.OrthogonalizationMethods.SVD:
            func = pmath.orthogonalize_svd
        else:
            raise NotImplementedError(
                f"Orthogonalization method {self.orthogonalize_incoherent_modes_method} "
                "is not supported."
            )
        probe[0] = func(
            probe[0],
            dim=(-2, -1),
            group_dim=0,
        )

        # Restore norm.
        norm_first_mode_new = pmath.norm(probe[0, 0], dim=(-2, -1))
        probe = probe * norm_first_mode_orig / norm_first_mode_new

        self.set_data(probe)

    def constrain_opr_mode_orthogonality(
        self, weights: Union[Tensor, ReconstructParameter], eps=1e-5
    ):
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

        # TODO: remove outliars by polynomial fitting (remove_variable_probe_ambiguities.m)

        # Normalize eigenmodes and adjust weights.
        eigenmodes = probe[1:, ...]
        vnorm = pmath.mnorm(eigenmodes, dim=(-2, -1), keepdims=True)
        eigenmodes /= vnorm + eps
        # Shape of weights:      (n_points, n_opr_modes).
        # Currently, only the first incoherent mode has OPR modes, and the
        # stored weights are for that mode.
        weights[:, 1:] = weights[:, 1:] * vnorm[:, 0, 0, 0]

        # Orthogonalize variable probes. With Gram-Schmidt, the first
        # OPR mode (i.e., the main mode) should not change during orthogonalization.
        probe = pmath.orthogonalize_gs(
            probe,
            dim=(-2, -1),
            group_dim=0,
        )

        if False:
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
            1.5
            * torch.quantile(
                aevol,
                0.95,
                dim=0,
                keepdims=True,
            ).type(weights.dtype),
        ) * torch.sign(weights)

        # Update stored data.
        self.set_data(probe)
        return weights

    def constrain_probe_power(
        self,
        object_: Object,
        opr_mode_weights: Union[Tensor, "OPRModeWeights"],
        propagator: Optional[WavefieldPropagator] = None,
    ) -> None:
        if self.probe_power <= 0.0:
            return

        if isinstance(opr_mode_weights, OPRModeWeights):
            opr_mode_weights = opr_mode_weights.data

        if propagator is None:
            propagator = FourierPropagator()

        # Shape of probe_composed:        (n_modes, h, w)
        if self.has_multiple_opr_modes:
            avg_weights = opr_mode_weights.mean(dim=0)
            probe_composed = self.get_unique_probes(avg_weights, mode_to_apply=0)
        else:
            probe_composed = self.get_opr_mode(0)

        # TODO: use propagator for forward simulation
        propagated_probe = propagator.propagate_forward(probe_composed)
        propagated_probe_power = torch.sum(propagated_probe.abs() ** 2)
        power_correction = torch.sqrt(self.probe_power / propagated_probe_power)

        self.set_data(self.data * power_correction)
        object_.set_data(object_.data / power_correction)

        logging.info("Probe and object scaled by {}.".format(power_correction))

    def post_update_hook(self) -> None:
        super().post_update_hook()

    def normalize_eigenmodes(self):
        """
        Normalize all eigenmodes (the second and following OPR modes) such that each of them
        has a squared norm equal to the number of pixels in the probe.
        """
        if not self.has_multiple_opr_modes:
            return
        eigen_modes = self.data[1:, ...]
        for i_opr_mode in range(eigen_modes.shape[0]):
            for i_mode in range(eigen_modes.shape[1]):
                eigen_modes[i_opr_mode, i_mode, :, :] /= (
                    pmath.mnorm(eigen_modes[i_opr_mode, i_mode, :, :], dim=(-2, -1)) + 1e-8
                )

        new_data = self.data
        new_data[1:, ...] = eigen_modes
        self.set_data(new_data)

    def opr_mode_orthogonalization_enabled(self, current_epoch: int) -> bool:
        enabled = self.optimization_enabled(current_epoch)
        return (
            enabled
            and self.has_multiple_opr_modes
            and self.orthogonalize_opr_modes
            and (current_epoch - self.optimization_plan.start) % self.orthogonalize_opr_modes_stride
            == 0
        )

    def save_tiff(self, path: str):
        """
        Save the probe's magnitude and phase as 2 TIFF files. Each file contains
        an array of tiles, where the rows correspond to incoherent probe modes
        and columns correspond to OPR modes.

        :param path: path to save. "_phase" and "_mag" will be appended to the filename.
        """
        fname = os.path.splitext(path)[0]
        mag_img = np.empty([self.shape[3] * self.shape[1], self.shape[2] * self.shape[0]])
        phase_img = np.empty([self.shape[3] * self.shape[1], self.shape[2] * self.shape[0]])
        data = self.data
        for i_mode in range(self.shape[1]):
            for i_opr_mode in range(self.shape[0]):
                mag_img[
                    i_mode * self.shape[3] : (i_mode + 1) * self.shape[3],
                    i_opr_mode * self.shape[2] : (i_opr_mode + 1) * self.shape[2],
                ] = data[i_opr_mode, i_mode, :, :].abs().detach().cpu().numpy()
                phase_img[
                    i_mode * self.shape[3] : (i_mode + 1) * self.shape[3],
                    i_opr_mode * self.shape[2] : (i_opr_mode + 1) * self.shape[2],
                ] = torch.angle(data[i_opr_mode, i_mode, :, :]).detach().cpu().numpy()
        tifffile.imsave(fname + "_mag.tif", mag_img)
        tifffile.imsave(fname + "_phase.tif", phase_img)


class OPRModeWeights(ReconstructParameter):
    # TODO: update_relaxation is only used for LSQML. We should create dataclasses
    # to contain additional options for ReconstructParameter classes, and subclass them for specific
    # reconstruction algorithms - for example, OPRModeWeightsOptions -> LSQMLOPRModeWeightsOptions.
    def __init__(
        self,
        *args,
        name="opr_weights",
        options: "api.options.base.OPRModeWeightsOptions" = None,
        **kwargs,
    ):
        """
        Weights of OPR modes for each scan point.

        :param name: name of the parameter.
        :param update_relaxation: relaxation factor, or effectively the step size, for
            the update step in LSQML.
        """
        super().__init__(*args, name=name, options=options, is_complex=False, **kwargs)
        if len(self.shape) != 2:
            raise ValueError("OPR weights must be of shape (n_scan_points, n_opr_modes).")

        if self.optimizable:
            if not (options.optimize_eigenmode_weights or options.optimize_intensity_variation):
                raise ValueError(
                    "When OPRModeWeights is optimizable, at least 1 of "
                    "optimize_eigenmode_weights and optimize_intensity_variation "
                    "should be set to True."
                )

        self.optimize_eigenmode_weights = options.optimize_eigenmode_weights
        self.optimize_intensity_variation = options.optimize_intensity_variation

        self.n_opr_modes = self.tensor.shape[1]

    def build_optimizer(self):
        if self.optimizer_class is None:
            return
        if self.optimizable:
            if isinstance(self.tensor, ComplexTensor):
                self.optimizer = self.optimizer_class([self.tensor.data], **self.optimizer_params)
            else:
                self.optimizer = self.optimizer_class([self.tensor], **self.optimizer_params)

    def get_weights(self, indices: Union[tuple[int, ...], slice]) -> Tensor:
        return self.data[indices]

    def optimization_enabled(self, epoch: int):
        enabled = super().optimization_enabled(epoch)
        return enabled and (self.optimize_eigenmode_weights or self.optimize_intensity_variation)

    def eigenmode_weight_optimization_enabled(self, epoch: int):
        enabled = super().optimization_enabled(epoch)
        return enabled and self.optimize_eigenmode_weights

    def intensity_variation_optimization_enabled(self, epoch: int):
        enabled = super().optimization_enabled(epoch)
        return enabled and self.optimize_intensity_variation


class ProbePositions(ReconstructParameter):
    pixel_size_m: float = 1.0
    conversion_factor_dict = {"nm": 1e9, "um": 1e6, "m": 1.0}

    def __init__(
        self,
        *args,
        name: str = "probe_positions",
        options: "api.options.base.ProbePositionOptions" = None,
        probe: Probe = None,
        object_step_size: float = None,
        **kwargs,
    ):
        """Probe positions. 

        :param data: a tensor of shape (N, 2) giving the probe positions in pixels.
            Input positions should be in row-major order, i.e., y-positons come first.
        """
        super().__init__(*args, name=name, options=options, is_complex=False, **kwargs)
        import ptychointerim.position_correction as position_correction
        self.pixel_size_m = options.pixel_size_m
        self.update_magnitude_limit = options.update_magnitude_limit
        self.position_correction = position_correction.PositionCorrection(
            probe=probe,
            object_step_size=object_step_size,
            correction_options=options.correction_options,
        )

    def get_positions_in_physical_unit(self, unit: str = "m"):
        return self.tensor * self.pixel_size_m * self.conversion_factor_dict[unit]


@dataclasses.dataclass
class ParameterGroup:
    def get_all_parameters(self) -> list[ReconstructParameter]:
        return list(self.__dict__.values())

    def get_optimizable_parameters(self) -> list[ReconstructParameter]:
        ovs = []
        for var in self.get_all_parameters():
            if var.optimizable:
                ovs.append(var)
        return ovs

    def get_config_dict(self):
        return {var.name: var.get_config_dict() for var in self.get_all_parameters()}


@dataclasses.dataclass
class PtychographyParameterGroup(ParameterGroup):
    object: Object

    probe: Probe

    probe_positions: ProbePositions

    opr_mode_weights: Optional[OPRModeWeights] = dataclasses.field(default_factory=DummyParameter)

    def __post_init__(self):
        if self.probe.has_multiple_opr_modes and self.opr_mode_weights is None:
            raise ValueError(
                "OPRModeWeights must be provided when the probe has multiple OPR modes."
            )


@dataclasses.dataclass
class Ptychography2DParameterGroup(PtychographyParameterGroup):
    object: Object2D
