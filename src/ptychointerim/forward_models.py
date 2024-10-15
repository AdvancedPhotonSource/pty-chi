from typing import Optional
import torch
from torch import Tensor
from torch.nn import ModuleList

import ptychointerim.ptychotorch.data_structures as ds
from ptychointerim.propagate import (
    WavefieldPropagatorParameters,
    AngularSpectrumPropagator,
    FourierPropagator,
)
from ptychointerim.metrics import MSELossOfSqrt


class ForwardModel(torch.nn.Module):
    def __init__(
        self,
        parameter_group: "ds.ParameterGroup",
        retain_intermediates: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        if not isinstance(parameter_group, ds.ParameterGroup):
            raise TypeError(f"variable_group must be a VariableGroup, not {type(parameter_group)}")

        self.parameter_group = parameter_group
        self.retain_intermediates = retain_intermediates
        self.optimizable_parameters: ModuleList[ds.ReconstructParameter] = ModuleList()
        self.propagator = None
        self.intermediate_variables = {}

    def register_optimizable_parameters(self):
        for var in self.parameter_group.__dict__.values():
            if var.optimizable:
                self.optimizable_parameters.append(var)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def post_differentiation_hook(self, *args, **kwargs):
        pass

    def record_intermediate_variable(self, name, var):
        if self.retain_intermediates:
            self.intermediate_variables[name] = var


class Ptychography2DForwardModel(ForwardModel):
    def __init__(
        self,
        parameter_group: "ds.Ptychography2DParameterGroup",
        retain_intermediates: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(parameter_group, *args, **kwargs)
        self.retain_intermediates = retain_intermediates

        self.far_field_propagator = FourierPropagator()

        # This step is essential as it sets the variables to be attributes of
        # the forward modelobject. Only with this can these buffers be copied
        # to the correct devices in DataParallel.
        self.object = parameter_group.object
        self.probe = parameter_group.probe
        self.probe_positions = parameter_group.probe_positions
        self.opr_mode_weights = parameter_group.opr_mode_weights

        # Intermediate variables. Only used if retain_intermediate is True.
        self.intermediate_variables = {
            "positions": None,
            "obj_patches": None,
            "psi": None,
            "psi_far": None,
        }

    def check_inputs(self):
        if self.probe.has_multiple_opr_modes:
            if self.opr_mode_weights is None:
                raise ValueError(
                    "OPRModeWeights must be given when the probe has multiple OPR modes."
                )

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

        self.record_intermediate_variable("psi", psi)
        return psi

    def forward_far_field(self, psi: Tensor) -> Tensor:
        """
        Propagate exit waves to far field.

        Parameters
        ----------
        psi: Tensor
            A (batch_size, n_probe_modes, h, w) tensor of exit waves.

        Returns
        -------
        Tensor
            A (batch_size, n_probe_modes, h, w) tensor of far field waves.
        """
        psi_far = self.far_field_propagator.propagate_forward(psi)
        self.record_intermediate_variable("psi_far", psi_far)
        return psi_far

    def forward(self, indices: Tensor, return_object_patches: bool = False) -> Tensor:
        """
        Run ptychographic forward simulation and calculate the measured intensities.

        Parameters
        ----------
        indices : Tensor
            A (N,) tensor of diffraction pattern indices in the batch.
        positions : Tensor
            A (N, 2) tensor of probe positions in pixels.

        Returns
        -------
        Tensor
            Measured intensities (squared magnitudes).
        """
        positions = self.probe_positions.tensor[indices]
        obj_patches = self.object.extract_patches(positions, self.probe.get_spatial_shape())

        self.record_intermediate_variable("positions", positions)
        self.record_intermediate_variable("obj_patches", obj_patches)

        psi = self.forward_real_space(indices, obj_patches)
        psi_far = self.forward_far_field(psi)

        y = torch.abs(psi_far) ** 2
        # Sum along probe modes
        y = y.sum(1)

        returns = [y]
        if return_object_patches:
            returns.append(obj_patches)
        if len(returns) == 1:
            return returns[0]
        else:
            return returns

    def post_differentiation_hook(self, indices, y_true, **kwargs):
        self.scale_gradients(y_true)

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
            self.object.tensor.data.grad = (
                self.object.tensor.data.grad
                / self.probe.get_all_mode_intensity().max()
                * patterns.numel()
            )
        # Assuming (obj_patches.abs() ** 2).max() == 1.0
        if self.probe.optimizable:
            self.probe.tensor.data.grad = self.probe.tensor.data.grad * (
                patterns.numel() / len(patterns)
            )


class MultislicePtychographyForwardModel(Ptychography2DForwardModel):
    def __init__(
        self,
        parameter_group: "ds.Ptychography2DParameterGroup",
        retain_intermediates: bool = False,
        wavelength_m: float = 1e-9,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            parameter_group=parameter_group,
            retain_intermediates=retain_intermediates,
            *args,
            **kwargs,
        )
        if not isinstance(self.parameter_group.object, ds.MultisliceObject):
            raise TypeError(
                f"Object must be a MultisliceObject, not {type(self.parameter_group.object)}"
            )

        self.wavelength_m = wavelength_m
        self.prop_params = None

        self.near_field_propagator = None
        self.build_propagator()

    def build_propagator(self):
        self.prop_params = WavefieldPropagatorParameters.create_simple(
            wavelength_m=self.wavelength_m,
            width_px=self.probe.shape[-1],
            height_px=self.probe.shape[-2],
            pixel_width_m=self.object.pixel_size_m,
            pixel_height_m=self.object.pixel_size_m,
            propagation_distance_m=self.object.slice_spacings_m[0],
        )
        self.near_field_propagator = AngularSpectrumPropagator(self.prop_params)

    def forward_real_space(self, indices, obj_patches):
        # Shape of obj_patches:   (batch_size, n_slices, h, w)
        if self.probe.has_multiple_opr_modes:
            # Shape of probe:     (batch_size, n_modes, h, w)
            probe = self.probe.get_unique_probes(
                self.opr_mode_weights.get_weights(indices), mode_to_apply=0
            )
        else:
            # Shape of probe:     (n_modes, h, w)
            probe = self.probe.data

        if self.retain_intermediates:
            slice_psis = []

        slice_psi_prop = probe
        for i_slice in range(self.parameter_group.object.n_slices):
            slice_patches = obj_patches[:, i_slice, ...]

            # Modulate wavefield.
            # Shape of slice_psi: (batch_size, n_modes, h, w)
            slice_psi = slice_patches[:, None, :, :] * slice_psi_prop

            if self.retain_intermediates:
                slice_psis.append(slice_psi)

            # Propagate wavefield.
            if i_slice < self.parameter_group.object.n_slices - 1:
                slice_psi_prop = self.propagate_to_next_slice(slice_psi, i_slice)
        if self.retain_intermediates:
            # Shape of slice_psis: (batch_size, n_slices - 1, n_modes, h, w)
            self.record_intermediate_variable("slice_psis", torch.stack(slice_psis, dim=1))
        return slice_psi
    
    def propagate_to_next_slice(self, psi: Tensor, slice_index: int):
        """
        Propagate wavefield to the next slice by the distance given by
        `self.object.slice_spacing_m[slice_index]`.

        Parameters
        ----------
        psi : Tensor.
            A (batch_size, n_modes, h, w) complex tensor giving the wavefield at 
            the current slice.
        slice_index : int
            The index of the current slice.

        Returns
        -------
        Tensor
            The wavefield propagated to the next slice. 
        """
        self.prop_params = WavefieldPropagatorParameters.create_simple(
                    wavelength_m=self.wavelength_m,
                    width_px=self.probe.shape[-1],
                    height_px=self.probe.shape[-2],
                    pixel_width_m=self.object.pixel_size_m,
                    pixel_height_m=self.object.pixel_size_m,
                    propagation_distance_m=self.object.slice_spacings_m[slice_index],
                )
        self.near_field_propagator.update(self.prop_params)
        slice_psi_prop = self.near_field_propagator.propagate_forward(psi)
        return slice_psi_prop
    
    def propagate_to_previous_slice(self, psi: Tensor, slice_index: int):
        """
        Propagate wavefield to the previous slice by the distance given by
        `self.object.slice_spacing_m[slice_index - 1]`.

        Parameters
        ----------
        psi : Tensor.
            A (batch_size, n_modes, h, w) complex tensor giving the wavefield at 
            the current slice.
        slice_index : int
            The index of the current slice.

        Returns
        -------
        Tensor
            The wavefield propagated to the next slice. 
        """
        if slice_index == 0:
            return psi
        self.prop_params = WavefieldPropagatorParameters.create_simple(
                    wavelength_m=self.wavelength_m,
                    width_px=self.probe.shape[-1],
                    height_px=self.probe.shape[-2],
                    pixel_width_m=self.object.pixel_size_m,
                    pixel_height_m=self.object.pixel_size_m,
                    propagation_distance_m=self.object.slice_spacings_m[slice_index - 1],
                )
        self.near_field_propagator.update(self.prop_params)
        slice_psi_prop = self.near_field_propagator.propagate_backward(psi)
        return slice_psi_prop

    def forward(self, indices: Tensor, return_object_patches: bool = False) -> Tensor:
        """
        Run ptychographic forward simulation and calculate the measured intensities.

        Parameters
        ----------
        indices : Tensor
            A (N,) tensor of diffraction pattern indices in the batch.
        positions : Tensor
            A (N, 2) tensor of probe positions in pixels.

        Returns
        -------
        Tensor
            Measured intensities (squared magnitudes).
        """
        return super().forward(indices, return_object_patches=return_object_patches)


class NoiseModel(torch.nn.Module):
    def __init__(self, eps=1e-6, valid_pixel_mask: Optional[Tensor] = None) -> None:
        super().__init__()
        self.eps = eps
        self.noise_statistics = None
        self.valid_pixel_mask = valid_pixel_mask

    def nll(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Calculate the negative log-likelihood.
        """
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError


class GaussianNoiseModel(NoiseModel):
    def __init__(self, sigma: float = 0.5, eps: float = 1e-6, *args, **kwargs) -> None:
        super().__init__(eps=eps, *args, **kwargs)
        self.noise_statistics = "gaussian"
        self.sigma = sigma
        self.loss_function = MSELossOfSqrt()

    def nll(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # This is averaged over all pixels, so it differs from Eq. 11a in Odstrcil (2018)
        # by a factor of 1 / y_pred.numel().
        loss = self.loss_function(y_pred, y_true) / self.sigma**2 * 0.5
        return loss


class PtychographyGaussianNoiseModel(GaussianNoiseModel):
    def __init__(self, sigma: float = 0.5, eps: float = 1e-6, *args, **kwargs) -> None:
        super().__init__(sigma=sigma, eps=eps, *args, **kwargs)

    def backward_to_psi_far(self, y_pred, y_true, psi_far):
        """
        Compute the gradient of the NLL with respect to far field wavefront.
        """
        # Shape of g:       (batch_size, h, w)
        # Shape of psi_far: (batch_size, n_probe_modes, h, w)
        g = 1 - torch.sqrt(y_true / (y_pred + self.eps) + self.eps)  # Eq. 12b
        if self.valid_pixel_mask is not None:
            g[:, torch.logical_not(self.valid_pixel_mask)] = 0
        w = 1 / (2 * self.sigma) ** 2
        g = 2 * w * g[:, None, :, :] * psi_far
        return g


class PoissonNoiseModel(NoiseModel):
    def __init__(self, eps: float = 1e-6, *args, **kwargs) -> None:
        super().__init__(eps=eps, *args, **kwargs)
        self.noise_statistics = "poisson"
        self.loss_function = torch.nn.PoissonNLLLoss(log_input=False)

    def nll(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # This is averaged over all pixels, so it differs from Eq. 11a in Odstrcil (2018)
        # by a factor of 1 / y_pred.numel().
        loss = self.loss_function(y_pred, y_true)
        return loss


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
