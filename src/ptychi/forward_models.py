import math

from typing import Optional, TYPE_CHECKING
import torch
from torch import Tensor
from torch.nn import ModuleList

import ptychi.data_structures.object
from ptychi.propagate import (
    WavefieldPropagatorParameters,
    AngularSpectrumPropagator,
    FourierPropagator,
)
from ptychi.metrics import MSELossOfSqrt

if TYPE_CHECKING:
    import ptychi.data_structures.parameter_group
    import ptychi.data_structures.base as ds


class ForwardModel(torch.nn.Module):
    def __init__(
        self,
        parameter_group: "ptychi.data_structures.parameter_group.ParameterGroup",
        retain_intermediates: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.parameter_group = parameter_group
        self.retain_intermediates = retain_intermediates
        self.optimizable_parameters: ModuleList[ds.ReconstructParameter] = ModuleList()
        self.propagator = None
        self.intermediate_variables = {}
        
        self.register_optimizable_parameters()

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


class PlanarPtychographyForwardModel(ForwardModel):
    def __init__(
        self,
        parameter_group: "ptychi.data_structures.parameter_group.PlanarPtychographyParameterGroup",
        retain_intermediates: bool = False,
        wavelength_m: Optional[float] = None,
        low_memory_mode: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(parameter_group, *args, **kwargs)
        self.retain_intermediates = retain_intermediates
        self.low_mem_mode = low_memory_mode

        self.far_field_propagator = FourierPropagator()

        # This step is essential as it sets the variables to be attributes of
        # the forward modelobject. Only with this can these buffers be copied
        # to the correct devices in DataParallel.
        self.object = parameter_group.object
        self.probe = parameter_group.probe
        self.probe_positions = parameter_group.probe_positions
        self.opr_mode_weights = parameter_group.opr_mode_weights

        self.wavelength_m = wavelength_m
        self.prop_params = None

        self.near_field_propagator = None
        self.build_propagator()

        # Intermediate variables. Only used if retain_intermediate is True.
        self.intermediate_variables = {
            "positions": None,
            "obj_patches": None,
            "psi": None,
            "psi_far": None,
        }

        self.check_inputs()

    def check_inputs(self):
        if self.probe.has_multiple_opr_modes:
            if self.opr_mode_weights is None:
                raise ValueError(
                    "OPRModeWeights must be given when the probe has multiple OPR modes."
                )
        if self.object.is_multislice:
            if self.wavelength_m is None:
                raise ValueError("Wavelength must be given when the object is multislice.")

    def build_propagator(self):
        if not self.object.is_multislice:
            return
        self.prop_params = WavefieldPropagatorParameters.create_simple(
            wavelength_m=self.wavelength_m,
            width_px=self.probe.shape[-1],
            height_px=self.probe.shape[-2],
            pixel_width_m=self.object.pixel_size_m,
            pixel_height_m=self.object.pixel_size_m,
            propagation_distance_m=self.object.slice_spacings_m[0],
        )
        self.near_field_propagator = AngularSpectrumPropagator(self.prop_params)

    def get_probe(self, indices: Tensor) -> Tensor:
        """
        Get the probe. If OPR modes are present, this will return the unique
        probe for all positions

        Parameters
        ----------
        indices : Tensor
            Indices of diffraction pattern in the batch.

        Returns
        -------
        Tensor
            The probe. The shape is (batch_size, n_modes, h, w) if OPR modes
            are present, and (n_modes, h, w) otherwise.
        """
        if self.probe.has_multiple_opr_modes:
            # Shape of probe:     (batch_size, n_modes, h, w)
            probe = self.probe.get_unique_probes(
                self.opr_mode_weights.get_weights(indices), mode_to_apply=0
            )
        else:
            # Shape of probe:     (n_modes, h, w)
            probe = self.probe.data
        return probe

    def propagate_through_object(self, probe, obj_patches, return_slice_psis=False):
        """
        Propagate through the planar object.

        Parameters
        ----------
        probe : Tensor
            A (batch_size, n_modes, h, w) or (n_modes, h, w) tensor
            of wavefields at the entering plane.
        obj_patches : Tensor
            A (batch_size, n_slices, h, w) tensor of object patches.
        return_slice_psis : bool
            If True, return the wavefields at each slice.

        Returns
        -------
        Tensor
            A (batch_size, n_modes, h, w) tensor of wavefields at the
            exiting plane.
        Tensor, optional
            A (batch_size, n_slices - 1, n_modes, h, w) tensors of pre-modulation
            wavefields at each slice except the first one. This is only returned
            when `return_slice_psis` is True.
        """
        if self.retain_intermediates:
            slice_psis = []

        slice_psi = probe
        for i_slice in range(self.parameter_group.object.n_slices):
            if self.retain_intermediates and i_slice > 0:
                slice_psis.append(slice_psi)

            # Modulate wavefield.
            # Shape of slice_psi: (batch_size, n_modes, h, w)
            slice_patches = obj_patches[:, i_slice, ...]
            slice_psi = slice_patches[:, None, :, :] * slice_psi

            # Propagate wavefield.
            if i_slice < self.parameter_group.object.n_slices - 1:
                slice_psi = self.propagate_to_next_slice(slice_psi, i_slice)
        if self.retain_intermediates:
            # Shape of slice_psis: (batch_size, n_slices - 1, n_modes, h, w)
            if len(slice_psis) > 0:
                self.record_intermediate_variable("slice_psis", torch.stack(slice_psis, dim=1))
            self.record_intermediate_variable("psi", slice_psi)
        if return_slice_psis:
            if "slice_psis" in self.intermediate_variables.keys():
                return slice_psi, self.intermediate_variables["slice_psis"]
            else:
                return slice_psi, None
        return slice_psi

    def forward_real_space(self, indices, obj_patches):
        """
        Propagate through the planar object.

        If `retain_intermediates` is True, the modulated and then propagated
        wavefield at each slice will also be stored in the intermediate
        variable dictionary as "slice_psis".

        Parameters
        ----------
        indices : Tensor
            Indices of diffraction pattern in the batch.
        obj_patches : Tensor
            A (batch_size, n_slices, h, w) tensor of object patches.

        Returns
        -------
        Tensor
            A (batch_size, n_modes, h, w) tensor of wavefields at the
            exiting plane.
        """
        # Shape of obj_patches:   (batch_size, n_slices, h, w)
        probe = self.get_probe(indices)
        slice_psi = self.propagate_through_object(probe, obj_patches, return_slice_psis=False)
        return slice_psi

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
        return_object_patches : bool
            If True, return the object patches along with the intensities.

        Returns
        -------
        Tensor
            Predicted intensities (squared magnitudes).
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

    def forward_low_memory(self, indices: Tensor, return_object_patches: bool = False) -> Tensor:
        """
        The forward model that should give the same result as `forward`, but
        uses less memory.

        Parameters
        ----------
        indices : Tensor
            A (N,) tensor of diffraction pattern indices in the batch.
        return_object_patches : bool
            If True, return the object patches along with the intensities.

        Returns
        -------
        Tensor
            Predicted intensities (squared magnitudes).
        """
        positions = self.probe_positions.tensor[indices]
        obj_patches = self.object.extract_patches(positions, self.probe.get_spatial_shape())

        self.record_intermediate_variable("positions", positions)
        self.record_intermediate_variable("obj_patches", obj_patches)

        probe = self.get_probe(indices)
        slice_psis = []
        y = 0.0
        for i_mode in range(self.probe.n_modes):
            exit_psi, slice_psis_current_mode = self.propagate_through_object(
                probe[..., i_mode : i_mode + 1, :, :], obj_patches, return_slice_psis=True
            )
            if slice_psis_current_mode is not None:
                slice_psis.append(slice_psis_current_mode)
            
            psi_far = self.far_field_propagator.propagate_forward(exit_psi)
            self.record_intermediate_variable("psi_far", psi_far)
            
            y = y + psi_far[..., 0, :, :].abs() ** 2

        # Concatenate all the modes.
        if len(slice_psis) > 0:
            slice_psis = torch.concat(slice_psis, dim=-3)
        
        returns = [y]
        if return_object_patches:
            returns.append(obj_patches)
        if len(returns) == 1:
            return returns[0]
        else:
            return returns

    def post_differentiation_hook(self, indices, y_true, **kwargs):
        self.compensate_for_fft_scaler()
        self.scale_gradients(y_true)
        
    def compensate_for_fft_scaler(self):
        """
        Compensate for the scaling of the FFT. If the normalization of FFT is "ortho",
        nothing is done. If the normalization is "backward" (default in `torch.fft`),
        all gradients are scaled by 1 / sqrt(N).
        """
        if self.far_field_propagator.norm == "backward" or self.far_field_propagator.norm is None:
            factor = self.probe.shape[-2] * self.probe.shape[-1]
            for var in self.optimizable_parameters:
                if var.get_grad() is not None:
                    var.set_grad(var.get_grad() / factor)

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
    def __init__(self, sigma: float = 0.5, eps: float = 1e-9, *args, **kwargs) -> None:
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
    def __init__(self, sigma: float = 0.5, eps: float = 1e-9, *args, **kwargs) -> None:
        super().__init__(sigma=sigma, eps=eps, *args, **kwargs)

    def backward_to_psi_far(self, y_pred, y_true, psi_far):
        """
        Compute the gradient of the NLL with respect to far field wavefront.
        """
        # Shape of g:       (batch_size, h, w)
        # Shape of psi_far: (batch_size, n_probe_modes, h, w)
        g = 1 - torch.sqrt(y_true) / (torch.sqrt(y_pred) + self.eps)  # Eq. 12b
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