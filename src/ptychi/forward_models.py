# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import logging
import dataclasses
from typing import Optional, TYPE_CHECKING
import torch
from torch import Tensor
from torch.nn import ModuleList

import ptychi.data_structures.object
import ptychi.data_structures.probe
from ptychi.propagate import (
    WavefieldPropagatorParameters,
    AngularSpectrumPropagator,
    FourierPropagator,
    FresnelTransformPropagator
)
from ptychi.metrics import MSELossOfSqrt
import ptychi.image_proc as ip
from ptychi.timing.timer_utils import timer
import ptychi.global_settings as glb
if TYPE_CHECKING:
    import ptychi.data_structures.parameter_group
    import ptychi.data_structures.base as dsbase
    
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class IntermediateVariables(dict):
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        return setattr(self, key, value)


class ForwardModel(torch.nn.Module):
    def __init__(
        self,
        parameter_group: "ptychi.data_structures.parameter_group.ParameterGroup",
        retain_intermediates: bool = False,
        detector_size: Optional[tuple[int, int]] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        The base forward model class.
        
        Parameters
        ----------
        parameter_group : ptychi.data_structures.parameter_group.ParameterGroup
            The parameter group.
        retain_intermediates : bool
            If True, intermediate variables are retained in `self.intermediate_variables`.
        detector_size : tuple[int, int], optional
            The size of the detector. If given and if the detector size is smaller than
            the size of the probe, the probe is cropped to the detector size.
        """
        super().__init__()

        self.parameter_group = parameter_group
        self.retain_intermediates = retain_intermediates
        self.optimizable_parameters: ModuleList[dsbase.ReconstructParameter] = ModuleList()
        self.propagator = None
        self.detector_size = detector_size
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
            if isinstance(self.intermediate_variables[name], list):
                self.intermediate_variables[name].append(var.detach())
            else:
                self.intermediate_variables[name] = var.detach()


class PlanarPtychographyForwardModel(ForwardModel):
    @dataclasses.dataclass
    class PlanarPtychographyIntermediateVariables(IntermediateVariables):
        positions: Tensor = None
        """Tensor of shape `[batch_size, 2]`, probe positions in the 
        current batch.
        """
        
        obj_patches: Tensor = None
        """Tensor of shape `[batch_size, n_slices, h, w]`, object patches of the 
        current batch.
        """
        
        psi: Tensor = None
        """Tensor of shape `[batch_size, n_probe_modes, h, w]`, wavefields at 
        the exit plane. For multislice objects, this is the wavefield at the 
        exit plane of the last slice.
        """
        
        psi_far: Tensor = None
        """Tensor of shape `[batch_size, n_probe_modes, h, w]`, wavefields at 
        the far field.
        """
        
        shifted_unique_probes: list[Tensor] = dataclasses.field(default_factory=list)
        """List of `n_slices` tensors of shape `[batch_size, n_probe_modes, h, w]`, unique 
        probes at each slice for every position. The position-specificity is due 
        to OPR modes and/or subpixel shifts 
        (when `apply_subpixel_shifts_on_probe == True`).
        """
    
    def __init__(
        self,
        parameter_group: "ptychi.data_structures.parameter_group.PlanarPtychographyParameterGroup",
        retain_intermediates: bool = False,
        detector_size: Optional[tuple[int, int]] = None,
        wavelength_m: Optional[float] = None,
        free_space_propagation_distance_m: Optional[float] = torch.inf,
        pad_for_shift: Optional[int] = 0,
        apply_subpixel_shifts_on_probe: bool = True,
        low_memory_mode: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        The forward model for planar ptychography, which supports multislice propagation
        but does not work for dense 3D objects that requires rotation. 
        
        While running the forward model, some intermediate variables are recorded in 
        `self.intermediate_variables` to be used by the reconstructor. For more details,
        see `PlanarPtychographyForwardModel.PlanarPtychographyIntermediateVariables`.

        Parameters
        ----------
        parameter_group : ptychi.data_structures.parameter_group.PlanarPtychographyParameterGroup
            The parameter group.
        retain_intermediates : bool
            If True, intermediate variables are retained in `self.intermediate_variables`.
        detector_size : tuple[int, int], optional
            The size of the detector. If given and if the detector size is smaller than
            the size of the probe, the probe is cropped to the detector size.
        wavelength_m : float, optional
            The wavelength of the probe in meters.
        free_space_propagation_distance_m : float, optional
            The free-space propagation distance in meters, or `inf` for far-field.
        pad_for_shift : int, optional
            If given, the probe (or object when `apply_subpixel_shifts_on_probe == False`) 
            is padded with border values by this amount before shifting.
        apply_subpixel_shifts_on_probe : bool
            If True, the subpixel parts of probe positions are accounted for by uniquely shifting
            the probe for each position, while all object patches are extracted at integer positions.
            This creates a unique probe for each position, stored in 
            `self.intermediate_variables["shifted_unique_probes"]`. If False, subpixel sihfts
            are applied on the object patches, and a common probe is used for all positions.
            The choice of the subpixel handling scheme must be consistent with the reconstructor.
        low_memory_mode : bool
            If True, incoherent probe modes are propagated sequentially rather than in parallel
            which reduces the memory usage.
        """
        super().__init__(
            parameter_group, retain_intermediates=retain_intermediates, detector_size=detector_size, 
            *args, **kwargs
        )
        self.low_mem_mode = low_memory_mode

        # This step is essential as it sets the variables to be attributes of
        # the forward modelobject. Only with this can these buffers be copied
        # to the correct devices in DataParallel.
        self.object = parameter_group.object
        self.probe = parameter_group.probe
        self.probe_positions = parameter_group.probe_positions
        self.opr_mode_weights = parameter_group.opr_mode_weights

        self.wavelength_m = wavelength_m
        self.free_space_propagation_distance_m = free_space_propagation_distance_m
        
        self.apply_subpixel_shifts_on_probe = apply_subpixel_shifts_on_probe
        self.pad_for_shift = pad_for_shift
        
        # Build free space propagator.
        self.free_space_propagator = None
        self.build_free_space_propagator()

        # Build in-object (multislice) propagator.
        self.in_object_propagator = None
        self.in_object_prop_params = None
        self.build_in_object_propagator()

        # Intermediate variables. Only used if retain_intermediate is True.
        self.intermediate_variables = self.PlanarPtychographyIntermediateVariables()

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
            
        if self.free_space_propagation_distance_m < torch.inf and not self.pad_for_shift:
            logger.warning(
                "It seems that you are running near-field propagation "
                f"(free_space_propagation_distance_m = {self.free_space_propagation_distance_m}). "
                "If you are using a bright-field probe, it is recommended to set `pad_for_shift` "
                "to a positive integer to avoid aliasing. Currently it is 0."
            )
        
        if not self.apply_subpixel_shifts_on_probe and not self.pad_for_shift:
            ValueError(
                "Using `pad_for_shift == 0` with `apply_subpixel_shifts_on_probe == False` is "
                "prohibited because this causes serious image artifacts when applying subpixel "
                "shifts to object patches. Please set `pad_for_shift` to at least 1 and try again."
            )

    def build_in_object_propagator(self):
        if not self.object.is_multislice:
            return
        self.in_object_prop_params = WavefieldPropagatorParameters.create_simple(
            wavelength_m=self.wavelength_m,
            width_px=self.probe.shape[-1],
            height_px=self.probe.shape[-2],
            pixel_width_m=self.object.pixel_size_m,
            pixel_height_m=self.object.pixel_size_m / self.object.options.pixel_size_aspect_ratio,
            propagation_distance_m=self.object.slice_spacings.data[0],
        )
        self.in_object_propagator = AngularSpectrumPropagator(self.in_object_prop_params)
        
    def build_free_space_propagator(self):
        if self.free_space_propagation_distance_m == torch.inf:
            self.free_space_propagator = FourierPropagator()
        else:
            params = WavefieldPropagatorParameters.create_simple(
                wavelength_m=self.wavelength_m,
                width_px=self.probe.shape[-1],
                height_px=self.probe.shape[-2],
                pixel_width_m=self.object.pixel_size_m,
                pixel_height_m=self.object.pixel_size_m,
                propagation_distance_m=self.free_space_propagation_distance_m,
            )
            # TODO: AngularSpectrumPropagator uses analytical transfer function. Using the FFT
            # of real-space impulse response might offer better sampling for large propagation distances.
            if params.is_fresnel_transform_preferrable():
                self.free_space_propagator = FresnelTransformPropagator(params)
            else:
                self.free_space_propagator = AngularSpectrumPropagator(params)
                
    def extract_object_patches(self, indices: Tensor) -> Tensor:
        positions = self.probe_positions.data[indices]
        if self.apply_subpixel_shifts_on_probe:
            obj_patches = self.object.extract_patches(
                positions.round().int(), self.probe.get_spatial_shape(),
                integer_mode=True
            )
        else:
            obj_patches = self.object.extract_patches(
                positions, self.probe.get_spatial_shape(),
                pad_for_shift=self.pad_for_shift,
                integer_mode=False
            )
        return obj_patches

    def get_unique_probes(self, indices: Tensor, always_return_probe_batch: bool = True) -> Tensor:
        """Get the unique probes for all positions in the batch. 
        If OPR modes are present, this will return the unique
        probe for all positions.

        Parameters
        ----------
        indices : Tensor
            Indices of diffraction pattern in the batch.
        always_return_probe_batch : bool
            If True, the probe is always returned as a (batch_size, n_modes, h, w) tensor even if
            there is only one OPR mode. Otherwise, the probe is returned as a (n_modes, h, w) 
            tensor if there is only one OPR mode.

        Returns
        -------
        Tensor
            A (batch_size, n_modes, h, w) tensor of unique probes.
        """
        if self.probe.has_multiple_opr_modes:
            # Shape of probe:     (batch_size, n_modes, h, w)
            probe = self.probe.get_unique_probes(
                self.opr_mode_weights.get_weights(indices), mode_to_apply=0
            )
        else:
            if always_return_probe_batch:
                probe = self.probe.data.repeat(indices.shape[0], 1, 1, 1)
            else:
                probe = self.probe.get_opr_mode(0)
        return probe
    
    def shift_unique_probes(self, indices: Tensor, unique_probes: Tensor, first_mode_only: bool = False):
        """Apply subpixel shifts to the unique probes to compensate for the fractional
        probe positions not accounted for when extracting object patches.
        
        Parameters
        ----------
        indices : Tensor
            Indices of diffraction pattern in the batch.
        unique_probes : Tensor
            A (batch_size, n_modes, h, w) tensor of unique probes.
        first_mode_only : bool
            If True, only the first mode is shifted.
        """
        orig_shape = unique_probes.shape
        fractional_shifts = self.probe_positions.data[indices] - self.probe_positions.data[indices].round()
        
        if first_mode_only:
            unique_probe_to_shift = unique_probes[..., 0, :, :]
        else:
            unique_probe_to_shift = unique_probes.reshape(unique_probes.shape[0] * unique_probes.shape[1], *unique_probes.shape[2:])

        unique_probe_shifted = ip.shift_images(
            unique_probe_to_shift, 
            fractional_shifts, 
            method=self.parameter_group.object.options.patch_interpolation_method,
            adjoint=False,
            pad=self.pad_for_shift
        )
        
        if first_mode_only:
            unique_probes = torch.cat([unique_probe_shifted[..., None, :, :], unique_probes[..., 1:, :, :]], dim=-3)
        else:
            unique_probes = unique_probe_shifted.reshape(orig_shape)
        return unique_probes
        
    @timer()
    def propagate_through_object(self, probe, obj_patches):
        """
        Propagate through the planar object.

        Parameters
        ----------
        probe : Tensor
            A (batch_size, n_modes, h, w) or (n_modes, h, w) tensor
            of wavefields at the entering plane.
        obj_patches : Tensor
            A (batch_size, n_slices, h, w) tensor of object patches.

        Returns
        -------
        Tensor
            A (batch_size, n_modes, h, w) tensor of wavefields at the
            exiting plane.
        """
        slice_psi = probe
        for i_slice in range(self.parameter_group.object.n_slices):
            if i_slice > 0:
                self.record_intermediate_variable("shifted_unique_probes", slice_psi)

            # Modulate wavefield.
            # Shape of slice_psi: (batch_size, n_modes, h, w)
            slice_patches = obj_patches[:, i_slice, ...]
            slice_psi = self.modulate_wavefield(slice_psi, slice_patches)

            # Propagate wavefield.
            if i_slice < self.parameter_group.object.n_slices - 1:
                slice_psi = self.propagate_to_next_slice(slice_psi, i_slice)
        self.record_intermediate_variable("psi", slice_psi)
        return slice_psi
    
    def modulate_wavefield(self, psi, slice_patches):
        """Modulate wavefield.
        
        This function uses complex multiplication with real-valued arguments
        and returns which is torch.compile friendly.

        Parameters
        ----------
        psi : Tensor
            A (batch_size, n_modes, h, w) tensor of incident wavefields.
        slice_patches : Tensor
            A (batch_size, h, w) tensor of object patches.

        Returns
        -------
        Tensor
            A (batch_size, n_modes, h, w) tensor of wavefields at the exiting plane.
        """
        return slice_patches[:, None, :, :] * psi

    @timer()
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
        probe = self.get_unique_probes(indices, always_return_probe_batch=self.apply_subpixel_shifts_on_probe)
        if self.apply_subpixel_shifts_on_probe:
            probe = self.shift_unique_probes(indices, probe, first_mode_only=True)
        self.record_intermediate_variable("shifted_unique_probes", probe)
        slice_psi = self.propagate_through_object(probe, obj_patches)
        return slice_psi

    @timer()
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
        psi_far = self.free_space_propagator.propagate_forward(psi)
        self.record_intermediate_variable("psi_far", psi_far)
        return psi_far

    @timer()
    def propagate_to_next_slice(self, psi: Tensor, slice_index: int):
        """
        Propagate wavefield to the next slice by the distance given by
        `self.object.slice_spacings.data[slice_index]`.

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
        self.in_object_prop_params = WavefieldPropagatorParameters.create_simple(
            wavelength_m=self.wavelength_m,
            width_px=self.probe.shape[-1],
            height_px=self.probe.shape[-2],
            pixel_width_m=self.object.pixel_size_m,
            pixel_height_m=self.object.pixel_size_m / self.object.options.pixel_size_aspect_ratio,
            propagation_distance_m=self.object.slice_spacings.data[slice_index],
        )
        self.in_object_propagator.update(self.in_object_prop_params)
        slice_psi_prop = self.in_object_propagator.propagate_forward(psi)
        return slice_psi_prop

    @timer()
    def propagate_to_previous_slice(self, psi: Tensor, slice_index: int):
        """
        Propagate wavefield to the previous slice by the distance given by
        `self.object.slice_spacings.data[slice_index - 1]`.

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
        self.in_object_prop_params = WavefieldPropagatorParameters.create_simple(
            wavelength_m=self.wavelength_m,
            width_px=self.probe.shape[-1],
            height_px=self.probe.shape[-2],
            pixel_width_m=self.object.pixel_size_m,
            pixel_height_m=self.object.pixel_size_m / self.object.options.pixel_size_aspect_ratio,
            propagation_distance_m=self.object.slice_spacings.data[slice_index - 1],
        )
        self.in_object_propagator.update(self.in_object_prop_params)
        slice_psi_prop = self.in_object_propagator.propagate_backward(psi)
        return slice_psi_prop
    
    def dip_generate(self):
        """Run embedded NN models in deep image prior parameters to generate data.
        """
        if isinstance(self.object, ptychi.data_structures.object.DIPObject):
            self.object.generate()
        if isinstance(self.probe, ptychi.data_structures.probe.DIPProbe):
            self.probe.generate()
        elif isinstance(self.probe, ptychi.data_structures.probe.SynthesisDictLearnProbe):
            self.probe.generate()

    @timer()
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
        indices = indices.to(self.object.tensor.data.device)
        self.intermediate_variables = self.PlanarPtychographyIntermediateVariables()
        
        self.dip_generate()
        
        positions = self.probe_positions.tensor[indices]
        obj_patches = self.extract_object_patches(indices)

        self.record_intermediate_variable("positions", positions)
        self.record_intermediate_variable("obj_patches", obj_patches)

        psi = self.forward_real_space(indices, obj_patches)
        psi_far = self.forward_far_field(psi)

        y = torch.abs(psi_far) ** 2
        # Sum along probe modes
        y = y.sum(1)
        
        y = self.conform_to_detector_size(y)

        returns = [y]
        if return_object_patches:
            returns.append(obj_patches)
        if len(returns) == 1:
            return returns[0]
        else:
            return returns

    @timer()        
    def conform_to_detector_size(self, y: Tensor) -> Tensor:
        if (
            self.detector_size is None
        ) or (
            self.detector_size[0] == y.shape[-2] and self.detector_size[1] == y.shape[-1]
        ):
            return y
        if self.detector_size[0] > y.shape[-2] or self.detector_size[1] > y.shape[-1]:
            raise ValueError(
                f"Detector size ({self.detector_size}) should not be larger than the probe size ({y.shape[-2:]})."
            )
        y = torch.fft.fftshift(y, dim=(-2, -1))
        y = ip.central_crop(y, self.detector_size)
        y = torch.fft.ifftshift(y, dim=(-2, -1))
        return y

    @timer()
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
        self.dip_generate()

        positions = self.probe_positions.tensor[indices]
        obj_patches = self.extract_object_patches(indices)

        self.record_intermediate_variable("positions", positions)
        self.record_intermediate_variable("obj_patches", obj_patches)

        probe = self.get_unique_probes(indices, always_return_probe_batch=self.apply_subpixel_shifts_on_probe)
        if self.apply_subpixel_shifts_on_probe:
            probe = self.shift_unique_probes(indices, probe, first_mode_only=True)
            self.record_intermediate_variable("shifted_unique_probes", probe)
        y = 0.0
        for i_mode in range(self.probe.n_modes):
            exit_psi = self.propagate_through_object(
                probe[..., i_mode : i_mode + 1, :, :], obj_patches
            )
            
            psi_far = self.free_space_propagator.propagate_forward(exit_psi)
            self.record_intermediate_variable("psi_far", psi_far)
            
            y = y + psi_far[..., 0, :, :].abs() ** 2

        y = self.conform_to_detector_size(y)

        # `self.intermediate_variables.shifted_unique_probes` is now a list of `n_modes * n_slices`
        # tensors. We concatenate the modes for each slice so that it becomes a list of `n_slices`
        # tensors.
        self.intermediate_variables.shifted_unique_probes = [
            torch.concat(self.intermediate_variables.shifted_unique_probes[i_mode::self.probe.n_modes], dim=-3)
            for i_mode in range(self.probe.n_modes)
        ]
        
        returns = [y]
        if return_object_patches:
            returns.append(obj_patches)
        if len(returns) == 1:
            return returns[0]
        else:
            return returns

    @timer()
    def post_differentiation_hook(self, indices, y_true, **kwargs):
        self.compensate_for_fft_scaler()
        self.scale_gradients(y_true)

    @timer()        
    def compensate_for_fft_scaler(self):
        """
        Compensate for the scaling of the FFT. If the normalization of FFT is "ortho",
        nothing is done. If the normalization is "backward" (default in `torch.fft`),
        all gradients are scaled by 1 / sqrt(N).
        """
        if (
            isinstance(self.object, ptychi.data_structures.object.DIPObject) or 
            isinstance(self.probe, ptychi.data_structures.probe.DIPProbe)
        ):
            return
        if not isinstance(self.free_space_propagator, FourierPropagator):
            return
        if self.free_space_propagator.norm == "backward" or self.free_space_propagator.norm is None:
            factor = self.probe.shape[-2] * self.probe.shape[-1]
            for var in self.optimizable_parameters:
                if var.get_grad() is not None:
                    var.set_grad(var.get_grad() / factor)

    @timer()
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
        
        For probe positions and OPR weights, we just compensate for the mean reduction:
        (1) multiply it by h * w.
        """
        if (
            isinstance(self.object, ptychi.data_structures.object.DIPObject) or 
            isinstance(self.probe, ptychi.data_structures.probe.DIPProbe)
        ):
            return
        # Directly modify the gradients here. Tensor.register_hook has memory leak issue.
        if self.object.optimizable:
            self.object.set_grad(
                self.object.get_grad()
                / self.probe.get_all_mode_intensity().max()
                * patterns.numel()
            )
        # Assuming (obj_patches.abs() ** 2).max() == 1.0
        if self.probe.optimizable:
            self.probe.set_grad(
                self.probe.get_grad() * (
                    patterns.numel() / len(patterns)
                )
            )
            
        if self.probe_positions.optimizable:
            self.probe_positions.set_grad(
                self.probe_positions.get_grad() * (
                    patterns.numel() / len(patterns)
                )
            )
            
        if self.opr_mode_weights.optimizable:
            self.opr_mode_weights.set_grad(
                self.opr_mode_weights.get_grad() * (
                    patterns.numel() / len(patterns)
                )
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

    @timer()
    def conform_to_exit_wave_size(
        self, 
        y_pred: Tensor, 
        y_true: Tensor, 
        valid_pixel_mask: Tensor,
        psi_shape: tuple[int, int], 
    ) -> tuple[Tensor, Tensor, Tensor]:
        if psi_shape[-2] != y_pred.shape[-2] or psi_shape[-1] != y_pred.shape[-1]:
            y_pred = torch.fft.fftshift(y_pred, dim=(-2, -1))
            y_true = torch.fft.fftshift(y_true, dim=(-2, -1))
            y_pred = ip.central_crop_or_pad(y_pred, psi_shape[-2:])
            y_true = ip.central_crop_or_pad(y_true, psi_shape[-2:])
            y_pred = torch.fft.ifftshift(y_pred, dim=(-2, -1))
            y_true = torch.fft.ifftshift(y_true, dim=(-2, -1))
            if valid_pixel_mask is not None:
                valid_pixel_mask = torch.fft.fftshift(valid_pixel_mask, dim=(-2, -1))
                valid_pixel_mask = ip.central_crop_or_pad(valid_pixel_mask, psi_shape[-2:])
                valid_pixel_mask = torch.fft.ifftshift(valid_pixel_mask, dim=(-2, -1))
        return y_pred, y_true, valid_pixel_mask


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

    @timer()
    def backward_to_psi_far(self, y_pred, y_true, psi_far):
        """
        Compute the gradient of the NLL with respect to far field wavefront:
        $g = \frac{\partial L}{\partial \psi_{far}}$.
        
        When `self.valid_pixel_mask` is not None, pixels of the gradient `g` where the
        mask is False are set to 0. When `g` is used to update the far-field wavefield
        `psi_far`, the invalid pixels are kept unchanged.
        
        Parameters
        ----------
        y_pred : Tensor
            A (batch_size, h, w) tensor of predicted intensities.
        y_true : Tensor
            A (batch_size, h, w) tensor of true intensities.
        psi_far : Tensor
            A (batch_size, n_probe_modes, h, w) tensor of far field wavefronts.

        Returns
        -------
        Tensor
            A (batch_size, n_probe_modes, h, w) tensor giving the adjoint of 
            the far field wavefront.
        """
        y_pred, y_true, valid_pixel_mask = self.conform_to_exit_wave_size(
            y_pred, y_true, self.valid_pixel_mask, psi_shape=psi_far.shape[-2:]
        )
        g = 1 - torch.sqrt(y_true) / (torch.sqrt(y_pred) + self.eps)  # Eq. 12b
        if valid_pixel_mask is not None:
            g[:, torch.logical_not(valid_pixel_mask)] = 0
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

    @timer()
    def backward_to_psi_far(self, y_pred: Tensor, y_true: Tensor, psi_far: Tensor):
        """
        Compute the gradient of the NLL with respect to far field wavefront.
        """
        y_pred, y_true, valid_pixel_mask = self.conform_to_exit_wave_size(
            y_pred, y_true, self.valid_pixel_mask, psi_far.shape[-2:]
        )
        g = 1 - y_true / (y_pred + self.eps)  # Eq. 12b
        if valid_pixel_mask is not None:
            g[:, torch.logical_not(valid_pixel_mask)] = 0
        g = g[:, None, :, :] * psi_far
        return g
