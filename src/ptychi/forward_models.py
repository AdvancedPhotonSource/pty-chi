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
from ptychi.utils import move_nested_tensors_to_device

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

    def clear_resampling_caches(self):
        """Clear derived grid caches, if the forward model defines any."""
        return None

    def record_intermediate_variable(self, name, var):
        if self.retain_intermediates:
            if isinstance(self.intermediate_variables[name], list):
                self.intermediate_variables[name].append(var.detach())
            else:
                self.intermediate_variables[name] = var.detach()

    def move_intermediate_variables_to_device(self, device):
        inter = self.intermediate_variables
        if inter is None:
            return
        if isinstance(device, str):
            device = torch.device(device)

        if dataclasses.is_dataclass(inter):
            for field in dataclasses.fields(inter):
                value = getattr(inter, field.name)
                if value is None:
                    continue
                setattr(inter, field.name, move_nested_tensors_to_device(value, device))
        elif isinstance(inter, dict):
            items = inter.items()
            for key, value in list(items):
                if value is None:
                    continue
                inter[key] = move_nested_tensors_to_device(value, device)
        else:
            for key, value in vars(inter).items():
                if value is None:
                    continue
                setattr(inter, key, move_nested_tensors_to_device(value, device))


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
        diffraction_pattern_blur_sigma: Optional[float] = None,
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
        diffraction_pattern_blur_sigma : float, optional
            The sigma of the Gaussian blur applied to the simulated diffraction patterns.
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

        # Resampling caches are ordinary Python attributes on purpose: they are
        # read-only derived values and must not appear in state_dict().  Each
        # cache keeps only its newest entry on a given device.
        self._resampling_caches = {
            "object": {},
            "preconditioner": {},
            "positions": {},
        }
        self._resampling_geometry_key = None
        self._resampling_geometry = None

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
        
        self.diffraction_pattern_blur_sigma = diffraction_pattern_blur_sigma
        
        self.check_inputs()

    @property
    def probe_pixel_size_m(self) -> float:
        value = self.probe.options.pixel_size_m
        return self.object.pixel_size_m if value is None else value

    @property
    def probe_pixel_size_aspect_ratio(self) -> float:
        value = self.probe.options.pixel_size_aspect_ratio
        return self.object.options.pixel_size_aspect_ratio if value is None else value

    @property
    def probe_pixel_height_m(self) -> float:
        return self.probe_pixel_size_m / self.probe_pixel_size_aspect_ratio

    @property
    def object_pixel_height_m(self) -> float:
        return self.object.pixel_size_m / self.object.options.pixel_size_aspect_ratio

    @property
    def probe_grid_object_shape(self) -> tuple[int, int]:
        native_h, native_w = self.object.lateral_shape
        return (
            max(1, round(native_h * self.object_pixel_height_m / self.probe_pixel_height_m)),
            max(1, round(native_w * self.object.pixel_size_m / self.probe_pixel_size_m)),
        )

    @property
    def probe_grid_scale(self) -> tuple[float, float]:
        target_h, target_w = self.probe_grid_object_shape
        native_h, native_w = self.object.lateral_shape
        return target_h / native_h, target_w / native_w

    @property
    def resampling_enabled(self) -> bool:
        # Pixel geometry, rather than rounded shape alone, defines the identity
        # fast path. This preserves exact legacy behavior for inherited grids.
        return not (
            self.probe_pixel_size_m == self.object.pixel_size_m
            and self.probe_pixel_height_m == self.object_pixel_height_m
        )

    def _current_geometry_key(self):
        return (
            tuple(self.object.lateral_shape),
            self.probe_grid_object_shape,
            self.object.pixel_size_m,
            self.object.options.pixel_size_aspect_ratio,
            self.probe_pixel_size_m,
            self.probe_pixel_size_aspect_ratio,
            tuple(self.probe.get_spatial_shape()),
        )

    def clear_resampling_caches(self):
        if hasattr(self, "_resampling_caches"):
            for cache in self._resampling_caches.values():
                cache.clear()
        self._resampling_geometry_key = self._current_geometry_key()
        native_shape = tuple(self.object.lateral_shape)
        target_shape = self.probe_grid_object_shape
        source_slices = []
        target_slices = []
        for source_size, target_size in zip(native_shape, target_shape):
            overlap = min(source_size, target_size)
            source_start = source_size // 2 - overlap // 2
            target_start = target_size // 2 - overlap // 2
            source_slices.append(slice(source_start, source_start + overlap))
            target_slices.append(slice(target_start, target_start + overlap))
        native_numel = native_shape[0] * native_shape[1]
        target_numel = target_shape[0] * target_shape[1]
        self._resampling_geometry = {
            "native_shape": native_shape,
            "target_shape": target_shape,
            "axis_scales": self.probe_grid_scale,
            "forward_normalization": target_numel / native_numel,
            "normalized_adjoint_normalization": native_numel / target_numel,
            "source_slices": tuple(source_slices),
            "target_slices": tuple(target_slices),
        }

    @property
    def resampling_geometry(self) -> dict:
        """Read-only static geometry used by the probe-grid transforms."""
        self._ensure_current_geometry()
        return self._resampling_geometry

    def _ensure_current_geometry(self):
        geometry_key = self._current_geometry_key()
        if self._resampling_geometry_key != geometry_key:
            self.clear_resampling_caches()
            # Propagators contain pixel-geometry-dependent state.
            if hasattr(self, "free_space_propagator"):
                self.build_free_space_propagator()
            if hasattr(self, "in_object_propagator"):
                self.build_in_object_propagator()

    def _apply(self, fn):
        result = super()._apply(fn)
        self.clear_resampling_caches()
        return result

    @staticmethod
    def _tensor_version_key(tensor: Tensor):
        try:
            storage_identity = tensor.untyped_storage().data_ptr()
        except AttributeError:
            storage_identity = tensor.storage().data_ptr()
        return storage_identity, tensor._version

    def _cache_value(self, kind, source_tensors, value_factory, *, dtype, device):
        """Return a version-keyed cached value, respecting autograd graph lifetime.

        Parameters
        ----------
        kind : str
            Cache namespace in ``self._resampling_caches``, such as ``"object"``,
            ``"preconditioner"``, or ``"positions"``.
        source_tensors : Iterable[Tensor]
            Backing tensors on which the derived value depends. Their storage
            identities and PyTorch mutation versions are included in the cache key.
        value_factory : Callable[[], Any]
            Zero-argument callable that computes the derived value after a cache miss.
        dtype : torch.dtype
            Data type of the derived value, included in the cache key.
        device : torch.device or str
            Device on which the derived value resides. Each cache namespace retains
            only its latest entry for a given device.

        Returns
        -------
        Any
            The cached value, or the newly computed value after a cache miss.

        Notes
        -----
        If gradient tracking is enabled and any source tensor requires gradients,
        the value is recomputed and not cached so an earlier autograd graph is never
        reused.
        """
        self._ensure_current_geometry()
        device = torch.device(device)
        sources = tuple(source_tensors)
        key = (
            tuple(self._tensor_version_key(source) for source in sources),
            str(device),
            dtype,
            self._current_geometry_key(),
        )
        graph_must_be_fresh = torch.is_grad_enabled() and any(
            source.requires_grad for source in sources
        )
        device_cache = self._resampling_caches[kind]
        if not graph_must_be_fresh:
            cached = device_cache.get(str(device))
            if cached is not None and cached[0] == key:
                return cached[1]
        value = value_factory()
        if not graph_must_be_fresh:
            device_cache[str(device)] = (key, value)
        return value

    def get_probe_grid_object(self) -> Tensor:
        """Return the complete multislice object represented on the probe grid."""
        self._ensure_current_geometry()
        data = self.object.data
        if not self.resampling_enabled:
            return data
        backing = self.object.tensor.data
        return self._cache_value(
            "object",
            (backing,),
            lambda: ip.fourier_resize(data, self.probe_grid_object_shape),
            dtype=data.dtype,
            device=data.device,
        )

    def get_probe_grid_preconditioner(self) -> Optional[Tensor]:
        """Return a cached real, nonnegative copy of the native preconditioner."""
        preconditioner = self.object.preconditioner
        if preconditioner is None or not self.resampling_enabled:
            return preconditioner
        return self._cache_value(
            "preconditioner",
            (preconditioner,),
            lambda: ip.fourier_resize(
                preconditioner, self.probe_grid_object_shape
            ).real.clamp_min_(0),
            dtype=preconditioner.dtype,
            device=preconditioner.device,
        )

    def get_probe_grid_position_data(self):
        """Return full transformed centers, snapped centers, and shift residuals."""
        positions = self.probe_positions.tensor
        origin = self.object.pos_origin_coords
        if not self.resampling_enabled:
            absolute = positions + origin
            snapped = positions.round() + origin
            residual = positions - positions.round()
            return absolute, snapped, residual

        def transform():
            scale = positions.new_tensor(self.probe_grid_scale)
            absolute = (positions + origin) * scale
            parity = positions.new_tensor(
                [((size - 1) / 2) % 1 for size in self.probe.get_spatial_shape()]
            )
            snapped = (absolute - parity).round() + parity
            return absolute, snapped, absolute - snapped

        return self._cache_value(
            "positions",
            (positions, origin),
            transform,
            dtype=positions.dtype,
            device=positions.device,
        )

    def transform_positions(self, positions: Tensor, *, absolute: bool = False) -> Tensor:
        """Map object-pixel positions to probe-grid pixels without modifying inputs."""
        if not self.resampling_enabled:
            return positions + self.object.pos_origin_coords if absolute else positions
        scale = positions.new_tensor(self.probe_grid_scale)
        if absolute:
            return (positions + self.object.pos_origin_coords) * scale
        return positions * scale

    def probe_displacements_to_object_pixels(self, displacements: Tensor) -> Tensor:
        if not self.resampling_enabled:
            return displacements
        return displacements / displacements.new_tensor(self.probe_grid_scale)

    def _snap_probe_grid_centers(self, positions: Tensor, patch_shape) -> Tensor:
        if not self.resampling_enabled:
            return positions.round() + self.object.pos_origin_coords
        centers = self.transform_positions(positions, absolute=True)
        parity = centers.new_tensor([((size - 1) / 2) % 1 for size in patch_shape])
        return (centers - parity).round() + parity

    def place_object_patches_on_probe_grid(
        self, positions: Tensor, patches: Tensor, *, integer_mode: bool = True
    ) -> Tensor:
        """Place patches on an empty probe-grid object buffer."""
        if integer_mode:
            centers = self._snap_probe_grid_centers(positions, patches.shape[-2:])
        else:
            centers = self.transform_positions(positions, absolute=True)
        image = torch.zeros(
            self.probe_grid_object_shape, dtype=patches.dtype, device=patches.device
        )
        if integer_mode:
            return ip.place_patches_integer(image, centers, patches, op="add")
        return self.object.place_patches_function(
            image, centers, patches, op="add", pad=self.pad_for_shift
        )

    def extract_probe_grid_patches(
        self, image: Tensor, positions: Tensor, patch_shape=None, *, integer_mode: bool = True
    ) -> Tensor:
        """Extract patches from a probe-grid buffer at object-domain positions."""
        if patch_shape is None:
            patch_shape = self.probe.get_spatial_shape()
        if integer_mode:
            centers = self._snap_probe_grid_centers(positions, patch_shape)
            return ip.extract_patches_integer(image, centers, patch_shape)
        centers = self.transform_positions(positions, absolute=True)
        return self.object.extract_patches_function(
            image, centers, patch_shape, pad=self.pad_for_shift
        )

    def object_update_to_native_grid(self, update: Tensor, *, normalized: bool = False) -> Tensor:
        """Apply the resize adjoint (or normalized pseudoinverse) to an update."""
        if not self.resampling_enabled:
            return update
        return ip.fourier_resize(
            update,
            tuple(self.object.lateral_shape),
            adjoint=True,
            normalized_adjoint=normalized,
        )

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
            pixel_width_m=self.probe_pixel_size_m,
            pixel_height_m=self.probe_pixel_height_m,
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
                pixel_width_m=self.probe_pixel_size_m,
                pixel_height_m=self.probe_pixel_height_m,
                propagation_distance_m=self.free_space_propagation_distance_m,
            )
            # TODO: AngularSpectrumPropagator uses analytical transfer function. Using the FFT
            # of real-space impulse response might offer better sampling for large propagation distances.
            if params.is_fresnel_transform_preferrable():
                self.free_space_propagator = FresnelTransformPropagator(params)
            else:
                self.free_space_propagator = AngularSpectrumPropagator(params)
                
    def extract_object_patches(self, indices: Tensor) -> Tensor:
        object_on_probe_grid = self.get_probe_grid_object()
        absolute, snapped, _ = self.get_probe_grid_position_data()
        absolute_positions = (
            snapped[indices] if self.apply_subpixel_shifts_on_probe else absolute[indices]
        )
        # PlanarObject.extract_patches expects positions relative to its configured
        # origin. Offset the probe-grid absolute centers so adding that origin in
        # the object method recovers the intended centers in object_on_probe_grid.
        positions = absolute_positions - self.object.pos_origin_coords
        return self.object.extract_patches(
            positions,
            self.probe.get_spatial_shape(),
            integer_mode=self.apply_subpixel_shifts_on_probe,
            pad_for_shift=self.pad_for_shift,
            object_array=object_on_probe_grid,
        )

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
        fractional_shifts = self.get_probe_grid_position_data()[2][indices]
        
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
            slice_psi = slice_patches[:, None, :, :] * slice_psi

            # Propagate wavefield.
            if i_slice < self.parameter_group.object.n_slices - 1:
                slice_psi = self.propagate_to_next_slice(slice_psi, i_slice)
        self.record_intermediate_variable("psi", slice_psi)
        return slice_psi

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
            pixel_width_m=self.probe_pixel_size_m,
            pixel_height_m=self.probe_pixel_height_m,
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
            pixel_width_m=self.probe_pixel_size_m,
            pixel_height_m=self.probe_pixel_height_m,
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
            
    def blur_diffraction_patterns(self, y: Tensor) -> Tensor:
        """
        Blur the diffraction patterns.
        """
        if self.diffraction_pattern_blur_sigma is None:
            return y
        y = ip.gaussian_filter(
            y, 
            sigma=self.diffraction_pattern_blur_sigma,
            size=round(self.diffraction_pattern_blur_sigma * 10) + 1
        )
        return y

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
        y = self.blur_diffraction_patterns(y)

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
        y = self.blur_diffraction_patterns(y)

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
    def __init__(
        self, 
        eps: float = 1e-6, 
        valid_pixel_mask: Optional[Tensor] = None, 
        exclude_measured_pixels_below: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.noise_statistics = None
        self.valid_pixel_mask = valid_pixel_mask
        self.exclude_measured_pixels_below = exclude_measured_pixels_below
        
    def nll(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Calculate the negative log-likelihood.
        """
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_constrained_pixel_mask(
        valid_pixel_mask: Optional[Tensor],
        exclude_measured_pixels_below: Optional[float],
        y_true: Tensor,
    ) -> Tensor:
        constrained_pixel_mask = torch.ones_like(y_true, dtype=torch.bool)
        if valid_pixel_mask is not None:
            constrained_pixel_mask = valid_pixel_mask.to(y_true.device)
            constrained_pixel_mask = constrained_pixel_mask.unsqueeze(0).expand(
                y_true.shape[0], -1, -1
            )
        if exclude_measured_pixels_below is not None:
            constrained_pixel_mask = torch.logical_and(
                constrained_pixel_mask,
                y_true > exclude_measured_pixels_below,
            )
        return constrained_pixel_mask

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
        mask is False are set to 0. When `self.exclude_measured_pixels_below` is not
        None, gradients at pixels with measured intensities less than or equal to that
        threshold are also set to 0.
        """
        # Shape of g:       (batch_size, h, w)
        # Shape of psi_far: (batch_size, n_probe_modes, h, w)
        y_pred, y_true, valid_pixel_mask = self.conform_to_exit_wave_size(
            y_pred, y_true, self.valid_pixel_mask, psi_far.shape[-2:]
        )
        constrained_pixel_mask = self.get_constrained_pixel_mask(
            valid_pixel_mask,
            self.exclude_measured_pixels_below,
            y_true,
        )
        g = 1 - torch.sqrt(y_true) / (torch.sqrt(y_pred) + self.eps)  # Eq. 12b

        g[torch.logical_not(constrained_pixel_mask)] = 0
            
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
        constrained_pixel_mask = self.get_constrained_pixel_mask(
            valid_pixel_mask,
            self.exclude_measured_pixels_below,
            y_true,
        )
        g = 1 - y_true / (y_pred + self.eps)  # Eq. 12b
        g[torch.logical_not(constrained_pixel_mask)] = 0
        g = g[:, None, :, :] * psi_far
        return g
