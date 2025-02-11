from typing import Tuple, TYPE_CHECKING, Optional
import logging
import copy
import math

import torch
from torch import Tensor

import ptychi.image_proc as ip
import ptychi.data_structures as ds
import ptychi.data_structures.base as dsbase
import ptychi.maths as pmath
from ptychi.timing.timer_utils import timer
from ptychi.utils import (
    get_default_complex_dtype, 
    to_tensor, 
    to_numpy, 
    chunked_processing,
    get_probe_renormalization_factor
)
import ptychi.maps as maps

if TYPE_CHECKING:
    import ptychi.api as api
    from ptychi.data_structures.probe_positions import ProbePositions
    
logger = logging.getLogger(__name__)


class Object(dsbase.ReconstructParameter):
    options: "api.options.base.ObjectOptions"

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
        self.roi_bbox: dsbase.BoundingBox = None
        
        center_pixel = torch.tensor(self.shape, device=torch.get_default_device()) / 2.0
        center_pixel = center_pixel.round() + 0.5
        self.register_buffer("center_pixel", center_pixel)

    def extract_patches(self, positions, patch_shape, *args, **kwargs):
        raise NotImplementedError

    def place_patches(self, positions, patches, *args, **kwargs):
        raise NotImplementedError

    @timer()
    def constrain_l1_norm(self):
        if self.options.l1_norm_constraint.weight <= 0:
            return
        data = self.data
        l1_grad = torch.sgn(data)
        data = data - self.options.l1_norm_constraint.weight * l1_grad
        self.set_data(data)
        logger.debug("L1 norm constraint applied to object.")

    def constrain_smoothness(self):
        raise NotImplementedError

    def constrain_total_variation(self) -> None:
        raise NotImplementedError

    def remove_grid_artifacts(self, *args, **kwargs):
        raise NotImplementedError
    
    def build_roi_bounding_box(self, positions: "ProbePositions"):
        pos = positions.data
        self.roi_bbox = dsbase.BoundingBox(
            sy=pos[:, 0].min(),
            ey=pos[:, 0].max(),
            sx=pos[:, 1].min(),
            ex=pos[:, 1].max(),
            origin=tuple(to_numpy(self.center_pixel)),
        )
    
    def get_object_in_roi(self):
        raise NotImplementedError
    
    def update_preconditioner(self):
        raise NotImplementedError
    
    def initialize_preconditioner(self):
        raise NotImplementedError
    
    def remove_object_probe_ambiguity(self):
        raise NotImplementedError


class PlanarObject(Object):
    """
    Object that consists of one or multiple planes, i.e., 2D object or
    multislice object. The object is stored in a (n_slices, h , w) tensor.
    """

    def __init__(
        self,
        name: str = "object",
        options: "api.options.base.ObjectOptions" = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(name=name, options=options, *args, **kwargs)

        if len(self.shape) != 3:
            raise ValueError("PlanarObject should have a shape of (n_slices, h, w).")
        if (
            self.options.slice_spacings_m is not None
            and len(self.options.slice_spacings_m) != self.n_slices - 1
            and self.n_slices > 1
        ):
            raise ValueError("The number of slice spacings must be n_slices - 1.")

        if self.is_multislice:
            if self.options.slice_spacings_m is None:
                raise ValueError("slice_spacings_m must be specified for multislice objects.")
            self.register_buffer("slice_spacings_m", to_tensor(self.options.slice_spacings_m))
        else:
            self.slice_spacing_m = None

        center_pixel = torch.tensor(self.shape[1:], device=torch.get_default_device()) / 2.0
        center_pixel = center_pixel.round() + 0.5
        self.register_buffer("center_pixel", center_pixel)

    @property
    def is_multislice(self) -> bool:
        return self.shape[0] > 1

    @property
    def n_slices(self):
        return self.shape[0]

    @property
    def lateral_shape(self):
        return self.shape[1:]

    @property
    def extract_patches_function(self) -> "ip.ExtractPatchesProtocol":
        return maps.get_patch_extractor_function_by_name(self.options.patch_interpolation_method)

    @property
    def place_patches_function(self) -> "ip.PlacePatchesProtocol":
        return maps.get_patch_placer_function_by_name(self.options.patch_interpolation_method)

    def get_slice(self, index):
        return self.data[index, ...]

    @timer()
    def extract_patches(
        self, 
        positions: Tensor, 
        patch_shape: Tuple[int, int], 
        integer_mode: bool = False,
    ):
        """
        Extract (n_patches, n_slices, h', w') patches from the object.

        Parameters
        ----------
        positions : Tensor
            Tensor of shape (N, 2) giving the center positions of the patches in pixels.
            The origin of the given positions are assumed to be `self.center_pixel`.
        patch_shape : tuple
            Tuple giving the lateral patch shape in pixels.
        integer_mode : bool, optional
            If True, the patches are extracted at the exact center between pixels,
            so that no interpolation is needed.

        Returns
        -------
        Tensor
            Tensor of shape (N, n_slices, h', w') containing the extracted patches.
        """
        # Positions are provided with the origin in the center of the object support.
        # We shift the positions so that the origin is in the upper left corner.
        positions = positions + self.center_pixel
        patches_all_slices = []
        for i_slice in range(self.n_slices):
            if integer_mode:
                patches = ip.extract_patches_integer(
                    self.get_slice(i_slice), positions, patch_shape
                )
            else:
                patches = self.extract_patches_function(
                    self.get_slice(i_slice), positions, patch_shape
                )
            patches_all_slices.append(patches)
        patches_all_slices = torch.stack(patches_all_slices, dim=1)
        return patches_all_slices

    @timer()
    def place_patches(
        self, 
        positions: Tensor, 
        patches: Tensor, 
        integer_mode: bool = False,
        *args, **kwargs
    ):
        """
        Place patches into the object.

        Parameters
        ----------
        positions : Tensor
            Tensor of shape (N, 2) giving the center positions of the patches in pixels.
            The origin of the given positions are assumed to be `self.center_pixel`.
        patches : Tensor
            Tensor of shape (n_patches, n_slices, H, W) of image patches.
        integer_mode : bool, optional
            If True, the patches are placed at the exact center between pixels,
            so that no interpolation is needed.
        """
        positions = positions + self.center_pixel
        updated_slices = []
        for i_slice in range(self.n_slices):
            if integer_mode:
                image = ip.place_patches_integer(
                    self.get_slice(i_slice), positions, patches[:, i_slice, ...], op="add"
                )
            else:
                image = self.place_patches_function(
                    self.get_slice(i_slice), positions, patches[:, i_slice, ...], op="add"
                )
            updated_slices.append(image)
        updated_slices = torch.stack(updated_slices, dim=0)
        self.tensor.set_data(updated_slices)

    @timer()
    def place_patches_on_empty_buffer(
        self, 
        positions: Tensor, 
        patches: Tensor, 
        integer_mode: bool = False,
        *args, **kwargs
    ):
        """
        Place patches into a zero array with the *lateral* shape of the object.

        Parameters
        ----------
        positions : Tensor
            Tensor of shape (N, 2) giving the center positions of the patches in pixels.
            The origin of the given positions are assumed to be `self.center_pixel`.
        patches : Tensor
            Tensor of shape (N, H, W) of image patches.
        integer_mode : bool, optional
            If True, the patches are placed at the exact center between pixels,
            so that no interpolation is needed.

        Returns
        -------
        image : Tensor
            A tensor with the lateral shape of the object with patches added onto it.
        """
        positions = positions + self.center_pixel
        image = torch.zeros(
            self.lateral_shape, dtype=get_default_complex_dtype(), device=self.tensor.data.device
        )
        if integer_mode:
            image = ip.place_patches_integer(image, positions, patches, op="add")
        else:
            image = self.place_patches_function(image, positions, patches, op="add")
        return image
    
    def get_object_in_roi(self):
        bbox = self.roi_bbox.get_bbox_with_top_left_origin()
        return self.data[:, int(bbox.sy):int(bbox.ey), int(bbox.sx):int(bbox.ex)]

    @timer()
    def constrain_smoothness(self) -> None:
        """
        Smooth the magnitude of the object.
        """
        if self.options.smoothness_constraint.alpha <= 0:
            return
        alpha = self.options.smoothness_constraint.alpha
        if alpha > 1.0 / 8:
            logger.warning(f"Alpha = {alpha} in smoothness constraint should be less than 1/8.")
        psf = torch.ones(3, 3, device=self.device) * alpha
        psf[2, 2] = 1 - 8 * alpha

        data = self.data
        for i_slice in range(self.n_slices):
            mag = data[i_slice].abs()
            mag = ip.convolve2d(mag, psf, "same")
            data[i_slice] = data[i_slice] / data[i_slice].abs() * mag
        self.set_data(data)

    @timer()
    def constrain_total_variation(self) -> None:
        if self.options.total_variation.weight <= 0:
            return
        data = self.data
        for i_slice in range(self.n_slices):
            data[i_slice] = ip.total_variation_2d_chambolle(
                data[i_slice], lmbda=self.options.total_variation.weight, niter=2
            )
        self.set_data(data)

    @timer()
    def remove_grid_artifacts(self):
        data = self.data
        for i_slice in range(self.n_slices):
            phase = torch.angle(data[i_slice])
            phase = ip.remove_grid_artifacts(
                phase,
                pixel_size_m=self.pixel_size_m,
                period_x_m=self.options.remove_grid_artifacts.period_x_m,
                period_y_m=self.options.remove_grid_artifacts.period_y_m,
                window_size=self.options.remove_grid_artifacts.window_size,
                direction=self.options.remove_grid_artifacts.direction,
            )
            data = data[i_slice].abs() * torch.exp(1j * phase)
        self.set_data(data)

    @timer()
    def regularize_multislice(self):
        """
        Regularize multislice by applying a low-pass transfer function to the
        3D Fourier space of the magnitude and phase (unwrapped) of all slices.

        Adapted from fold_slice (regulation_multilayers.m).
        """
        if not self.is_multislice or self.options.multislice_regularization.weight <= 0:
            return
        if self.preconditioner is None:
            raise ValueError("Regularization requires a preconditioner.")

        # TODO: use CPU if GPU memory usage is too large.
        fourier_coords = []
        for s in (self.n_slices, self.lateral_shape[0], self.lateral_shape[1]):
            u = torch.fft.fftfreq(s)
            fourier_coords.append(u)
        fourier_coords = torch.meshgrid(*fourier_coords, indexing="ij")
        # Calculate force of regularization based on the idea that DoF = resolution^2/lambda
        w = 1 - torch.atan(
            (
                self.options.multislice_regularization.weight
                * torch.abs(fourier_coords[0])
                / torch.sqrt(fourier_coords[1] ** 2 + fourier_coords[2] ** 2 + 1e-3)
            )
            ** 2
        ) / (torch.pi / 2)
        relax = 1
        alpha = 1
        # Low-pass transfer function.
        w_a = w * torch.exp(-alpha * (fourier_coords[1] ** 2 + fourier_coords[2] ** 2))
        obj = self.data

        # Find correction for amplitude.
        aobj = torch.abs(obj)
        fobj = torch.fft.fftn(aobj)
        fobj = fobj * w_a
        aobj_upd = torch.fft.ifftn(fobj)
        # Push towards zero.
        aobj_upd = 1 + 0.9 * (aobj_upd - 1)

        # Find correction for phase.
        w_phase = torch.clip(10 * (self.preconditioner / self.preconditioner.max()), max=1)

        if self.options.multislice_regularization.unwrap_phase:
            pobj = [
                ip.unwrap_phase_2d(
                    obj[i_slice],
                    weight_map=w_phase,
                    fourier_shift_step=0.5,
                    image_grad_method=self.options.multislice_regularization.unwrap_image_grad_method,
                    image_integration_method=self.options.multislice_regularization.unwrap_image_integration_method,
                )
                for i_slice in range(self.n_slices)
            ]
            pobj = torch.stack(pobj, dim=0)
        else:
            pobj = torch.angle(obj)
        fobj = torch.fft.fftn(pobj)
        fobj = fobj * w_a
        pobj_upd = torch.fft.ifftn(fobj)

        aobj_upd = torch.real(aobj_upd) - aobj
        pobj_upd = w_phase * (torch.real(pobj_upd) - pobj)
        corr = (1 + relax * aobj_upd) * torch.exp(1j * relax * pobj_upd)
        obj = obj * corr
        self.set_data(obj)
        
    def remove_object_probe_ambiguity(
        self, 
        probe: "ds.probe.Probe", 
        *, 
        update_probe_in_place: bool = True
    ) -> Optional[Tensor]:
        """
        Remove the object-probe ambiguity by scaling the object by its norm,
        and adjusting the probe power accordingly.
        
        Parameters
        ----------
        probe : ds.probe.Probe
            The probe to adjust.
        update_probe_in_place : bool, optional
            Whether to update the probe in place. If False, the tensor of the updated probe is returned.
        """
        bbox = self.roi_bbox.get_bbox_with_top_left_origin()
        roi_slicer = bbox.get_slicer()
        w = self.preconditioner[*roi_slicer]
        w = w / pmath.mnorm(w, dim=(-2, -1))
        
        # Get the norm of the object within the ROI for each slice.
        obj_data = self.data
        obj_norm = torch.sqrt(torch.mean(torch.abs(obj_data[..., *roi_slicer]) ** 2 * w, dim=(-2, -1)))
        
        # Scale the object such that the mean transmission is 1.
        obj_data = obj_data / obj_norm[:, None, None]
        
        # Adjust the probe power accordingly.
        probe_data = probe.data
        probe_data = probe_data * torch.prod(obj_norm)
        
        self.set_data(obj_data)
        if update_probe_in_place:
            probe.set_data(probe_data)
        else:
            return probe_data
        
        
    def calculate_illumination_map(
        self, 
        probe: "ds.probe.Probe",
        probe_positions: "ds.probe_positions.ProbePositions",
        use_all_modes: bool = False
    ) -> Tensor:
        """Calculate the illumination map by overlaying the probe intensity
        at all positions on a zero array. The map is used to calculate 
        the preconditioner.
        
        Parameters
        ----------
        probe : ds.probe.Probe
            The probe to use for the illumination map.
        probe_positions : ds.probe_positions.ProbePositions
            The positions of the probe.
        use_all_modes : bool, optional
            Whether to use all modes of the probe.

        Returns
        -------
        Tensor
            The illumination map of the object.
        """
        positions_all = probe_positions.tensor
        # Shape of probe:        (n_probe_modes, h, w)
        object_ = self.get_slice(0)

        if use_all_modes:
            probe_int = probe.get_all_mode_intensity(opr_mode=0)[None, :, :]
        else:
            probe_int = probe.get_mode_and_opr_mode(mode=0, opr_mode=0)[None, ...].abs() ** 2
        # Shape of probe_int:    (n_scan_points, h, w)
        probe_int = probe_int.repeat(len(positions_all), 1, 1)

        # Stitch probes of all positions on the object buffer
        # TODO: allow setting chunk size externally
        probe_sq_map = chunked_processing(
            func=ip.place_patches_integer,
            common_kwargs={"op": "add"},
            chunkable_kwargs={
                "positions": positions_all.round().int() + self.center_pixel,
                "patches": probe_int,
            },
            iterated_kwargs={
                "image": torch.zeros_like(object_.real).type(torch.get_default_dtype())
            },
            chunk_size=64,
        )
        return probe_sq_map

    def update_preconditioner(
        self,
        probe: "ds.probe.Probe",
        probe_positions: "ds.probe_positions.ProbePositions",
        patterns: Tensor = None,
        use_all_modes: bool = False,
    ) -> None:
        """Update the preconditioner. This function reproduces the behavior
        of PtychoShelves in `ptycho_solver`: it averages the new illumination
        map and the old preconditioner to reduce oscillations.
        
        Parameters
        ----------
        probe : ds.probe.Probe
            The probe to use for the illumination map.
        probe_positions : ds.probe_positions.ProbePositions
            The positions of the probe.
        patterns : Tensor, optional
            A (n_scan_points, h, w) tensor giving the diffraction patterns. Only needed
            if the preconditioner does not exist and needs to be initialized.
        use_all_modes : bool, optional
            Whether to use the sum of all probe modes' intensities for the illumination map.
        """
        if self.preconditioner is None:
            self.initialize_preconditioner(probe, probe_positions, patterns)
        illum_map = self.calculate_illumination_map(probe, probe_positions, use_all_modes=use_all_modes)
        self.preconditioner = (self.preconditioner + illum_map) / 2
    
    def initialize_preconditioner(
        self, 
        probe: "ds.probe.Probe", 
        probe_positions: "ds.probe_positions.ProbePositions",
        patterns: Tensor,
    ) -> None:
        """Initialize the preconditioner. This function reproduces the behavior
        of PtychoShelves in `init_solver` and `load_from_p`: the probe is first
        renormalized before being used to calculate the illumination map.
        Diffraction patterns are needed to calculate the renormalization factor.
        
        Parameters
        ----------
        probe : ds.probe.Probe
            The probe to use for the illumination map.
        probe_positions : ds.probe_positions.ProbePositions
            The positions of the probe.
        patterns : ds.patterns.Patterns
            A (n_scan_points, h, w) tensor giving the diffraction patterns.
        """
        probe_data = probe.data
        probe_renormalization_factor = get_probe_renormalization_factor(patterns)
        probe_data = probe_data / (math.sqrt(probe.shape[-1] * probe.shape[-2]) * 2 * probe_renormalization_factor)
        probe_temp = ds.probe.Probe(data=probe_data, options=copy.deepcopy(probe.options))
        self.preconditioner = self.calculate_illumination_map(probe_temp, probe_positions, use_all_modes=False)
