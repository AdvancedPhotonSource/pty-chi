from typing import Tuple, TYPE_CHECKING
import logging

import torch
from torch import Tensor

import ptychi.image_proc as ip
import ptychi.data_structures.base as ds
from ptychi.utils import get_default_complex_dtype, to_tensor, to_numpy
import ptychi.maps as maps

if TYPE_CHECKING:
    import ptychi.api as api
    from ptychi.data_structures.probe_positions import ProbePositions
    
logger = logging.getLogger(__name__)


class Object(ds.ReconstructParameter):
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
        self.l1_norm_constraint_weight = options.l1_norm_constraint.weight
        self.l1_norm_constraint_stride = options.l1_norm_constraint.stride
        self.smoothness_constraint_alpha = options.smoothness_constraint.alpha
        self.smoothness_constraint_stride = options.smoothness_constraint.stride
        self.total_variation_weight = options.total_variation.weight
        self.total_variation_stride = options.total_variation.stride
        center_pixel = torch.tensor(self.shape, device=torch.get_default_device()) / 2.0
        self.roi_bbox: ds.BoundingBox = None

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
        logger.debug("L1 norm constraint applied to object.")

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

    def constrain_smoothness(self):
        raise NotImplementedError

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

    def remove_grid_artifacts_enabled(self, current_epoch: int):
        if (
            self.options.remove_grid_artifacts.enabled
            and self.optimization_enabled(current_epoch)
            and (current_epoch - self.optimization_plan.start)
            % self.options.remove_grid_artifacts.stride
            == 0
        ):
            return True
        else:
            return False

    def remove_grid_artifacts(self, *args, **kwargs):
        raise NotImplementedError
    
    def build_roi_bounding_box(self, positions: "ProbePositions"):
        pos = positions.data
        self.roi_bbox = ds.BoundingBox(
            sy=pos[:, 0].min(),
            ey=pos[:, 0].max(),
            sx=pos[:, 1].min(),
            ex=pos[:, 1].max(),
            origin=tuple(to_numpy(self.center_pixel)),
        )
    
    def get_object_in_roi(self):
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

    def extract_patches(self, positions: Tensor, patch_shape: Tuple[int, int]):
        """
        Extract (n_patches, n_slices, h', w') patches from the object.

        Parameters
        ----------
        positions : Tensor
            Tensor of shape (N, 2) giving the center positions of the patches in pixels.
            The origin of the given positions are assumed to be `self.center_pixel`.
        patch_shape : tuple
            Tuple giving the lateral patch shape in pixels.

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
            patches = self.extract_patches_function(
                self.get_slice(i_slice), positions, patch_shape
            )
            patches_all_slices.append(patches)
        patches_all_slices = torch.stack(patches_all_slices, dim=1)
        return patches_all_slices

    def place_patches(self, positions: Tensor, patches: Tensor, *args, **kwargs):
        """
        Place patches into the object.

        Parameters
        ----------
        positions : Tensor
            Tensor of shape (N, 2) giving the center positions of the patches in pixels.
            The origin of the given positions are assumed to be `self.center_pixel`.
        patches : Tensor
            Tensor of shape (n_patches, n_slices, H, W) of image patches.
        """
        positions = positions + self.center_pixel
        updated_slices = []
        for i_slice in range(self.n_slices):
            image = self.place_patches_function(
                self.get_slice(i_slice), positions, patches[:, i_slice, ...]
            )
            updated_slices.append(image)
        updated_slices = torch.stack(updated_slices, dim=0)
        self.tensor.set_data(updated_slices)

    def place_patches_on_empty_buffer(self, positions: Tensor, patches: Tensor, *args, **kwargs):
        """
        Place patches into a zero array with the *lateral* shape of the object.

        Parameters
        ----------
        positions : Tensor
            Tensor of shape (N, 2) giving the center positions of the patches in pixels.
            The origin of the given positions are assumed to be `self.center_pixel`.
        patches : Tensor
            Tensor of shape (N, H, W) of image patches.

        Returns
        -------
        image : Tensor
            A tensor with the lateral shape of the object with patches added onto it.
        """
        positions = positions + self.center_pixel
        image = torch.zeros(
            self.lateral_shape, dtype=get_default_complex_dtype(), device=self.tensor.data.device
        )
        image = self.place_patches_function(image, positions, patches, op="add")
        return image
    
    def get_object_in_roi(self):
        bbox = self.roi_bbox.get_bbox_with_top_left_origin()
        return self.data[:, int(bbox.sy):int(bbox.ey), int(bbox.sx):int(bbox.ex)]

    def constrain_smoothness(self) -> None:
        """
        Smooth the magnitude of the object.
        """
        if self.smoothness_constraint_alpha > 1.0 / 8:
            logger.warning(
                f"Alpha = {self.smoothness_constraint_alpha} in smoothness constraint should be less than 1/8."
            )
        psf = torch.ones(3, 3, device=self.device) * self.smoothness_constraint_alpha
        psf[2, 2] = 1 - 8 * self.smoothness_constraint_alpha

        data = self.data
        for i_slice in range(self.n_slices):
            mag = data[i_slice].abs()
            mag = ip.convolve2d(mag, psf, "same")
            data[i_slice] = data[i_slice] / data[i_slice].abs() * mag
        self.set_data(data)

    def constrain_total_variation(self) -> None:
        data = self.data
        for i_slice in range(self.n_slices):
            data[i_slice] = ip.total_variation_2d_chambolle(
                data[i_slice], lmbda=self.total_variation_weight, niter=2
            )
        self.set_data(data)

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

    def multislice_regularization_enabled(self, current_epoch: int):
        if (
            self.options.multislice_regularization.weight > 0
            and self.optimization_enabled(current_epoch)
            and (current_epoch - self.optimization_plan.start)
            % self.options.multislice_regularization.stride
            == 0
        ):
            return True
        else:
            return False

    def regularize_multislice(self):
        """
        Regularize multislice by applying a low-pass transfer function to the
        3D Fourier space of the magnitude and phase (unwrapped) of all slices.

        Adapted from fold_slice (regulation_multilayers.m).
        """
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
        # The weight map is clipped at 0.1. Values that are too close to 0 would result in
        # small but non-zero pixels in the image after it is multiplied with the weight map.
        # During phase unwrapping, if the phase gradient is calculated using finite difference
        # with Fourier shift, these values can dangle around 0, causing the phase of the
        # complex gradient to flip between pi and -pi.
        w_phase = torch.clip(10 * (self.preconditioner / self.preconditioner.max()), max=1)
        w_phase_clipped = torch.clip(w_phase, min=0.1)

        if self.options.multislice_regularization.unwrap_phase:
            pobj = [
                ip.unwrap_phase_2d(
                    obj[i_slice],
                    weight_map=w_phase_clipped,
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
