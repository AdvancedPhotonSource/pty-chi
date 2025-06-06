# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import TYPE_CHECKING

import torch
import numpy as np

import ptychi.position_correction as position_correction
import ptychi.data_structures.base as dsbase
import ptychi.image_proc as ip
import ptychi.maths as pmath
import ptychi.api.enums as enums
from ptychi.utils import to_numpy

if TYPE_CHECKING:
    import ptychi.api as api
    from ptychi.data_structures.probe import Probe
    from ptychi.data_structures.object import PlanarObject


class ProbePositions(dsbase.ReconstructParameter):
    options: "api.options.base.ProbePositionOptions"

    def __init__(
        self,
        *args,
        name: str = "probe_positions",
        options: "api.options.base.ProbePositionOptions" = None,
        **kwargs,
    ):
        """
        Probe positions.

        Parameters
        ----------
        data: a tensor of shape (N, 2) giving the probe positions in pixels.
            Input positions should be in row-major order, i.e., y-positions come first.
        """
        super().__init__(*args, name=name, options=options, is_complex=False, **kwargs)
        self.position_correction = position_correction.PositionCorrection(
            options=options.correction_options
        )
    
        self.register_buffer("initial_positions", self.data.detach())
        self.register_buffer(
            "affine_transform_matrix", 
            self.get_identity_affine_transform_matrix()
        )
        self.register_buffer("position_weights", torch.ones_like(self.data))
        self.affine_transform_components = {
            "scale": 1.0,
            "asymmetry": 0.0,
            "rotation": 0.0,
            "shear": 0.0,
        }

    @property
    def n_scan_points(self):
        return len(self.data)
    
    def get_identity_affine_transform_matrix(self, translation: bool = True):
        m = torch.eye(2, device=self.data.device)
        if translation:
            m = torch.cat([m, torch.zeros(2, 1, device=self.data.device)], dim=1)
        return m
    
    def get_slice_for_correction(self, n_slices: int = None):
        i_slice = self.options.correction_options.slice_for_correction
        if i_slice is None:
            if n_slices is None:
                raise ValueError(
                    "When `slice_for_correction` is not set, `n_slices` must "
                    "be provided to determine the middle slice."
                )
            i_slice = n_slices // 2
        return i_slice

    def get_positions_in_pixel(self):
        return self.data
    
    def position_mean_constraint_enabled(self, current_epoch: int):
        return self.options.constrain_position_mean and self.optimization_enabled(current_epoch)
    
    def constrain_position_mean(self):
        data = self.data
        data = data - data.mean(0)
        self.set_data(data)

    def update_position_weights(
        self, 
        probe: "Probe", 
        object_: "PlanarObject", 
        batch_size: int = 100,
    ):
        """Update the position weights based on patch-wise total variation.
        
        Parameters
        ----------
        probe: Probe
            The probe to use for the update.
        object_: PlanarObject
            The object to use for the update.
        batch_size: int
            The batch size for calculating patch-wise total variation. The computation
            is done in batches to avoid memory issues. This can, but does not have to
            be the same as the batch size used for reconstruction.
        """
        half_probe_shape = [x // 2 for x in probe.shape[-2:]]
        probe_intensity = ip.central_crop_or_pad(
            probe.data[0, 0].abs() ** 2, half_probe_shape
        )
        total_variations = torch.zeros_like(self.data)
        
        obj_slice = object_.get_slice(self.get_slice_for_correction(object_.n_slices))
        roi_bbox = object_.roi_bbox.get_bbox_with_top_left_origin()
        obj_slice = obj_slice / torch.max(obj_slice[roi_bbox.get_slicer()].abs())
        
        for ind_st in range(0, self.n_scan_points, batch_size):
            ind_end = min(ind_st + batch_size, self.n_scan_points)
            obj_patches = object_.extract_patches(
                self.data[ind_st:ind_end],
                patch_shape=half_probe_shape,
                integer_mode=True
            )
            obj_patches = obj_patches[:, self.get_slice_for_correction(object_.n_slices)]
            obj_patches = ip.vignette(
                obj_patches, margin=int(np.mean(half_probe_shape) * 0.15), sigma=np.mean(half_probe_shape) * 0.05, method="gaussian"
            )
            grads_y, grads_x = ip.fourier_gradient(obj_patches)
            
            illumination_map = object_.preconditioner
            illumination_map = ip.gaussian_filter(illumination_map, sigma=np.mean(probe.shape[-2:]) * 0.1)
            illum_patches = ip.extract_patches_integer(
                object_.preconditioner, 
                self.data[ind_st:ind_end] + object_.pos_origin_coords,
                half_probe_shape
            )
            
            grads_y = grads_y.abs() * illum_patches * probe_intensity
            grads_x = grads_x.abs() * illum_patches * probe_intensity
            total_variations[ind_st:ind_end] = torch.sqrt(torch.stack([
                    torch.mean(grads_y, dim=(-2, -1)),
                    torch.mean(grads_x, dim=(-2, -1))
                ], dim=-1)
            )
        
        self.position_weights = total_variations ** 4
        self.position_weights = self.position_weights / self.position_weights.mean()
        
    def update_affine_transform_matrix(self):
        """Fit the affine transformation matrix that relates the initial positions to 
        the current positions.
        """
        dofs = self.options.affine_transform_constraint.degrees_of_freedom
        x = self.position_weights * self.data
        x0 = self.position_weights * self.initial_positions
        
        if enums.AffineDegreesOfFreedom.TRANSLATION in dofs:
            x = torch.cat([x, torch.ones_like(x[..., 0:1])], dim=-1)
            x0 = torch.cat([x0, torch.ones_like(x0[..., 0:1])], dim=-1)
        
        # `fit_linear_transform_matrix(x0, x)` finds A in x0 A = x. When we apply affine transform,
        # we do x = (A x0^T)^T = x0 A^T. Therefore, we need to take the transpose of the result.
        a_mat = pmath.fit_linear_transform_matrix(x0, x).T
        
        if enums.AffineDegreesOfFreedom.TRANSLATION in dofs:
            a_mat = a_mat[:, :-1]
        
        scale, asymmetry, rotation, shear = pmath.decompose_2x2_affine_transform_matrix(a_mat)
        if enums.AffineDegreesOfFreedom.SCALE not in dofs:
            scale = 1.0
        if enums.AffineDegreesOfFreedom.ASYMMETRY not in dofs:
            asymmetry = 0.0
        if enums.AffineDegreesOfFreedom.ROTATION not in dofs:
            rotation = 0.0
        if enums.AffineDegreesOfFreedom.SHEAR not in dofs:
            shear = 0.0
        
        a_mat = pmath.compose_2x2_affine_transform_matrix(
            scale, asymmetry, rotation, shear
        )
        self.affine_transform_matrix = torch.cat([a_mat, torch.zeros(2, 1, device=a_mat.device)], dim=1)
        
        # Update saved components.
        self.affine_transform_components["scale"] = to_numpy(scale)
        self.affine_transform_components["asymmetry"] = to_numpy(asymmetry)
        self.affine_transform_components["rotation"] = to_numpy(rotation)
        self.affine_transform_components["shear"] = to_numpy(shear)
        
    def apply_affine_transform_constraint(self):
        estimated_positions = self.affine_transform_matrix @ \
            torch.cat([self.initial_positions, torch.ones_like(self.initial_positions[..., 0:1])], dim=-1).T
        estimated_positions = estimated_positions.T
        residuals = self.data - estimated_positions
        
        residuals = residuals - residuals.mean(0)
        errors = residuals.abs()
        max_error = self.options.affine_transform_constraint.max_expected_error
        
        relax = 0.1
        flexibility = relax / (1 + self.position_weights)
        flexibility = torch.clip(flexibility + (errors - max_error).clip(min=0) ** 2 / max_error ** 2, max=10 * relax)
        
        pos_new = self.data * (1 - flexibility) + flexibility * estimated_positions
        self.set_data(pos_new)

    def step_optimizer(self, *args, **kwargs):
        """Step the optimizer with gradient filled in. This function
        can optionally impose a limit on the magnitude of the update.
        """
        limit_user = self.options.correction_options.update_magnitude_limit
        if limit_user is not None and limit_user <= 0:
            raise ValueError("`update_magnitude_limit` should either be None or a positive number.")
        if limit_user == torch.inf:
            limit_user = None
        data0 = self.data
            
        self.optimizer.step()
        
        if not self.options.correction_options.clip_update_magnitude_by_mad and limit_user is None:
            return
        
        # Get update.
        data = self.data
        dx = data - data0
        update_mag = dx.abs()
        update_signs = dx.sign()
        
        # Truncate update by the limit and reapply it.
        if self.options.correction_options.clip_update_magnitude_by_mad:
            limit_mad = pmath.mad(dx, dim=0) * 10
        else:
            limit_mad = torch.inf
        if limit_user is not None:
            limit = torch.clip(limit_mad, max=limit_user)
        else:
            limit = limit_mad
        dx = update_mag.clip(max=limit) * update_signs
        
        with torch.no_grad():
            self.set_data(data0 + dx)
