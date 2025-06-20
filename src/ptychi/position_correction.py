# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import torch
import ptychi.image_proc as ip
import ptychi.api as api
from ptychi.timing.timer_utils import timer


class PositionCorrection:
    """
    Class containing the various position correction functions used to
    calculate updates to the probe positions.
    """

    def __init__(
        self,
        options: "api.options.base.PositionCorrectionOptions" = None,
    ):
        self.options = options

    @timer()
    def get_update(
        self,
        chi: torch.Tensor,
        obj_patches: torch.Tensor,
        delta_o_patches: torch.Tensor,
        unique_probes: torch.Tensor,
        object_step_size: float,
    ):
        """
        Calculate the position update step direction using the selected position correction function.

        Parameters
        ----------
        chi : torch.Tensor
            A (batch_size, n_modes, h, w) tensor of the exit wave update.
        obj_patches : torch.Tensor
            A (batch_size, n_slices, h, w) tensor of patches of the object. The slice dimension
            is only there to maintain the consistency to the general shape of object patches. 
            Correction algorithms only use the first slice. If position correction should be done
            using other slices, pass the correct slice of the object patches to this function as
            `obj_patches[:, i_slice:i_slice + 1]`.
        delta_o_patches : torch.Tensor
            A (batch_size, n_slices or 1, h, w) tensor of patches of the update to be applied to the object.
        unique_probes : torch.Tensor
            A (batch_size, n_modes, h, w) tensor of unique probes for all positions in the batch.
            The mode dimension is only there to maintain the consistency to the general shape of probe
            patches. Correction algorithms only use the first mode.
        object_step_size : float
            The step size/learning rate of the object optimizer.

        Returns
        -------
        Tensor
            A (n_positions, 2) tensor of updates to the probe positions.
        """
        if self.options.correction_type is api.PositionCorrectionTypes.GRADIENT:
            return self.get_gradient_update(chi, obj_patches, unique_probes)
        elif self.options.correction_type is api.PositionCorrectionTypes.CROSS_CORRELATION:
            return self.get_cross_correlation_update(
                obj_patches, delta_o_patches, unique_probes, object_step_size
            )

    @timer()
    def get_cross_correlation_update(
        self,
        obj_patches: torch.Tensor,
        delta_o_patches: torch.Tensor,
        probe: torch.Tensor,
        object_step_size: float,
    ):
        """
        Use cross-correlation position correction to compute an update to the probe positions.

        Based on the paper:
        - Translation position determination in ptychographic coherent diffraction imaging (2013) - Fucai Zhang
        """
        # Just take the first slice.
        obj_patches = obj_patches[:, 0]
        probe = probe[:, 0]
        delta_o_patches = delta_o_patches[:, 0]
        
        updated_obj_patches = obj_patches + delta_o_patches * object_step_size

        n_positions = len(obj_patches)
        delta_pos = torch.zeros((n_positions, 2))

        probe_thresh = probe.abs().max() * self.options.cross_correlation_probe_threshold
        probe_mask = probe.abs() > probe_thresh

        for i in range(n_positions):
            delta_pos[i] = -ip.find_cross_corr_peak(
                updated_obj_patches[i] * probe_mask[i],
                obj_patches[i] * probe_mask[i],
                scale=self.options.cross_correlation_scale,
                real_space_width=self.options.cross_correlation_real_space_width,
            )

        return delta_pos

    @timer()
    def get_gradient_update(
        self, chi: torch.Tensor, obj_patches: torch.Tensor, probe: torch.Tensor, eps=1e-6
    ):
        """
        Calculate the update direction for probe positions. This routine calculates the gradient with regards
        to probe positions themselves, in contrast to the delta of probe caused by a 1-pixel shift as in
        Odstrcil (2018). However, this is the method implemented in both PtychoShelves and Tike.

        Denote probe positions as s. Given dL/dP = -chi * O.conj() (Eq. 24a), dL/ds = dL/dO * dO/ds =
        real(-chi * P.conj() * grad_O.conj()), where grad_O is the spatial gradient of the probe in x or y.
        """
        # Just take the first slice.
        obj_patches = obj_patches[:, 0]
        
        # Take the first mode of probe and chi.
        probe = probe[:, 0, :, :]
        chi_m0 = chi[:, 0, :, :]
        
        if self.options.differentiation_method == api.ImageGradientMethods.GAUSSIAN:
            dody, dodx = ip.gaussian_gradient(obj_patches, sigma=0.33)
        elif self.options.differentiation_method == api.ImageGradientMethods.FOURIER_DIFFERENTIATION:
            dody, dodx = ip.fourier_gradient(obj_patches)
        elif self.options.differentiation_method == api.ImageGradientMethods.FOURIER_SHIFT:
            dody, dodx = ip.fourier_shift_gradient(obj_patches)
        elif self.options.differentiation_method == api.ImageGradientMethods.NEAREST:
            dody, dodx = ip.nearest_neighbor_gradient(obj_patches, "backward")
        pdodx = dodx * probe
        dldx = (torch.real(pdodx.conj() * chi_m0)).sum(-1).sum(-1)
        denom_x = (pdodx.abs() ** 2).sum(-1).sum(-1)
        zero_mask = denom_x == 0
        if torch.count_nonzero(zero_mask) > 0:
            denom_x[zero_mask] = denom_x[zero_mask] + max(denom_x.max(), eps)
        dldx = dldx / denom_x

        pdody = dody * probe
        dldy = (torch.real(pdody.conj() * chi_m0)).sum(-1).sum(-1)
        denom_y = (pdody.abs() ** 2).sum(-1).sum(-1)
        zero_mask = denom_y == 0
        if torch.count_nonzero(zero_mask) > 0:
            denom_y[zero_mask] = denom_y[zero_mask] + max(denom_y.max(), eps)
        dldy = dldy / denom_y

        delta_pos = torch.stack([dldy, dldx], dim=1)

        return delta_pos
