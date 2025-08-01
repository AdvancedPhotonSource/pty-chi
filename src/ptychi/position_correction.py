# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import torch
import ptychi.image_proc as ip
import ptychi.api as api
from ptychi.timing.timer_utils import timer
import ptychi.global_settings as glb
import ptychi.maths as pmath


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
        dldy, dldx = self._calculate_normalized_position_gradient(
            dodx.real, dodx.imag, dody.real, dody.imag, chi_m0.real, chi_m0.imag, probe.real, probe.imag
        )
        delta_pos = torch.stack([dldy, dldx], dim=1)
        return delta_pos
    
    @torch.compile(disable=not glb.get_use_torch_compile())
    def _calculate_normalized_position_gradient(
        self,
        dodx_real,
        dodx_imag,
        dody_real,
        dody_imag,
        chi_real,
        chi_imag,
        probe_real,
        probe_imag,
        eps=1e-6,
    ):
        pdodx_r, pdodx_i = pmath.complex_mul_ra(dodx_real, dodx_imag, probe_real, probe_imag)
        numer_x, _ = pmath.complex_mul_conj_ra(chi_real, chi_imag, pdodx_r, pdodx_i)
        denom_x = pdodx_r ** 2 + pdodx_i ** 2
        numer_x = torch.sum(numer_x, dim=(-2, -1))
        denom_x = torch.sum(denom_x, dim=(-2, -1))
        dldx = numer_x / (denom_x + max(denom_x.max(), eps))
        
        pdody_r, pdody_i = pmath.complex_mul_ra(dody_real, dody_imag, probe_real, probe_imag)
        numer_y, _ = pmath.complex_mul_conj_ra(chi_real, chi_imag, pdody_r, pdody_i)
        denom_y = pdody_r ** 2 + pdody_i ** 2
        numer_y = torch.sum(numer_y, dim=(-2, -1))
        denom_y = torch.sum(denom_y, dim=(-2, -1))
        dldy = numer_y / (denom_y + max(denom_y.max(), eps))
        return dldy, dldx
