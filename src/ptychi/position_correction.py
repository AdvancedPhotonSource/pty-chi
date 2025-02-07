import torch
import ptychi.data_structures.probe
from ptychi.image_proc import find_cross_corr_peak, gaussian_gradient
import ptychi.data_structures.opr_mode_weights as base
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
        self.correction_type = options.correction_type
        self.cross_correlation_scale = options.cross_correlation_scale
        self.cross_correlation_real_space_width = options.cross_correlation_real_space_width
        self.cross_correlation_probe_threshold = options.cross_correlation_probe_threshold

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
            A (batch_size, n_slices, h, w) tensor of patches of the object.
        delta_o_patches : torch.Tensor
            A (batch_size, n_slices or 1, h, w) tensor of patches of the update to be applied to the object.
        unique_probes : torch.Tensor
            A (batch_size, n_modes, h, w) tensor of unique probes for all positions in the batch.
        object_step_size : float
            The step size/learning rate of the object optimizer.

        Returns
        -------
        Tensor
            A (n_positions, 2) tensor of updates to the probe positions.
        """
        probe_m0 = unique_probes[:, 0]

        if self.correction_type is api.PositionCorrectionTypes.GRADIENT:
            return self.get_gradient_update(chi, obj_patches, probe_m0)
        elif self.correction_type is api.PositionCorrectionTypes.CROSS_CORRELATION:
            return self.get_cross_correlation_update(
                obj_patches, delta_o_patches, probe_m0, object_step_size
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
        delta_o_patches = delta_o_patches[:, 0]
        
        updated_obj_patches = obj_patches + delta_o_patches * object_step_size

        n_positions = len(obj_patches)
        delta_pos = torch.zeros((n_positions, 2))

        probe_thresh = probe.abs().max() * self.cross_correlation_probe_threshold
        probe_mask = probe.abs() > probe_thresh

        for i in range(n_positions):
            delta_pos[i] = -find_cross_corr_peak(
                updated_obj_patches[i] * probe_mask[i],
                obj_patches[i] * probe_mask[i],
                scale=self.cross_correlation_scale,
                real_space_width=self.cross_correlation_real_space_width,
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
        
        chi_m0 = chi[:, 0, :, :]
        dody, dodx = gaussian_gradient(obj_patches, sigma=0.33)

        pdodx = dodx * probe
        dldx = (torch.real(pdodx.conj() * chi_m0)).sum(-1).sum(-1)
        denom_x = (pdodx.abs() ** 2).sum(-1).sum(-1)
        dldx = dldx / (denom_x + max(denom_x.max(), eps))

        pdody = dody * probe
        dldy = (torch.real(pdody.conj() * chi_m0)).sum(-1).sum(-1)
        denom_y = (pdody.abs() ** 2).sum(-1).sum(-1)
        dldy = dldy / (denom_y + max(denom_y.max(), eps))

        delta_pos = torch.stack([dldy, dldx], dim=1)

        return delta_pos
