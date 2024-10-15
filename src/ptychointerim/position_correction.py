import torch
from ptychointerim.image_proc import find_cross_corr_peak, gaussian_gradient
from ptychointerim.ptychotorch.data_structures import Probe
import ptychointerim.api as api


class PositionCorrection:
    def __init__(
        self,
        probe: Probe = None,
        object_step_size: float = None,
        correction_options: api.options.base.ProbePositionOptions.CorrectionOptions = None,
    ):
        self.probe = probe
        self.object_step_size = object_step_size
        self.correction_type = correction_options.correction_type
        self.scale = correction_options.cross_correlation_options.scale
        self.real_space_width = correction_options.cross_correlation_options.real_space_width
        self.probe_threshold = correction_options.cross_correlation_options.probe_threshold

    def get_update(
        self,
        chi: torch.Tensor,
        obj_patches: torch.Tensor,
        delta_o_patches: torch.Tensor,
    ):
        if self.correction_type is api.PositionCorrectionTypes.GRADIENT:
            return self.get_gradient_update(chi, obj_patches)
        elif self.correction_type is api.PositionCorrectionTypes.CROSS_CORRELATION:
            return self.get_cross_correlation_update(obj_patches, delta_o_patches)

    def get_cross_correlation_update(
        self, obj_patches: torch.Tensor, delta_o_patches: torch.Tensor
    ):
        """
        Use cross-correlation position correction to compute an update to the probe positions.

        Based on the paper:
        - Translation position determination in ptychographic coherent diffraction imaging (2013) - Fucai Zhang

        :param obj_patches: A (batch_size, h, w) tensor of patches of the object.
        :param delta_o_patches: A (batch_size, h, w) tensor of patches of the update to be applied to the object.
        """

        updated_obj_patches = obj_patches + delta_o_patches * self.object_step_size

        probe_m0 = self.probe.get_mode_and_opr_mode(0, 0)

        N_positions = len(obj_patches)
        delta_pos = torch.zeros((N_positions, 2))

        probe_thresh = probe_m0.abs().max() * self.probe_threshold
        probe_mask = probe_m0.abs() > probe_thresh

        for i in range(N_positions):
            delta_pos[i] = -find_cross_corr_peak(
                updated_obj_patches[i] * probe_mask,
                obj_patches[i] * probe_mask,
                scale=self.scale,
                real_space_width=self.real_space_width,
            )

        return delta_pos

    def get_gradient_update(self, chi: torch.Tensor, obj_patches: torch.Tensor, eps=1e-6):
        """
        Calculate the update direction for probe positions. This routine calculates the gradient with regards
        to probe positions themselves, in contrast to the delta of probe caused by a 1-pixel shift as in
        Odstrcil (2018). However, this is the method implemented in both PtychoShelves and Tike.

        Denote probe positions as s. Given dL/dP = -chi * O.conj() (Eq. 24a), dL/ds = dL/dO * dO/ds =
        real(-chi * P.conj() * grad_O.conj()), where grad_O is the spatial gradient of the probe in x or y.

        :param chi: A (batch_size, h, w) tensor of the exit wave update.
        :param obj_patches: A (batch_size, h, w) tensor of patches of the object.
        """

        probe_m0 = self.probe.get_mode_and_opr_mode(0, 0)

        chi_m0 = chi[:, 0, :, :]
        dody, dodx = gaussian_gradient(obj_patches, sigma=0.33)

        pdodx = dodx * probe_m0
        dldx = (torch.real(pdodx.conj() * chi_m0)).sum(-1).sum(-1)
        denom_x = (pdodx.abs() ** 2).sum(-1).sum(-1)
        dldx = dldx / (denom_x + max(denom_x.max(), eps))

        pdody = dody * probe_m0
        dldy = (torch.real(pdody.conj() * chi_m0)).sum(-1).sum(-1)
        denom_y = (pdody.abs() ** 2).sum(-1).sum(-1)
        dldy = dldy / (denom_y + max(denom_y.max(), eps))

        delta_pos = torch.stack([dldy, dldx], dim=1)

        return delta_pos
