# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import TYPE_CHECKING

import torch
from torch import Tensor

import ptychi.api.enums as enums
import ptychi.data_structures.base as dsbase
import ptychi.image_proc as ip

if TYPE_CHECKING:
    import ptychi.api.options.base as base_options


class RealSpaceScaling(dsbase.ReconstructParameter):
    options: "base_options.RealSpaceScalingOptions"

    def __init__(
        self,
        *args,
        name: str = "real_space_scaling",
        options: "base_options.RealSpaceScalingOptions" = None,
        **kwargs,
    ):
        """Global real-space scaling factor applied to the exit wavefield.

        Parameters
        ----------
        data : Tensor, optional
            A real tensor of shape ``(1,)`` containing the scaling factor.
        """
        super().__init__(*args, name=name, options=options, is_complex=False, **kwargs)
        if self.shape != (1,):
            raise ValueError("RealSpaceScaling must contain exactly one scalar element.")

    def post_update_hook(self, *args, **kwargs):
        """Clamp the parameter tensor in place.

        The updated tensor has shape ``(1,)``.
        """
        with torch.no_grad():
            self.tensor.clamp_(min=1e-6)

    def get_update(
        self,
        chi: Tensor,
        obj_patches: Tensor,
        probe: Tensor,
        eps: float = 1e-6,
    ) -> Tensor:
        """Estimate the update direction of the real-space scaling factor.

        This follows the same first-order approximation used by PtychoShelves
        for detector-scale refinement: object gradients are combined with a
        radial weighting and projected onto the exit-wave update.

        Parameters
        ----------
        chi : Tensor
            A complex tensor of shape ``(batch_size, n_probe_modes, h, w)``
            giving the exit-wave update at the current slice.
        obj_patches : Tensor
            A complex tensor of shape ``(batch_size, n_slices, h, w)``
            containing object patches for the current batch.
        probe : Tensor
            A complex tensor of shape ``(batch_size, n_probe_modes, h, w)``
            containing the incident wavefields at the current slice.
        eps : float
            Small stabilizer added to the denominator.

        Returns
        -------
        Tensor
            A real tensor of shape ``(1,)`` containing the additive update
            direction for the global scaling factor.
        """
        obj_patches = obj_patches[:, 0]
        probe = probe[:, 0]
        chi_m0 = chi[:, 0]

        if self.options.differentiation_method == enums.ImageGradientMethods.GAUSSIAN:
            dody, dodx = ip.gaussian_gradient(obj_patches, sigma=0.33)
        elif self.options.differentiation_method == enums.ImageGradientMethods.FOURIER_DIFFERENTIATION:
            dody, dodx = ip.fourier_gradient(obj_patches)
        elif self.options.differentiation_method == enums.ImageGradientMethods.FOURIER_SHIFT:
            dody, dodx = ip.fourier_shift_gradient(obj_patches)
        elif self.options.differentiation_method == enums.ImageGradientMethods.NEAREST:
            dody, dodx = ip.nearest_neighbor_gradient(obj_patches, "backward")
        else:
            raise ValueError(
                f"Unsupported differentiation method: {self.options.differentiation_method}"
            )

        h, w = obj_patches.shape[-2:]
        xgrid = -torch.linspace(-1, 1, w, device=obj_patches.device, dtype=obj_patches.real.dtype)
        ygrid = -torch.linspace(-1, 1, h, device=obj_patches.device, dtype=obj_patches.real.dtype)
        xgrid = xgrid * ip.tukey_window(w, 0.1, device=xgrid.device, dtype=xgrid.dtype)
        ygrid = ygrid * ip.tukey_window(h, 0.1, device=ygrid.device, dtype=ygrid.dtype)
        xgrid = xgrid.view(1, 1, w)
        ygrid = ygrid.view(1, h, 1)

        dm_o = dodx * xgrid + dody * ygrid
        dm_op = dm_o * probe
        nom = torch.real(dm_op.conj() * chi_m0).sum(dim=(-1, -2))
        denom = (dm_op.abs() ** 2).sum(dim=(-1, -2))
        denom_bias = torch.maximum(
            denom.max(),
            torch.tensor(eps, device=denom.device, dtype=denom.dtype),
        )
        delta_scale = nom / (denom + denom_bias)
        delta_scale = 0.5 * delta_scale.mean() / ((h + w) * 0.5)
        return delta_scale.reshape(1)
