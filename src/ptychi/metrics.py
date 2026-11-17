# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Optional, Literal

import torch
from torch import Tensor


def validate_predicted_intensity(y_pred: Tensor) -> None:
    """Require finite, nonnegative intensities supplied to an intensity-domain loss.

    Callers select constrained pixels before validation. Zero is permitted for
    the losses' existing denominator/log stabilizers; predictions are unchanged.
    """
    if not torch.all(torch.isfinite(y_pred) & (y_pred >= 0)):
        raise ValueError(
            "Predicted intensities must be finite and nonnegative on constrained pixels. "
            "Check detector response offsets or reduce the reconstruction step size."
        )


class MSELossOfSqrt(torch.nn.MSELoss):
    def __init__(self, eps=1e-7, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps

    def forward(
        self, y_pred: Tensor, y_true: Tensor, weight_map: Optional[Tensor] = None
    ) -> Tensor:
        validate_predicted_intensity(y_pred)
        if weight_map is None:
            loss = super().forward(torch.sqrt(y_pred + self.eps), torch.sqrt(y_true + self.eps))
        else:
            d = ((torch.sqrt(y_pred + self.eps) - torch.sqrt(y_true + self.eps)) ** 2) * weight_map

            if self.reduction == "mean":
                loss = torch.mean(d)
            elif self.reduction == "sum":
                loss = torch.sum(d)

        return loss


class TotalVariationLoss(torch.nn.Module):
    def __init__(self, reduction: Literal["mean", "sum"] = "mean", *args, **kwargs) -> None:
        """Calculates the total variation of one or more 2D
        images along the last two dimensions.
        """
        super().__init__(*args, **kwargs)
        self.reduction = reduction

    def forward(self, img: Tensor) -> Tensor:
        """The forward pass.

        Parameters
        ----------
        img : Tensor
            A (..., H, W) tensor.

        Returns
        -------
        Tensor
            The total variation of the input image(s).
        """
        tv = torch.norm(img[..., :, 1:] - img[..., :, :-1], p=2, dim=(-2, -1)) + torch.norm(
            img[..., 1:, :] - img[..., :-1, :], p=2, dim=(-2, -1)
        )
        if self.reduction == "mean":
            return torch.mean(tv)
        elif self.reduction == "sum":
            return torch.sum(tv)
