from typing import Optional

import torch
from torch import Tensor


class MSELossOfSqrt(torch.nn.MSELoss):
    
    def __init__(self, eps=1e-7, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps
    
    def forward(self, y_pred: Tensor, y_true: Tensor, weight_map: Optional[Tensor] = None) -> Tensor:
        if weight_map is None:
            l = super().forward(torch.sqrt(y_pred + self.eps), torch.sqrt(y_true + self.eps))
        else:
            d = ((torch.sqrt(y_pred + self.eps) - torch.sqrt(y_true + self.eps)) ** 2) * weight_map
            if self.reduction == 'mean':
                l = torch.mean(d)
            elif self.reduction == 'sum':
                l = torch.sum(d)
        return l
