import torch
from torch import Tensor


def trim_mean(x: torch.Tensor, fraction: float = 0.1) -> torch.Tensor:
    """
    Calculate the mean of a tensor after removing a certain percentage of the
    lowest and highest values.

    :param x: tensor.
    :param fraction: the fraction of trim, between 0 and 1.
    :return: the trimmed mean.
    """
    lb = torch.quantile(x, fraction)
    ub = torch.quantile(x, 1 - fraction)
    mask = (x >= lb) & (x <= ub)
    if torch.count_nonzero(mask) > 0:
        return torch.mean(x[mask])
    else:
        return torch.mean(x)
    