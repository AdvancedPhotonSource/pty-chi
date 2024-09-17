from typing import Union, Optional, Tuple

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


def orthogonalize_gs(
    x: Tensor,
    dim: Union[int, Tuple[int, ...]] = -1,
    group_dim: Union[int, None] = None,
):
    """
    Gram-schmidt orthogonalization for complex arrays. Adapted from
    Tike (https://github.com/AdvancedPhotonSource/tike).
    :param x: data to be orthogonalized.
    :param dim: The axis/axes to be orthogonalized. By default only the last axis is
        orthogonalized. If axis is a tuple, then the number of orthogonal
        vectors is the length of the last dimension not included in axis. The
        other dimensions are broadcast.
    :param group_dim: The axis along which to orthogonalize. Other dimensions are broadcast.
    """
    # Find group_dim, the last dimension not included in axis; we iterate over N
    # vectors in the Gram-schmidt algorithm. Dimensions that are not N or
    # included in axis are leading dimensions for broadcasting.
    try:
        dim = tuple(a % x.ndim for a in dim)
    except TypeError:
        dim = (dim % x.ndim,)
    if group_dim is None:
        group_dim = x.ndim - 1
        while group_dim in dim:
            group_dim -= 1
    group_dim = group_dim % x.ndim
    if group_dim in dim:
        raise ValueError("Cannot orthogonalize a single vector.")
    # Move axis bcast_dim to the front for convenience
    x = torch.moveaxis(x, group_dim, 0)
    u = x.clone()
    for i in range(1, len(x)):
        u[i:] -= project(x[i:], u[i - 1:i], dim=dim)
    return torch.moveaxis(u, 0, group_dim)


def project(a, b, dim=None):
    """Return complex vector projection of a onto b for along given axis."""
    bh = b / inner(b, b, dim=dim, keepdims=True)
    return inner(a, b, dim=dim, keepdims=True) * bh


def inner(x, y, dim=None, keepdims=False):
    """Return the complex inner product; the order of the operands matters."""
    return (x * y.conj()).sum(dim, keepdims=keepdims)


def mnorm(x, dim=-1, keepdims=False):
    """Return the vector 2-norm of x but replace sum with mean."""
    return torch.sqrt(torch.mean((x * x.conj()).real, dim=dim, keepdims=keepdims))


def norm(x, dim=-1, keepdims=False):
    """Return the vector 2-norm of x along given axis."""
    return torch.sqrt(torch.sum((x * x.conj()).real, dim=dim, keepdims=keepdims))