from typing import Union, Optional, Tuple, Literal

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
) -> Tensor:
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
    x, dim, group_dim = _prepare_data_for_orthogonalization(x, dim, group_dim, move_group_dim_to=0)
    u = x.clone()
    for i in range(1, len(x)):
        u[i:] -= project(x[i:], u[i - 1:i], dim=dim)
    return torch.moveaxis(u, 0, group_dim)


def orthogonalize_svd(
    x: Tensor,
    dim: Union[int, Tuple[int, ...]] = -1,
    group_dim: Union[int, None] = None,
    preserve_norm: bool = False
) -> Tensor:
    """
    SVD orthogonalization for complex arrays. Adapted from PtychoShelves (probe_modes_ortho.m).

    :param x: data to be orthogonalized.
    :param dim: The axis/axes to be orthogonalized. By default only the last axis is
        orthogonalized. If axis is a tuple, then the number of orthogonal
        vectors is the length of the last dimension not included in axis. The
        other dimensions are broadcast.
    :param group_dim: The axis along which to orthogonalize. Other dimensions are broadcast.
    """
    orig_shape = x.shape
    
    x, dim, group_dim = _prepare_data_for_orthogonalization(x, dim, group_dim, move_group_dim_to=None)
    if preserve_norm:
        orig_norm = norm(x, dim=list(dim) + [group_dim], keepdims=True)
    
    # Move group_dim to the end, and move dim right before group_dim.
    x = torch.moveaxis(x, list(sorted(dim)) + [group_dim], list(range(-len(dim) - 1, 0)))
    
    # Now the axes of x is rearranged to (bcast_dims, *dim, group_dim).
    # Straighten dimensions given by `dim`.`
    x = x.reshape(list(orig_shape[:-(len(dim) + 1)]) + [-1] + [orig_shape[group_dim]])
    
    # Shape of x:        (shape of bcast_dims, prod(shape of dim), shape of group_dim)
    u, _, _ = torch.linalg.svd(x, full_matrices=False)
    k = orig_shape[group_dim]
    u = u[..., :k]
        
    # Restore u to the original shape.
    # Unflatten `dim`.
    x = u.reshape(list(orig_shape[:-(len(dim) + 1)]) + [orig_shape[i] for i in dim] + [orig_shape[group_dim]])
    # Then, Move `group_dim` and `dim` back. 
    x = torch.moveaxis(x, list(range(-len(dim) - 1, 0)), list(sorted(dim)) + [group_dim])
    
    if preserve_norm:
        new_norm = norm(x, dim=list(dim) + [group_dim], keepdims=True)
        x = x * (orig_norm / new_norm)
    
    return x

def project(a, b, dim=None, eps=1e-5):
    """Return complex vector projection of a onto b for along given axis."""
    bh = b / (inner(b, b, dim=dim, keepdims=True) + eps)
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


def _prepare_data_for_orthogonalization(
    x: Tensor,
    dim: Union[int, Tuple[int, ...]] = -1,
    group_dim: Union[int, None] = None,
    move_group_dim_to: Optional[Union[int, Literal['before_dim']]] = None,
) -> Tuple[Tensor, Tuple[int, ...], int]:
    """
    Find `dim` and `group_dim`. 
    Permute the axes of x, such that group_dim is at the front.
    """
    # If dim contains negative indices, convert them to positive indices.
    try:
        dim = tuple(a % x.ndim for a in dim)
    except TypeError:
        dim = (dim % x.ndim,)
    # Find group_dim, the last dimension not included in axis; we iterate over N
    # vectors in the Gram-schmidt algorithm. Dimensions that are not N or
    # included in axis are leading dimensions for broadcasting.
    if group_dim is None:
        group_dim = x.ndim - 1
        while group_dim in dim:
            group_dim -= 1
    group_dim = group_dim % x.ndim
    if group_dim in dim:
        raise ValueError("Cannot orthogonalize a single vector.")
    if move_group_dim_to is not None:
        if move_group_dim_to == 'before_dim':
            x = torch.moveaxis(x, group_dim, min(dim) - 1)
        else:
            x = torch.moveaxis(x, group_dim, move_group_dim_to)
    return x, dim, group_dim
