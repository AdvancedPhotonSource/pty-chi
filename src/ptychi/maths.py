from typing import Literal, Optional, Tuple, Union

import torch
import numpy as np
import cupy
import cupyx.scipy.fft as cupyfft

from ptychi.timing.timer_utils import timer


@timer()
def trim_mean(
    x: torch.Tensor, 
    fraction: float = 0.1, 
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Calculate the mean of a tensor after removing a certain percentage of the
    lowest and highest values.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    fraction : float, optional
        The fraction of trim, between 0 and 1. Default is 0.1.
    dim : int or tuple of int, optional
        The axis/axes along which to calculate the mean.

    Returns
    -------
    trimmed_mean : tensor
        The trimmed mean.
    """
    lb = torch.quantile(x, fraction, dim=dim, keepdim=True)
    ub = torch.quantile(x, 1 - fraction, dim=dim, keepdim=True)
    mask = (x >= lb) & (x <= ub)

    mask_check = torch.sum(mask, dim=dim)
    mask_check = torch.all(mask_check > 0)
    if mask_check:
        x = x.clone()
        x[~mask] = torch.nan
        return torch.nanmean(x, dim=dim, keepdim=keepdim)
    else:
        return torch.mean(x, dim=dim, keepdim=keepdim)
    

def angle(x: torch.Tensor, eps=1e-5) -> torch.Tensor:
    """
    A stable version of `torch.angle()`, which calculates
    arctan(imag / (real + eps)) to prevent unstable results when
    the real part is small (so that the ratio dangles between -inf
    and inf).

    Parameters
    ----------
    x : torch.Tensor
        The complex tensor as input.
    eps : float
        The stablization constant.

    Returns
    -------
    torch.Tensor
        The phase of the complex tensor.
    """
    return torch.atan2(x.imag, x.real + eps)


@timer()
def orthogonalize_gs(
    x: torch.Tensor,
    dim: Union[int, Tuple[int, ...]] = -1,
    group_dim: Union[int, None] = None,
) -> torch.Tensor:
    """
    Gram-schmidt orthogonalization for complex arrays. Adapted from
    Tike (https://github.com/AdvancedPhotonSource/tike).

    Parameters
    ----------
    x : Tensor
        Data to be orthogonalized.
    dim : int or tuple of int, optional
        The axis/axes to be orthogonalized. By default only the last axis is
        orthogonalized. If axis is a tuple, then the number of orthogonal
        vectors is the length of the last dimension not included in axis. The
        other dimensions are broadcast.
    group_dim : int, optional
        The axis along which to orthogonalize. Other dimensions are broadcast.

    Returns
    -------
    Tensor
        Orthogonalized data.
    """
    x, dim, group_dim = _prepare_data_for_orthogonalization(x, dim, group_dim, move_group_dim_to=0)
    u = x.clone()
    for i in range(1, len(x)):
        u[i:] -= project(x[i:], u[i - 1 : i], dim=dim)
    return torch.moveaxis(u, 0, group_dim)


@timer()
def orthogonalize_svd(
    x: torch.Tensor,
    dim: Union[int, Tuple[int, ...]] = -1,
    group_dim: Union[int, None] = None,
    preserve_norm: bool = False,
) -> torch.Tensor:
    """
    SVD orthogonalization for complex arrays. Adapted from PtychoShelves (probe_modes_ortho.m).

    Parameters
    ----------
    x : Tensor
        Data to be orthogonalized.
    dim : int or tuple of int, optional
        The axis/axes to be orthogonalized. By default only the last axis is
        orthogonalized. If axis is a tuple, then the number of orthogonal
        vectors is the length of the last dimension not included in axis. The
        other dimensions are broadcast.
    group_dim : int, optional
        The axis along which to orthogonalize. Other dimensions are broadcast.

    Returns
    -------
    Tensor
        Orthogonalized data.
    """
    x, dim, group_dim = _prepare_data_for_orthogonalization(
        x, dim, group_dim, move_group_dim_to=None
    )
    if preserve_norm:
        orig_norm = norm(x, dim=list(dim) + [group_dim], keepdims=True)

    # Move group_dim to the end, and move dim right before group_dim.
    x = torch.moveaxis(x, list(sorted(dim)) + [group_dim], list(range(-len(dim) - 1, 0)))
    group_dim_shape = [x.shape[-1]]
    dim_shape = list(x.shape[-(len(dim) + 1) : -1])
    batch_dim_shape = list(x.shape[: -(len(dim) + 1)])

    # Now the axes of x is rearranged to (bcast_dims, *dim, group_dim).
    # Straighten dimensions given by `dim`.`
    # Shape of x:        (bcast_dims_shape, prod(dim_shape), group_dim_shape)
    x = x.reshape(batch_dim_shape + [-1] + group_dim_shape)
    
    # Use higher precision.
    orig_dtype = x.dtype
    if orig_dtype.is_complex:
        x = x.type(torch.complex128)
    else:
        x = x.type(torch.float64)

    # Creat an shape(group_dim) x shape(group_dim) covariance matrix and eigendecompose it.
    if x.ndim == 2:
        covmat = x.transpose(-1, -2) @ x.conj()
    else:
        covmat = torch.bmm(x.transpose(-1, -2), x.conj())

    evals, evecs = torch.linalg.eig(covmat)

    sorted_inds = torch.argsort(evals.abs(), dim=-1, descending=True)
    # Shape of evecs:    (bcast_dims_shape, group_dim_shape, group_dim_shape)
    evecs = torch.gather(
        evecs,
        dim=-1,
        index=sorted_inds[..., None, :].repeat(*([1] * (evecs.ndim - 2)), evecs.shape[-2], 1),
    )

    if x.ndim == 2:
        x = x @ evecs.conj()
    else:
        x = torch.bmm(x, evecs.conj())

    # Restore u to the original shape.
    # Unflatten `dim`.
    x = x.reshape(batch_dim_shape + dim_shape + group_dim_shape)
    # Then, Move `group_dim` and `dim` back.
    x = torch.moveaxis(x, list(range(-len(dim) - 1, 0)), list(sorted(dim)) + [group_dim])

    if preserve_norm:
        new_norm = norm(x, dim=list(dim) + [group_dim], keepdims=True)
        x = x * (orig_norm / new_norm)

    return x.type(orig_dtype)


def project(a, b, dim=None):
    """Return complex vector projection of a onto b for along given axis."""
    projected_length = inner(a, b, dim=dim, keepdims=True) / inner(b, b, dim=dim, keepdims=True)
    return projected_length * b

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
    x: torch.Tensor,
    dim: Union[int, Tuple[int, ...]] = -1,
    group_dim: Union[int, None] = None,
    move_group_dim_to: Optional[Union[int, Literal["before_dim"]]] = None,
) -> Tuple[torch.Tensor, Tuple[int, ...], int]:
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
        if move_group_dim_to == "before_dim":
            x = torch.moveaxis(x, group_dim, min(dim) - 1)
        else:
            x = torch.moveaxis(x, group_dim, move_group_dim_to)
    return x, dim, group_dim


def polyfit(x: torch.Tensor, y: torch.Tensor, deg: int = 1):
    """
    Fit a polynomial to the data and return the coefficients
    from high to low order.
    
    Parameters
    ----------
    x : torch.Tensor
        The independent variable.
    y : torch.Tensor
        The dependent variable.
    deg : int
        The degree of the polynomial to fit.
    
    Returns
    -------
    torch.Tensor
        The coefficients of the polynomial.
    """
    x_powers = x[:, None] ** torch.arange(deg, -1, -1)
    return torch.linalg.lstsq(x_powers, y, rcond=None)[0]


def polyval(x: torch.Tensor, coeffs: torch.Tensor):
    """
    Evaluate a polynomial at the given points.
    
    Parameters
    ----------
    x : torch.Tensor
        The independent variable.
    coeffs : torch.Tensor
        The coefficients of the polynomial from high to low order.
    
    Returns
    -------
    torch.Tensor
        The values of the polynomial at the given points.
    """
    return (coeffs * x.view(-1, 1) ** torch.arange(len(coeffs) - 1, -1, -1)).sum(1)


def fft2_precise(x, norm=None):
    """
    2D FFT with double precision.
    """
    return torch.fft.fft2(x.type(torch.complex128), norm=norm).type(x.dtype)


def ifft2_precise(x, norm=None):
    """
    2D FFT with double precision.
    """
    return torch.fft.ifft2(x.type(torch.complex128), norm=norm).type(x.dtype)


def is_all_integer(x: torch.Tensor) -> bool:
    """Check if all elements in a tensor are integers."""
    return torch.allclose(torch.eq(x, torch.round(x)), atol=1e-7)


def masked_argmax(
    x: torch.Tensor | np.ndarray, 
    mask: torch.Tensor | np.ndarray
) -> torch.Tensor | np.ndarray:
    """
    Find the index of the maximum value in a tensor within a mask.
    
    This function works for both torch and numpy arrays.
    
    Parameters
    ----------
    x : torch.Tensor | np.ndarray
        The input tensor.
    mask : torch.Tensor | np.ndarray
        The mask. Search is only performed within the masked region.
    
    Returns
    -------
    torch.Tensor | np.ndarray
        The index of the maximum value in the tensor within the mask.
    """
    if isinstance(x, torch.Tensor):
        x = torch.where(mask, x, -torch.inf)
        return torch.argmax(x)
    else:
        x = np.where(mask, x, -np.inf)
        return np.argmax(x)


def masked_argmin(
    x: torch.Tensor | np.ndarray, 
    mask: torch.Tensor | np.ndarray
) -> torch.Tensor | np.ndarray:
    """
    Find the index of the minimum value in a tensor within a mask.
    
    This function works for both torch and numpy arrays.
    
    Parameters
    ----------
    x : torch.Tensor | np.ndarray
        The input tensor.
    mask : torch.Tensor | np.ndarray
        The mask. Search is only performed within the masked region.
    
    Returns
    -------
    torch.Tensor | np.ndarray
        The index of the minimum value in the tensor within the mask.
    """
    if isinstance(x, torch.Tensor):
        x = torch.where(mask, x, torch.inf)
        return torch.argmin(x)
    else:
        x = np.where(mask, x, np.inf)
        return np.argmin(x)
