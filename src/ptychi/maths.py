# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Literal, Optional, Tuple, Union, Callable

import torch
import numpy as np

from ptychi.timing.timer_utils import timer
import ptychi.global_settings as glb

_use_double_precision_for_fft = True
_allow_nondeterministic_algorithms = False


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
    
    
def get_use_double_precision_for_fft():
    return _use_double_precision_for_fft


def set_use_double_precision_for_fft(use_double_precision_for_fft: bool):
    global _use_double_precision_for_fft
    _use_double_precision_for_fft = use_double_precision_for_fft
    
    
def get_allow_nondeterministic_algorithms():
    return _allow_nondeterministic_algorithms


def set_allow_nondeterministic_algorithms(allow: bool):
    global _allow_nondeterministic_algorithms
    _allow_nondeterministic_algorithms = allow


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
    if torch.is_complex(x):
        z_r, z_i = complex_mul_conj_ra(x.real, x.imag, y.real, y.imag)
        z = torch.complex(z_r, z_i)
        return torch.sum(z, dim=dim, keepdims=keepdims)
    else:
        return torch.sum(x * y, dim=dim, keepdims=keepdims)


def mnorm(x, dim=-1, keepdims=False):
    """Return the vector 2-norm of x but replace sum with mean."""
    if torch.is_complex(x):
        x2_r, _ = complex_mul_conj_ra(x.real, x.imag, x.real, x.imag)
        return torch.sqrt(torch.mean(x2_r, dim=dim, keepdims=keepdims))
    else:
        return torch.sqrt(torch.mean((x ** 2), dim=dim, keepdims=keepdims))


def norm(x, dim=-1, keepdims=False):
    """Return the vector 2-norm of x along given axis."""
    if torch.is_complex(x):
        x_r, x_i = complex_mul_conj_ra(x.real, x.imag, x.real, x.imag)
        x2 = torch.complex(x_r, x_i)
        return torch.sqrt(torch.sum(x2.real, dim=dim, keepdims=keepdims))
    else:
        return torch.sqrt(torch.sum(x ** 2, dim=dim, keepdims=keepdims))


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


def real_dtype_to_complex(dtype: torch.dtype) -> torch.dtype:
    """Convert a real dtype to a complex dtype with the same precision.
    If a complex dtype is provided, it is returned unchanged.

    Parameters
    ----------
    dtype : torch.dtype
        The real dtype to convert.

    Returns
    -------
    torch.dtype
        The complex dtype with the same precision.
    """
    if dtype.is_complex:
        return dtype
    elif dtype == torch.float64:
        return torch.complex128
    elif dtype == torch.float32:
        return torch.complex64
    elif dtype == torch.float16:
        return torch.complex32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    

def complex_dtype_to_real(dtype: torch.dtype) -> torch.dtype:
    """Convert a complex dtype to a real dtype with the same precision.
    If a real dtype is provided, it is returned unchanged.

    Parameters
    ----------
    dtype : torch.dtype
        The complex dtype to convert.

    Returns
    -------
    torch.dtype
        The real dtype with the same precision.
    """
    if not dtype.is_complex:
        return dtype
    elif dtype == torch.complex128:
        return torch.float64
    elif dtype == torch.complex64:
        return torch.float32
    elif dtype == torch.complex32:
        return torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def fft2_precise(x, norm=None):
    """
    2D FFT with double precision.
    """
    if get_use_double_precision_for_fft():
        final_dtype = real_dtype_to_complex(x.dtype)
        return torch.fft.fft2(x.type(torch.complex128), norm=norm).type(final_dtype)
    else:
        return torch.fft.fft2(x, norm=norm)


def ifft2_precise(x, norm=None):
    """
    2D FFT with double precision.
    """
    if get_use_double_precision_for_fft():
        final_dtype = real_dtype_to_complex(x.dtype)
        return torch.fft.ifft2(x.type(torch.complex128), norm=norm).type(final_dtype)
    else:
        return torch.fft.ifft2(x, norm=norm)


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


def decompose_2x2_affine_transform_matrix(
    m: torch.Tensor
) -> torch.Tensor:
    """Decompose a 2x2 affine transformation matrix (without translation) into 
    scale, asymmetry, rotation, and shear.
    
    Let x1, x2, x3, x4 be the scale, asymmetry, rotation, and shear. The affine
    transformation matrix is decomposed into
    x1 * ((1+x2/2, 0     )  * ((cos x3,  sin x3)  * ((1,      0)
          (0,      1-x2/2))    (-sin x3, cos x3))    (tan x4, 1))
    and the 
    """
    def f(x):
        y = torch.stack([
            (1 + x[1] / 2) * (torch.cos(x[2]) + torch.sin(x[2]) * torch.tan(x[3])),
            (1 + x[1] / 2) * torch.sin(x[2]),
            (1 - x[1] / 2) * (-torch.sin(x[2]) + torch.cos(x[2]) * torch.tan(x[3])),
            (1 - x[1] / 2) * torch.cos(x[2])
        ]) * x[0]
        return y
    x = nonlinear_lsq(torch.tensor([1.0, 0.0, 0.0, 0.0]), m.reshape(-1), f)
    return x


def compose_2x2_affine_transform_matrix(
    scale: torch.Tensor,
    asymmetry: torch.Tensor,
    rotation: torch.Tensor,
    shear: torch.Tensor
) -> torch.Tensor:
    """
    Compose a 2x2 affine transformation matrix from scale, asymmetry, rotation, and shear.
    """
    x = [scale, asymmetry, rotation, shear]
    m = torch.stack([
        (1 + x[1] / 2) * (torch.cos(x[2]) + torch.sin(x[2]) * torch.tan(x[3])),
        (1 + x[1] / 2) * torch.sin(x[2]),
        (1 - x[1] / 2) * (-torch.sin(x[2]) + torch.cos(x[2]) * torch.tan(x[3])),
        (1 - x[1] / 2) * torch.cos(x[2])
    ]) * x[0]
    return m.reshape(2, 2)


def nonlinear_lsq(
    x0: torch.Tensor,
    y: torch.Tensor,
    f: Callable[[torch.Tensor], torch.Tensor],
    n_iter: int = 20,
    **kwargs
) -> torch.Tensor:
    """Solve a nonlinear least squares problem.
    
    Parameters
    ----------
    x0 : torch.Tensor
        The initial guess for the solution.
    y : torch.Tensor
        The data.
    f : Callable[torch.Tensor, torch.Tensor]
        The function to minimize.
    """
    x = x0.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([x], **kwargs)
    loss_history = []
    
    def compute_loss():
        return ((y - f(x)).abs() ** 2).sum()
    
    def closure():
        optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        return loss
    
    for _ in range(n_iter):
        loss_history.append(compute_loss().item())
        optimizer.step(closure)
    
    # Convergence check.
    loss_history = torch.tensor(loss_history)
    window_size = max(len(loss_history) // 10, 1)
    avg_loss_1 = loss_history[-window_size:].mean()
    avg_loss_2 = loss_history[-2 * window_size:-window_size].mean()
    if torch.abs(avg_loss_2) > 0 and avg_loss_1 / avg_loss_2 > 1.1:
        raise ValueError("Non-linear least squares did not converge.")
    return x.detach()


def fit_linear_transform_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fit a linear transformation matrix A, so that y = A @ x.
    
    Parameters
    ----------
    x : torch.Tensor
        A (n_points, n_features_in) tensor of input data.
    y : torch.Tensor
        A (n_points, n_features_out) tensor of output data.
    
    Returns
    -------
    torch.Tensor
        The linear transformation matrix.
    """
    a_mat = torch.linalg.lstsq(x, y, rcond=None)[0]
    return a_mat

  
def reprod(a, b):
    """
    Real part of the product of 2 arrays
    """
    return a.real * b.real + a.imag * b.imag


def redot(a, b, axis=None):
    """
    Real part of the dot product of 2 arrays
    """
    res = torch.sum(reprod(a, b), axis=axis)
    return res


def mad(x: torch.Tensor, dim=None):
    """
    Mean absolute deviation
    """
    return torch.mean(torch.abs(x - torch.mean(x, dim=dim)), dim=dim)


@torch.compile(disable=not glb.get_use_torch_compile())
def complex_mul_ra(
    a_real: torch.Tensor,
    a_imag: torch.Tensor,
    b_real: torch.Tensor,
    b_imag: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Complex multiplication with real-valued arguments and returns.
    """
    return a_real * b_real - a_imag * b_imag, a_real * b_imag + a_imag * b_real


@torch.compile(disable=not glb.get_use_torch_compile())
def complex_mul_conj_ra(
    a_real: torch.Tensor,
    a_imag: torch.Tensor,
    b_real: torch.Tensor,
    b_imag: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate a * b.conj() with real-valued arguments and returns.
    """ 
    return a_real * b_real + a_imag * b_imag, -a_real * b_imag + a_imag * b_real
