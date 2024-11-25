from typing import Union, Literal, Callable, Optional
import math

import torch
from torch import Tensor
from torchvision.transforms import GaussianBlur
import numpy as np
from numpy import ndarray

import ptychi.maths as pmath
from ptychi.propagate import FourierPropagator


default_complex_dtype = torch.complex64


def get_suggested_object_size(positions_px, probe_shape, extra=0):
    h = np.ceil(positions_px[:, 0].max() - positions_px[:, 0].min()) + probe_shape[0] + extra
    w = np.ceil(positions_px[:, 1].max() - positions_px[:, 1].min()) + probe_shape[1] + extra
    return (int(h), int(w))


def rescale_probe(
    probe: Union[ndarray, Tensor],
    patterns: Union[ndarray, Tensor],
    weights: Optional[Union[ndarray, Tensor]] = None,
) -> None:
    """
    Scale probe so that the sum of intensity matches that of the diffraction patterns.

    Parameters
    ----------
    probe : Tensor
        A (n_modes, h, w) or (n_opr_modes, n_modes, h, w) tensor of the probe.
    patterns : Tensor
        A (n, h, w) tensor of diffraction patterns.
    weights : Tensor, optional
        A (n_opr_modes,) tensor of weights for each OPR mode.

    Returns
    -------
    scaled_probe : Tensor
        The scaled probe.
    """
    propagator = FourierPropagator()

    probe_tensor = torch.tensor(probe)

    if probe_tensor.ndim == 3:
        i_probe = (
            (torch.abs(propagator.propagate_forward(probe_tensor)) ** 2)
            .sum()
            .detach()
            .cpu()
            .numpy()
        )
    else:
        weights = torch.tensor(weights)
        probe_corrected = (probe_tensor * weights[:, None, None, None]).sum(0)
        i_probe = (
            (torch.abs(propagator.propagate_forward(probe_corrected)) ** 2)
            .sum()
            .detach()
            .cpu()
            .numpy()
        )

    patterns = to_numpy(patterns)
    i_data = np.sum(np.mean(patterns, axis=0))
    factor = i_data / i_probe
    probe = probe * np.sqrt(factor)
    return probe


def orthogonalize_initial_probe(
    probe: Tensor, 
    secondary_mode_energy: float = 0.02, 
    method: Literal["hermite"] = "hermite"
) -> Tensor:
    """
    Orthogonalize initial probe. 
    
    Parameters
    ----------
    probe : Tensor
        A (n_opr_modes, n_modes, h, w) tensor of the probe.
    secondary_mode_energy : float, optional
        The energy of the secondary mode relative to the principal mode, which is
        always 1.0.
    method: Literal["hermite"], optional
        The method to use for orthogonalization.
    
    Returns
    -------
    Tensor
        The orthogonalized probe.
    """
    n_modes = probe.shape[1]
    mode_energies = torch.zeros(n_modes)
    mode_energies[1:] = secondary_mode_energy
    mode_energies[0] = 1.0 - mode_energies.sum()
    
    e_total = torch.sum(torch.abs(probe[0, 0]) ** 2)
    mode_energies = mode_energies * e_total
    
    if method == "hermite":
        m = math.ceil(math.sqrt(n_modes)) - 1
        n = math.ceil(n_modes / (m + 1)) - 1
        h = generate_secondary_probe_modes_hermite(probe[0, 0], m, n)
        probe[0, 1:, :, :] = h[1:n_modes, :, :]
    else:
        raise ValueError(f"Unknown orthogonalization method: {method}")
    
    # Normalize.
    for i_mode in range(n_modes):
        probe[0, i_mode] = probe[0, i_mode] * torch.sqrt(mode_energies[i_mode] / torch.sum(torch.abs(probe[0, i_mode] ** 2)))
    return probe


def generate_secondary_probe_modes_hermite(probe: Tensor, m: int, n: int) -> Tensor:
    """
    Generate secondary probe modes using Hermite polynomials.
    
    Parameters
    ----------
    probe : Tensor
        A (h, w) tensor of the primary mode of the probe.
    m, n : int
        The orders of the Hermite polynomial.
    
    Returns
    -------
    Tensor
        A ((m + 1) * (n + 1), h, w) tensor of the secondary probe modes.
    """
    x = torch.arange(probe.shape[-1]) - probe.shape[-1] / 2 + 1
    y = torch.arange(probe.shape[-2]) - probe.shape[-2] / 2 + 1
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    
    cenx = torch.sum(xx * torch.abs(probe) ** 2) / torch.sum(torch.abs(probe) ** 2)
    ceny = torch.sum(yy * torch.abs(probe) ** 2) / torch.sum(torch.abs(probe) ** 2)
    varx = torch.sum((xx - cenx) ** 2 * torch.abs(probe) ** 2) / torch.sum(torch.abs(probe) ** 2)
    vary = torch.sum((yy - ceny) ** 2 * torch.abs(probe) ** 2) / torch.sum(torch.abs(probe) ** 2)
    
    counter = 0
    h = torch.empty([(m + 1) * (n + 1), *probe.shape], dtype=probe.dtype)
    for nii in range(n + 1):
        for mii in range(m + 1):
            auxfunc = ((xx - cenx) ** mii) * ((yy - ceny) ** nii) * probe
            if counter > 0:
                auxfunc = auxfunc * torch.exp(-((xx - cenx) ** 2 / (2 * varx)) - ((yy - ceny) ** 2 / (2 * vary)))
            auxfunc = auxfunc / torch.sqrt(torch.sum(torch.abs(auxfunc) ** 2))    

            # Orthogonalize the current mode to the previous ones.
            for ii in range(counter):
                auxfunc = auxfunc - h[ii] * torch.sum(h[ii] * auxfunc.conj(), dim=(-1, -2))
            auxfunc = auxfunc / torch.sqrt(torch.sum(torch.abs(auxfunc) ** 2))
            h[counter] = auxfunc
            counter += 1
    return h


def get_probe_renormalization_factor(patterns: Tensor | ndarray) -> float:
    """
    Calculate the renormalization factor that should be applied to the probe
    to match the maximum power of the diffraction patterns.
    
    Parameters
    ----------
    patterns : Tensor | ndarray
        A (n, h, w) buffer of diffraction patterns.
    
    Returns
    -------
    float
        The renormalization factor.
    """
    if isinstance(patterns, Tensor):
        patterns = patterns.detach().cpu().numpy()
    max_power = np.max(np.sum((patterns), axis=(1, 2))) / (patterns[0].size)
    return np.sqrt(1 / max_power)
            

def generate_initial_object(shape: tuple[int, ...], method: Literal["random"] = "random") -> Tensor:
    if method == "random":
        obj_mag = generate_gaussian_random_image(shape, loc=0.98, sigma=0.02, smoothing=3.0)
        obj_mag = obj_mag.clamp(0.0, 1.0)
        obj_phase = generate_gaussian_random_image(shape, loc=0.0, sigma=0.02, smoothing=3.0)
        obj_phase = obj_phase.clamp(-torch.pi, torch.pi)
        obj = obj_mag * torch.exp(1j * obj_phase)
    else:
        raise ValueError(f"Unknown object initialization method: {method}")
    obj = obj.type(get_default_complex_dtype())
    return obj


def add_additional_opr_probe_modes_to_probe(
    probe: Tensor, n_opr_modes_to_add: int, normalize: bool = True
) -> Tensor:
    if probe.ndim != 4:
        raise ValueError("probe must be a (n_opr_modes, n_modes, h, w) tensor.")
    n_modes = probe.shape[1]
    opr_modes = torch.empty(
        [n_opr_modes_to_add, n_modes, probe.shape[-2], probe.shape[-1]],
        dtype=get_default_complex_dtype(),
    )
    for i in range(n_opr_modes_to_add):
        for j in range(n_modes):
            mag = generate_gaussian_random_image(
                probe.shape[-2:], loc=0.01, sigma=0.01, smoothing=3.0
            ).clamp(0.0, 1.0)
            phase = generate_gaussian_random_image(
                probe.shape[-2:], loc=0.0, sigma=0.05, smoothing=3.0
            ).clamp(-torch.pi, torch.pi)
            opr_mode = mag * torch.exp(1j * phase)
            if normalize:
                opr_mode = opr_mode / pmath.mnorm(opr_mode, dim=(-2, -1))
            opr_modes[i, j, ...] = opr_mode
    probe = torch.cat([probe, opr_modes], dim=0)
    return probe


def generate_initial_opr_mode_weights(
    n_points: int, n_opr_modes: int, eigenmode_weight: float = 0.0
) -> Tensor:
    """
    Generate initial weights for OPR modes, where the weights of the main OPR mode are set to 1,
    and the weights of eigenmodes are set to 0.

    Parameters
    ----------
    n_points : int
        number of scan points.
    n_opr_modes : int
        number of OPR modes.
    eigenmode_weight : float
        initial weight for eigenmodes.

    Returns
    -------
    weights : Tensor
        a (n_points, n_opr_modes) tensor of weights.
    """
    return torch.cat(
        [torch.ones([n_points, 1]), torch.full([n_points, n_opr_modes - 1], eigenmode_weight)],
        dim=1,
    )


def generate_gaussian_random_image(
    shape: tuple[int, ...], loc: float = 0.9, sigma: float = 0.1, smoothing: float = 3.0
) -> Tensor:
    img = torch.randn(shape, dtype=torch.get_default_dtype()) * sigma + loc
    if smoothing > 0.0:
        img = GaussianBlur(kernel_size=(9, 9), sigma=(3, 3))(img[None, None, :, :])
        img = img[0, 0, ...]
    return img


def to_tensor(data: Union[ndarray, Tensor], device=None, dtype=None) -> Tensor:
    if device is None:
        device = torch.get_default_device()
    if isinstance(data, (np.ndarray, list, tuple)):
        data = torch.tensor(data, device=device)

    if dtype is None:
        if data.dtype.is_complex:
            dtype = get_default_complex_dtype()
        elif not data.dtype.is_complex:
            dtype = torch.get_default_dtype()

    if data.dtype != dtype:
        data = data.type(dtype)
    if str(data.device) != str(device):
        data = data.to(device)
    return data


def to_numpy(data: Union[ndarray, Tensor]) -> ndarray:
    if isinstance(data, Tensor):
        data = data.detach().cpu().numpy()
    return data


def set_default_complex_dtype(dtype):
    global default_complex_dtype
    default_complex_dtype = dtype


def get_default_complex_dtype():
    return default_complex_dtype


def chunked_processing(
    func: Callable,
    common_kwargs: dict,
    chunkable_kwargs: dict,
    iterated_kwargs: dict,
    chunk_size: int = 96,
):
    """
    Parameters
    ----------
    func : callable
        The callable to be executed.
    common_kwargs : dict
        A dictionary of arguments that should stay constant across chunks.
    chunkable_kwargs : dict
        A dictionary of arguments that should be chunked.
    iterated_kwargs : dict
        A dictionary of arguments that should be returned by `func`, then passed to `func`
        for the next chunk. The order of arguments should be the same as the returns of
        `func`.
    chunk_size : int, optional
        The size of each chunk. Default is 96.

    Returns
    -------
    The returns of `func` as if it is executed for the entire data.
    """
    full_batch_size = tuple(chunkable_kwargs.values())[0].shape[0]
    for key, value in tuple(chunkable_kwargs.items())[1:]:
        if value.shape[0] != full_batch_size:
            raise ValueError(
                "All chunkable arguments must have the same batch size, but {} \
                has shape {}.".format(key, value.shape)
            )

    chunks_of_chunkable_args = []
    ind_st = 0
    while ind_st < full_batch_size:
        ind_end = min(ind_st + chunk_size, full_batch_size)
        chunk = {key: value[ind_st:ind_end] for key, value in chunkable_kwargs.items()}
        chunks_of_chunkable_args.append(chunk)
        ind_st = ind_end

    for kwargs_chunk in chunks_of_chunkable_args:
        ret = func(**common_kwargs, **kwargs_chunk, **iterated_kwargs)
        if isinstance(ret, tuple):
            for i, key in enumerate(iterated_kwargs.keys()):
                iterated_kwargs[key] = ret[i]
        else:
            iterated_kwargs[tuple(iterated_kwargs.keys())[0]] = ret
    if len(iterated_kwargs) == 1:
        return tuple(iterated_kwargs.values())[0]
    else:
        return tuple(iterated_kwargs.values())
