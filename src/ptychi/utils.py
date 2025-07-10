# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Union, Literal, Callable, Optional, Sequence, TYPE_CHECKING
import math
import gc

import torch
from torch import Tensor
from torchvision.transforms import GaussianBlur
import numpy as np
from numpy import ndarray

import ptychi.maths as pmath
import ptychi.propagate as propagate
from ptychi.timing.timer_utils import timer

if TYPE_CHECKING:
    from ptychi.api.task import PtychographyTask


_default_complex_dtype = torch.complex64


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
        A (n_points, n_opr_modes) tensor of weights for each OPR mode.

    Returns
    -------
    scaled_probe : Tensor
        The scaled probe.
    """
    propagator = propagate.FourierPropagator()

    probe_tensor = torch.tensor(probe)
    
    if probe_tensor.ndim == 4:
        if probe_tensor.shape[0] == 1 or weights is None:
            probe_tensor = probe_tensor[0]

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
        weights = weights.mean(dim=0) 
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
        A (n_opr_modes, n_modes, h, w) tensor of the probe. This function only generates
        incoherent modes; OPR modes are kept as they are. Only the first incoherent mode
        of the input probe is used. As such, the rest of the incoherent modes can be
        arbotrarily initialized, but the shape of the input probe should be indicate the
        number of incoherent modes intended to be generated.
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


@timer()
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


@timer()
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


@timer()
def add_additional_opr_probe_modes_to_probe(
    probe: Tensor, n_opr_modes_to_add: int, normalize: bool = True
) -> Tensor:
    """
    Add additional OPR modes to the probe.
    
    Parameters
    ----------
    probe : Tensor
        A (n_opr_modes, n_modes, h, w) tensor of the probe.
    n_opr_modes_to_add : int
        The number of OPR modes to add.
    normalize : bool, optional
        Whether to normalize the OPR modes using `mnorm` so that the power
        of each mode is the number of pixels in a mode.
    
    Returns
    -------
    Tensor
        A (n_opr_modes + n_opr_modes_to_add, n_modes, h, w) tensor of the probe 
        with additional OPR modes.
    """
    if probe.ndim != 4:
        raise ValueError("probe must be a (n_opr_modes, n_modes, h, w) tensor.")
    n_modes = probe.shape[1]
    opr_modes = torch.empty(
        [n_opr_modes_to_add, n_modes, probe.shape[-2], probe.shape[-1]],
        dtype=get_default_complex_dtype(),
    )
    
    for i in range(n_opr_modes_to_add):
        for j in range(n_modes):
            real = generate_gaussian_random_image(
                probe.shape[-2:], loc=0, sigma=1, smoothing=0
            )
            imag = generate_gaussian_random_image(
                probe.shape[-2:], loc=0, sigma=1, smoothing=0
            )
            opr_mode = real + 1j * imag
            opr_modes[i, j, ...] = opr_mode
    probe = torch.cat([probe, opr_modes], dim=0)
    
    if normalize:
        pnorm = pmath.mnorm(probe, dim=(-2, -1), keepdims=True)
        probe[1:] = probe[1:] / pnorm[1:, :]
    return probe


@timer()
def generate_initial_opr_mode_weights(
    n_points: int, n_opr_modes: int, eigenmode_weight: Optional[float] = None, probe: Optional[Tensor] = None
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
    probe: Tensor
        The probe. If provided, the weights will be normalized to match the power of the probe.

    Returns
    -------
    weights : Tensor
        a (n_points, n_opr_modes) tensor of weights.
    """
    if eigenmode_weight is None:
        eigenmode_weights = torch.randn([n_points, n_opr_modes - 1]) * 1e-6
    else:
        eigenmode_weights = torch.full([n_points, n_opr_modes - 1], eigenmode_weight)
    weights = torch.cat(
        [torch.ones([n_points, 1]), eigenmode_weights],
        dim=1,
    )
    if probe is not None:
        pnorm = pmath.mnorm(probe, dim=(-2, -1), keepdims=False)
        weights[:, 1:] = weights[:, 1:] / torch.mean(pnorm[1:], dim=1)
    return weights


@timer()
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
    global _default_complex_dtype
    _default_complex_dtype = dtype


def get_default_complex_dtype():
    return _default_complex_dtype


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


def calculate_data_size_gb(
    shape: Sequence[int], 
    dtype: torch.dtype | np.dtype
) -> float:
    """
    Calculate the size of the data in GB.
    """
    shape = list(shape)
    return np.prod(shape) * dtype.itemsize / 1024 ** 3


def get_max_batch_size(
    probe_shape: Sequence[int], 
    object_shape: Sequence[int],
    double_precision: bool = False, 
    data_saved_on_device: bool = False, 
    all_data_shape: Sequence[int] = None,
    reconstructor_type: Literal["lsqml"] = "lsqml",
    margin_factor: float = 0.2
) -> int:
    """
    Estimate the maximum batch size that fits in the available device memory.
    
    We estimate the memory usage using an empirical formula:
    ```
    mem = x0 * n_p * batch_size + x1 * n_p + x2 * object_numel
    ```
    where `n_p = n_modes * probe_size ** 2` and the coefficients `x0`, `x1`, `x2` 
    were fit from experimental data.
    
    Parameters
    ----------
    probe_shape : Sequence[int]
        The shape of the 4D probe, expected to be (n_opr_modes, n_modes, h, w).
    object_shape : Sequence[int]
        The shape of the object, expected to be (n_slcies, h, w).
    double_precision : bool, optional
        Whether to use double precision.
    data_saved_on_device : bool, optional
        Whether the raw data is kept on device.
    all_data_shape : Sequence[int], optional
        The shape of the data, expected to be (n_points, h, w).
    reconstructor_type : Literal["lsqml"], optional
        The type of reconstructor. Currently only `lsqml` is supported.
    margin_factor : float, optional
        The fraction of the device memory to be left free.
    
    Returns
    -------
    int
        The suggested batch size.
    """
    if reconstructor_type == "lsqml":
        x0 = 6.26e-8
        x1 = 3.73e-7
        x2 = 1.39e-7
    else:
        raise ValueError(f"Unknown reconstructor type: {reconstructor_type}")
    
    dtype = torch.float64 if double_precision else torch.float32
    n_p = np.prod(list(probe_shape[1:]))
    n_o = np.prod(list(object_shape))
    if data_saved_on_device:
        data_size_gb = calculate_data_size_gb(all_data_shape, dtype)
    else:
        data_size_gb = 0.0
        
    mem_avail = torch.cuda.mem_get_info()[0] * (1 - margin_factor) / 1024 ** 3
    mem_compute = mem_avail - data_size_gb
    batch_size = (mem_compute - x1 * n_p - x2 * n_o) / (x0 * n_p)
    batch_size = batch_size * (8 / dtype.itemsize)
    return max(int(batch_size), 1)


def auto_transfer_to_device(data: Tensor) -> Tensor:
    """Automatically determine the device that the data should be placed on, 
    and transfer the data to that device.
    
    The logic of this function is as follows:
    1. If `torch.get_default_device()` is `cuda`, transfer the data to `cuda`.
    2. If `torch.get_default_device()` is `cpu`, it could either be GPU is unavailable or
       intentionally disabled, OR the current code is executed by DataParallel. 
    2.1. If `torch.cuda.device_count()` is 0, we assume it is the former case, and
         we keep the data as is.
    2.2. If `torch.cuda.device_count()` is not 0, we assume it is the latter case, and
         we transfer the data to `cuda`.
    """
    if torch.get_default_device().type == "cuda":
        return data.cuda()
    else:
        if torch.cuda.device_count() == 0:
            return data
        else:
            return data.cuda()


def clear_memory(task: Optional["PtychographyTask"] = None):
    """Clear the memory of the device used. If a `Task` object is provided,
    it will be deleted and the memory will be released.
    
    Parameters
    ----------
    task : PtychographyTask, optional
        The `Task` object to be deleted.
    """
    if task is not None:
        del task
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def jsonize(val):
    """Convert a value to a JSON-serializable object."""
    if isinstance(val, np.generic):
        return val.item()
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, torch.Tensor):
        return val.tolist()
    elif isinstance(val, (list, tuple, dict, str, int, float, bool, type(None))):
        return val
    else:
        raise TypeError(f"Object of type {type(val).__name__} is not JSON serializable")