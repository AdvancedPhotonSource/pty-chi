# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Union, Optional
import dataclasses

from numpy import ndarray, inf
from torch import Tensor

import ptychi.api.options.base as base


__all__ = ["PtychographyDataOptions"]


@dataclasses.dataclass
class PtychographyDataOptions(base.Options):

    data: Union[ndarray, Tensor] = None
    """
    The intensity data. Use collected data as they are; data should NOT be FFT-shifted 
    or square-rooted.
    """

    free_space_propagation_distance_m: float = inf
    """The free-space propagation distance in meters, or `inf` for far-field."""

    wavelength_m: float = 1e-9
    """The wavelength in meters."""

    fft_shift: bool = True
    """Whether to FFT-shift the diffraction data."""

    detector_pixel_size_m: float = 1e-8
    """The detector pixel size in meters."""

    valid_pixel_mask: Optional[Union[ndarray, Tensor]] = None
    """A 2D boolean mask where valid pixels are True."""
    
    save_data_on_device: bool = False
    """Whether to save the diffraction data on acceleration devices like GPU."""
