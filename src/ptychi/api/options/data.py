from typing import Union, Optional

from numpy import ndarray, inf
from torch import Tensor


import dataclasses


@dataclasses.dataclass
class PtychographyDataOptions:

    data: Union[ndarray, Tensor] = None
    """
    The intensity data. Use collected data as they are; data should NOT be FFT-shifted 
    or square-rooted.
    """

    free_space_propagation_distance_m: float = inf
    """The free-space propagation distance in meters, or `inf` for far-field."""

    wavelength_m: float = 1e-9
    """The wavelength in meters."""

    detector_pixel_size_m: float = 1e-8
    """The detector pixel size in meters."""

    valid_pixel_mask: Optional[Union[ndarray, Tensor]] = None
    """A 2D boolean mask where valid pixels are True."""