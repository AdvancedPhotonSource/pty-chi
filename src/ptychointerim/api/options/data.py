from typing import Union, Optional

from numpy import ndarray
from torch import Tensor


import dataclasses


@dataclasses.dataclass
class PtychographyDataOptions:

    data: Union[ndarray, Tensor] = None
    """The data."""

    propagation_distance_m: float = 1.0
    """The propagation distance in meters."""

    wavelength_m: float = 1e-9
    """The wavelength in meters."""

    detector_pixel_size_m: float = 1e-8
    """The detector pixel size in meters."""

    valid_pixel_mask: Optional[Union[ndarray, Tensor]] = None
    """A 2D boolean mask where valid pixels are True."""