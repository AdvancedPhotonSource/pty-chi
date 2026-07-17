# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Optional
from pydantic.dataclasses import dataclass

from numpy import inf

import ptychi.api.options.base as base


__all__ = ["PtychographyDataOptions"]


@dataclass
class PtychographyDataOptions(base.Options):

    data: Optional[base.DataArray] = None
    """
    The intensity data. Use collected data as they are; data should NOT be FFT-shifted 
    or square-rooted.
    """

    free_space_propagation_distance_m: float = inf
    """The free-space propagation distance in meters, or `inf` for far-field."""

    wavelength_m: float = 1e-9
    """The wavelength in meters."""

    fft_shift: bool = True
    """Whether to FFT-shift the diffraction data when building the dataset. For far-field
    ptychography, the forward model does not shift the image after FFT, meaning the
    predicted intensity has its DC component at the top left corner. To match the prediction,
    measured intensity should be pre-shifted if the DC component of the given diffraction
    patterns is at the center. 
    
    However, if the given diffraction patterns are already shifted so that the DC compoenent
    is at the top left, or if you are reconstructing near-field ptychography data where
    the forward model does not involve Fraunhofer diffraction implemented with FFT, ensure
    this option is set to `False` to avoid the erroneous shifting. 
    """

    detector_pixel_size_m: float = 1e-8
    """The detector pixel size in meters."""

    valid_pixel_mask: Optional[base.DataArray] = None
    """A 2D boolean mask where valid pixels are True."""
    
    save_data_on_device: bool = False
    """Whether to save the diffraction data on acceleration devices like GPU."""
    
    def get_non_data_fields(self) -> dict:
        d = self.__dict__.copy()
        del d['data']
        del d['valid_pixel_mask']
        return d
