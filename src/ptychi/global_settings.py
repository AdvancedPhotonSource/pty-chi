# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import os
import torch

_default_complex_dtype = torch.complex64


def set_default_complex_dtype(dtype):
    """Set the default complex dtype.

    Parameters
    ----------
    dtype : torch.dtype
        The default complex dtype.
    """
    global _default_complex_dtype
    _default_complex_dtype = dtype


def get_default_complex_dtype():
    """Get the default complex dtype.

    Returns
    -------
    torch.dtype
        The default complex dtype.
    """
    return _default_complex_dtype

