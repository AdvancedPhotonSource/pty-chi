# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ptychi")
except PackageNotFoundError:
    pass

from .device import Device, list_available_devices

__all__ = [
    "Device",
    "list_available_devices",
]
