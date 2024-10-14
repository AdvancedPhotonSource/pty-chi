from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ptychointerim")
except PackageNotFoundError:
    pass

from .device import Device, list_available_devices
import ptychointerim.ptychopack
import ptychointerim.ptychotorch

__all__ = [
    "Device",
    "list_available_devices",
    "ptychopack",
    "ptychotorch",
]
