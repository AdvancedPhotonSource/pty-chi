from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ptychopack")
except PackageNotFoundError:
    pass

from .api import (CorrectionPlan, CorrectionPlanElement, DataProduct, DetectorData,
                  IterativeAlgorithm)
from .device import list_available_devices, Device
from .dm import DifferenceMap
from .pie import PtychographicIterativeEngine
from .raar import RelaxedAveragedAlternatingReflections

__all__ = [
    "CorrectionPlan",
    "CorrectionPlanElement",
    "DataProduct",
    "DetectorData",
    "Device",
    "DifferenceMap",
    "IterativeAlgorithm",
    "PtychographicIterativeEngine",
    "RelaxedAveragedAlternatingReflections",
    "list_available_devices",
]
