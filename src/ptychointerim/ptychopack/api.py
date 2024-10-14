from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy
import numpy.typing
import torch

from ptychointerim.propagate import FourierPropagator, WavefieldPropagator

BooleanArray: TypeAlias = numpy.typing.NDArray[numpy.bool_]
IntegerArray: TypeAlias = numpy.typing.NDArray[numpy.integer[Any]]
RealArray: TypeAlias = numpy.typing.NDArray[numpy.floating[Any]]
ComplexArray: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]

BooleanTensor: TypeAlias = torch.Tensor
ComplexTensor: TypeAlias = torch.Tensor
RealTensor: TypeAlias = torch.Tensor


@dataclass(frozen=True)
class CorrectionPlanElement:
    start: int
    stop: int
    stride: int

    def is_enabled(self, iteration: int) -> bool:
        if self.start <= iteration and iteration < self.stop:
            return (iteration - self.start) % self.stride == 0

        return False


@dataclass(frozen=True)
class CorrectionPlan:
    object_correction: CorrectionPlanElement
    probe_correction: CorrectionPlanElement
    position_correction: CorrectionPlanElement

    @property
    def number_of_iterations(self) -> int:
        return max(
            self.object_correction.stop,
            self.probe_correction.stop,
            self.position_correction.stop,
        )


@dataclass(frozen=True)
class DetectorData:
    diffraction_patterns: RealTensor  # [N, H, W]
    """diffraction patterns"""
    bad_pixels: BooleanTensor  # [H, W]
    """bad pixel mask to exclude detector dead regions or saturated pixels"""

    @classmethod
    def from_numpy(
        cls,
        diffraction_patterns: numpy.typing.NDArray[numpy.integer[Any]],
        bad_pixels: numpy.typing.NDArray[numpy.bool_] | None = None,
    ) -> DetectorData:
        diffraction_patterns_tensor = torch.tensor(
            diffraction_patterns, dtype=torch.float, requires_grad=False
        )

        return cls(
            torch.fft.ifftshift(diffraction_patterns_tensor, dim=(-2, -1)),
            torch.full(diffraction_patterns.shape[1:], False)
            if bad_pixels is None
            else torch.tensor(bad_pixels),
        )


@dataclass(frozen=True)
class DataProduct:
    positions_px: RealTensor  # [N, 2]
    """positions are in pixel units"""
    probe: ComplexTensor  # [C, I, H, W]
    """probe shape is the number of coherent (OPR) modes, number of incoherent
    (mixed-states) modes, height in pixels, and width in pixels"""
    object_: ComplexTensor  # [L, H, W]
    """object shape is the number of (multi-slice) layers, height in pixels,
    and width in pixels"""
    propagators: Sequence[WavefieldPropagator]  # [L]
    """sequence of wavefield propagators, one for each object layer"""

    @classmethod
    def from_numpy(
        cls, positions_px: RealArray, probe: ComplexArray, object_: ComplexArray
    ) -> DataProduct:
        propagators = [FourierPropagator()]
        return cls(
            torch.tensor(positions_px, dtype=torch.float, requires_grad=False),
            torch.tensor(probe, dtype=torch.cfloat, requires_grad=False),
            torch.tensor(object_, dtype=torch.cfloat, requires_grad=False),
            propagators,
        )


class IterativeAlgorithm(ABC):
    @abstractmethod
    def iterate(self, plan: CorrectionPlan) -> Sequence[float]:
        pass

    @abstractmethod
    def get_product(self) -> DataProduct:
        pass
