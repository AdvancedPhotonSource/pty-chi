import torch

from .api import ComplexTensor, RealTensor


def squared_modulus(values: ComplexTensor) -> RealTensor:
    return torch.abs(values) ** 2


class ObjectPatchInterpolator:
    def __init__(self, object_: ComplexTensor, position_px: RealTensor, size: torch.Size) -> None:
        # top left corner of object support
        xmin = position_px[-1] - size[-1] / 2
        ymin = position_px[-2] - size[-2] / 2

        # whole components (pixel indexes)
        xmin_wh = xmin.int()
        ymin_wh = ymin.int()

        # fractional (subpixel) components
        xmin_fr = xmin - xmin_wh
        ymin_fr = ymin - ymin_wh

        # bottom right corner of object patch support
        xmax_wh = xmin_wh + size[-1] + 1
        ymax_wh = ymin_wh + size[-2] + 1

        # reused quantities
        xmin_fr_c = 1.0 - xmin_fr
        ymin_fr_c = 1.0 - ymin_fr

        # barycentric interpolant weights
        self._weight00 = ymin_fr_c * xmin_fr_c
        self._weight01 = ymin_fr_c * xmin_fr
        self._weight10 = ymin_fr * xmin_fr_c
        self._weight11 = ymin_fr * xmin_fr

        # extract patch support region from full object
        self._object_support = object_[ymin_wh:ymax_wh, xmin_wh:xmax_wh]

    def get_patch(self) -> ComplexTensor:
        """interpolate object support to extract patch"""
        object_patch = self._weight00 * self._object_support[:-1, :-1]
        object_patch += self._weight01 * self._object_support[:-1, 1:]
        object_patch += self._weight10 * self._object_support[1:, :-1]
        object_patch += self._weight11 * self._object_support[1:, 1:]
        return object_patch

    def update_patch(self, object_update: ComplexTensor) -> None:
        """add patch update to object support"""
        self._object_support[:-1, :-1] += self._weight00 * object_update
        self._object_support[:-1, 1:] += self._weight01 * object_update
        self._object_support[1:, :-1] += self._weight10 * object_update
        self._object_support[1:, 1:] += self._weight11 * object_update
