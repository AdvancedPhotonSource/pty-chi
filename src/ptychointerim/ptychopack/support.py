from typing import Final, TypeAlias

import torch

BooleanTensor: TypeAlias = torch.Tensor
ComplexTensor: TypeAlias = torch.Tensor
RealTensor: TypeAlias = torch.Tensor


def squared_modulus(values: ComplexTensor) -> RealTensor:
    return torch.abs(values)**2


def project(u: ComplexTensor, v: ComplexTensor) -> ComplexTensor:
    return u * torch.vdot(u, v) / torch.vdot(u, u)


def gram_schmidt(V: ComplexTensor) -> ComplexTensor:
    U = V.detach().clone()

    for i in range(1, V.shape[0]):
        for j in range(i):
            U[i, :] -= project(U[j, :], V[i, :])

    return U


def orthogonalize_probe(self, probes: ComplexTensor) -> ComplexTensor:
    probes_temp = gram_schmidt(probes.reshape(N_probe, N_roi**2))
    probes[:, :, :] = probes_temp.reshape(N_probe, N_roi, N_roi)
    #sort probes based on power
    power = torch.sum(squared_modulus(probes), dim=(-2, -1))
    power_ind = torch.argsort(-power)
    probes[:, :, :] = probes[power_ind, :, :]
    return probes


def correct_positions_serial_cross_correlation(object_patch1: ComplexTensor,
                                               object_patch2: ComplexTensor,
                                               scale: float) -> RealTensor:
    # fourier transform images
    F = torch.fft.fft2(object_patch1)
    G = torch.fft.fft2(object_patch2)

    # start setting up cross-correlation
    FG = F * G.conj()

    # set up spatial and frequency coordinates
    N, M = object_patch1.shape  # FIXME
    x = torch.linspace(-10. / scale, 10. / scale, 21).reshape((1, 21))
    y = torch.linspace(-10. / scale, 10. / scale, 21).reshape((21, 1))
    u = torch.linspace(-.5 + 1. / M * .5, .5 - 1. / M * .5, M).reshape((M, 1))
    v = torch.linspace(-.5 + 1. / N * .5, .5 - 1. / N * .5, N).reshape((1, N))

    # perform the inverse fourier transform to obtain zoomed-in correlation
    corr2 = torch.dot(torch.exp(2j * torch.pi * torch.dot(y, v)),
                      torch.dot(FG, torch.exp(2j * torch.pi * torch.dot(u, x))))

    # find the peak of the correlation
    xmax2 = torch.argmax(torch.max(torch.abs(corr2), dim=0))
    ymax2 = torch.argmax(torch.max(torch.abs(corr2), dim=1))

    shift = torch.tensor([x[0, xmax2], y[ymax2, 0]])

    return shift


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
        xmin_fr_c = 1. - xmin_fr
        ymin_fr_c = 1. - ymin_fr

        # barycentric interpolant weights
        self._weight00 = ymin_fr_c * xmin_fr_c
        self._weight01 = ymin_fr_c * xmin_fr
        self._weight10 = ymin_fr * xmin_fr_c
        self._weight11 = ymin_fr * xmin_fr

        # extract patch support region from full object
        self._object_support = object_[ymin_wh:ymax_wh, xmin_wh:xmax_wh]

    def get_patch(self) -> ComplexTensor:
        '''interpolate object support to extract patch'''
        object_patch = self._weight00 * self._object_support[:-1, :-1]
        object_patch += self._weight01 * self._object_support[:-1, 1:]
        object_patch += self._weight10 * self._object_support[1:, :-1]
        object_patch += self._weight11 * self._object_support[1:, 1:]
        return object_patch

    def update_patch(self, object_update: ComplexTensor) -> None:
        '''add patch update to object support'''
        self._object_support[:-1, :-1] += self._weight00 * object_update
        self._object_support[:-1, 1:] += self._weight01 * object_update
        self._object_support[1:, :-1] += self._weight10 * object_update
        self._object_support[1:, 1:] += self._weight11 * object_update
