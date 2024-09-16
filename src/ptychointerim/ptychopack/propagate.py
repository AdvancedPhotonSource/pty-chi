from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import cmath

import torch
from torch.fft import fft2, fftfreq, fftshift, ifft2, ifftshift

from .device import Device
from .support import ComplexTensor, RealTensor


@dataclass(frozen=True)
class WavefieldPropagatorParameters:
    width_px: int
    """number of pixels in the x-direction"""
    height_px: int
    """number of pixels in the y-direction"""
    pixel_width_wlu: float
    """pixel width in wavelengths"""
    pixel_aspect_ratio: float
    """pixel aspect ratio (width / height)"""
    propagation_distance_wlu: float
    """propagation distance in wavelengths"""

    @classmethod
    def create_simple(
        cls,
        wavelength_m: float,
        width_px: int,
        height_px: int,
        pixel_width_m: float,
        pixel_height_m: float,
        propagation_distance_m: float,
    ) -> WavefieldPropagatorParameters:
        """Creates propagator paramaters dataclass from quantities with physical length units.
        :param wavelength_m: illumination wavelength in meters
        :type wavelength_m: float
        :param width_px: number of pixels in the x-direction
        :type width_px: int
        :param height_px: number of pixels in the y-direction
        :type height_px: int
        :param pixel_width_m: source plane pixel width in meters
        :type pixel_width_m: float
        :param pixel_height_m: source plane pixel height in meters
        :type pixel_height_m: float
        :param propagation_distance_m: propagation distance in meters
        :type propagation_distance_m: float
        :returns: a dataclass that contains the nondimensionalized propagator parameters
        :rtype: WavefieldPropagatorParameters
        """
        return cls(
            width_px,
            height_px,
            pixel_width_m / wavelength_m,
            pixel_width_m / pixel_height_m,
            propagation_distance_m / wavelength_m,
        )

    @property
    def fresnel_number(self) -> float:
        pixel_width_wlu_sq = self.pixel_width_wlu * self.pixel_width_wlu
        return pixel_width_wlu_sq / self.propagation_distance_wlu

    def get_spatial_coordinates(self) -> tuple[RealTensor, RealTensor]:
        ii = torch.arange(self.width_px)
        jj = torch.arange(self.height_px)
        JJ, II = torch.meshgrid(jj, ii, indexing="ij")
        XX = II - self.width_px // 2
        YY = JJ - self.height_px // 2
        return YY, XX

    def get_frequency_coordinates(self) -> tuple[RealTensor, RealTensor]:
        fx = fftfreq(self.width_px)
        fy = fftfreq(self.height_px)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")
        return FY, FX


class WavefieldPropagator(ABC):

    @abstractmethod
    def propagate_forward(self, wavefield: ComplexTensor) -> ComplexTensor:
        pass

    @abstractmethod
    def propagate_backward(self, wavefield: ComplexTensor) -> ComplexTensor:
        pass

    @abstractmethod
    def to(self, device: Device) -> WavefieldPropagator:
        pass

    def cpu(self) -> WavefieldPropagator:
        return self.to(Device.CPU())


class FourierPropagator(WavefieldPropagator):

    def propagate_forward(self, wavefield: ComplexTensor) -> ComplexTensor:
        return fft2(wavefield)

    def propagate_backward(self, wavefield: ComplexTensor) -> ComplexTensor:
        return ifft2(wavefield)

    def to(self, device: Device) -> WavefieldPropagator:
        return self


class AngularSpectrumPropagator(WavefieldPropagator):

    def __init__(self, parameters: WavefieldPropagatorParameters) -> None:
        ar = parameters.pixel_aspect_ratio

        i2piz = 2j * torch.pi * parameters.propagation_distance_wlu
        FY, FX = parameters.get_frequency_coordinates()
        F2 = torch.square(FX) + torch.square(ar * FY)
        ratio = F2 / (parameters.pixel_width_wlu**2)
        tf = torch.exp(i2piz * torch.sqrt(1 - ratio))

        self._transfer_function = torch.where(ratio < 1, tf, 0)

    def propagate_forward(self, wavefield: ComplexTensor) -> ComplexTensor:
        return ifft2(self._transfer_function * fft2(wavefield))

    def propagate_backward(self, wavefield: ComplexTensor) -> ComplexTensor:
        return ifft2(fft2(wavefield) / self._transfer_function)


class FresnelTransformPropagator(WavefieldPropagator):

    def __init__(self, parameters: WavefieldPropagatorParameters) -> None:
        ipi = 1j * torch.pi

        Fr = parameters.fresnel_number
        ar = parameters.pixel_aspect_ratio
        N = parameters.width_px
        M = parameters.height_px
        YY, XX = parameters.get_spatial_coordinates()

        self._C0 = Fr / (1j * ar)
        self._C1 = cmath.exp(2j * cmath.pi * parameters.propagation_distance_wlu)
        self._C2 = torch.exp((torch.square(XX / N) + torch.square(ar * YY / M)) * ipi / Fr)

        self._B = torch.exp(ipi * Fr * (torch.square(XX) + torch.square(YY / ar)))

    def propagate_forward(self, wavefield: ComplexTensor) -> ComplexTensor:
        A = self._C2 * self._C1 * self._C0
        return A * fft2(wavefield * self._B)

    def propagate_backward(self, wavefield: ComplexTensor) -> ComplexTensor:
        A = self._C2 * self._C1 / self._C0
        return self._B * ifft2(wavefield * A)


class FraunhoferPropagator(WavefieldPropagator):

    def __init__(self, parameters: WavefieldPropagatorParameters) -> None:
        ipi = 1j * torch.pi

        Fr = parameters.fresnel_number
        ar = parameters.pixel_aspect_ratio
        N = parameters.width_px
        M = parameters.height_px
        YY, XX = parameters.get_spatial_coordinates()

        self._C0 = Fr / (1j * ar)
        self._C1 = cmath.exp(2j * cmath.pi * parameters.propagation_distance_wlu)
        self._C2 = torch.exp((torch.square(XX / N) + torch.square(ar * YY / M)) * ipi / Fr)

    def propagate_forward(self, wavefield: ComplexTensor) -> ComplexTensor:
        A = self._C2 * self._C1 * self._C0
        return A * fft2(wavefield)

    def propagate_backward(self, wavefield: ComplexTensor) -> ComplexTensor:
        A = self._C2 * self._C1 / self._C0
        return ifft2(wavefield * A)
