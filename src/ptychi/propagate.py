from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias
import cmath

import torch
from torch.fft import fft2, fftfreq, ifft2

BooleanTensor: TypeAlias = torch.Tensor
ComplexTensor: TypeAlias = torch.Tensor
RealTensor: TypeAlias = torch.Tensor


@dataclass(frozen=True)
class WavefieldPropagatorParameters:
    width_px: int
    """number of pixels in the x-direction"""
    height_px: int
    """number of pixels in the y-direction"""
    wavelength_m: float
    """illumination wavelength in meters"""
    pixel_width_m: float
    """pixel width in meters"""
    pixel_width_wlu: float
    """pixel width in wavelengths"""
    pixel_aspect_ratio: float
    """pixel aspect ratio (width / height)"""
    propagation_distance_m: float
    """propagation distance in meters"""
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
        """
        Creates propagator paramaters dataclass from quantities with physical length units.

        Parameters
        ----------
        wavelength_m : float
            Illumination wavelength in meters
        width_px : int
            Number of pixels in the x-direction
        height_px : int
            Number of pixels in the y-direction
        pixel_width_m : float
            Source plane pixel width in meters
        pixel_height_m : float
            Source plane pixel height in meters
        propagation_distance_m : float
            Propagation distance in meters

        Returns
        -------
        WavefieldPropagatorParameters
            a dataclass that contains the nondimensionalized propagator parameters
        """
        return cls(
            width_px,
            height_px,
            wavelength_m,
            pixel_width_m,
            pixel_width_m / wavelength_m,
            pixel_width_m / pixel_height_m,
            propagation_distance_m,
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
        XX = XX.to(torch.get_default_device())
        YY = YY.to(torch.get_default_device())
        return YY, XX

    def get_frequency_coordinates(self) -> tuple[RealTensor, RealTensor]:
        fx = fftfreq(self.width_px)
        fy = fftfreq(self.height_px)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")
        FY = FY.to(torch.get_default_device())
        FX = FX.to(torch.get_default_device())
        return FY, FX


class WavefieldPropagator(ABC, torch.nn.Module):
    @abstractmethod
    def propagate_forward(self, wavefield: ComplexTensor) -> ComplexTensor:
        pass

    @abstractmethod
    def propagate_backward(self, wavefield: ComplexTensor) -> ComplexTensor:
        pass

    def forward(self, wavefield: ComplexTensor) -> ComplexTensor:
        return self.propagate_forward(wavefield)


class FourierPropagator(WavefieldPropagator):
    def __init__(self, norm=None) -> None:
        super().__init__()
        self.norm = norm

    def propagate_forward(self, wavefield: ComplexTensor) -> ComplexTensor:
        return fft2(wavefield, norm=self.norm)

    def propagate_backward(self, wavefield: ComplexTensor) -> ComplexTensor:
        return ifft2(wavefield, norm=self.norm)


class AngularSpectrumPropagator(WavefieldPropagator):
    def __init__(self, parameters: WavefieldPropagatorParameters) -> None:
        super().__init__()

        _transfer_function = self.get_transfer_function(parameters)
        
        # Separate registered buffer into real and imaginary parts to prevent it
        # from breaking in DataParallel.
        self.register_buffer('_transfer_function_real', _transfer_function.real)
        self.register_buffer('_transfer_function_imag', _transfer_function.imag)

    def update(self, parameters: WavefieldPropagatorParameters) -> None:
        _transfer_function = self.get_transfer_function(parameters)
        self._transfer_function_real[...] = _transfer_function.real
        self._transfer_function_imag[...] = _transfer_function.imag
        
    def get_transfer_function(self, parameters: WavefieldPropagatorParameters) -> ComplexTensor:
        k = 2 * torch.pi / parameters.wavelength_m
        nx = parameters.width_px
        ny = parameters.height_px
        ygrid, xgrid = torch.fft.fftfreq(ny), torch.fft.fftfreq(nx)
        kx = 2 * torch.pi * xgrid / parameters.pixel_width_m
        ky = 2 * torch.pi * ygrid / (parameters.pixel_width_m * parameters.pixel_aspect_ratio)
        kyy, kxx = torch.meshgrid(ky, kx, indexing="ij")
        sqrt_arg = k ** 2 - kxx ** 2 - kyy ** 2
        tf = torch.exp(1j * parameters.propagation_distance_m * torch.sqrt(sqrt_arg))
        tf = torch.where(sqrt_arg < 0, 0, tf)
        return tf

    def propagate_forward(self, wavefield: ComplexTensor) -> ComplexTensor:
        tf = self._transfer_function_real + 1j * self._transfer_function_imag
        return ifft2(tf * fft2(wavefield))

    def propagate_backward(self, wavefield: ComplexTensor) -> ComplexTensor:
        tf = self._transfer_function_real + 1j * self._transfer_function_imag
        return ifft2(fft2(wavefield) / tf)


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
