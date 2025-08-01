# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias
import cmath
import math

import torch
from torch.fft import fftfreq

from ptychi.global_settings import get_default_complex_dtype
from ptychi.timing.timer_utils import timer
import ptychi.maths as pmath
import ptychi.utils as utils


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
            width_px=width_px,
            height_px=height_px,
            wavelength_m=wavelength_m,
            pixel_width_wlu=pixel_width_m / wavelength_m,
            pixel_aspect_ratio=pixel_width_m / pixel_height_m,
            propagation_distance_wlu=propagation_distance_m / wavelength_m,
        )

    @property
    def fresnel_number(self) -> float:
        pixel_width_wlu_sq = self.pixel_width_wlu * self.pixel_width_wlu
        return pixel_width_wlu_sq / self.propagation_distance_wlu
    
    @property
    def pixel_width_m(self) -> float:
        return self.pixel_width_wlu * self.wavelength_m
    
    @property
    def pixel_height_m(self) -> float:
        return self.pixel_width_m / self.pixel_aspect_ratio
    
    @property
    def pixel_height_wlu(self) -> float:
        return self.pixel_width_wlu / self.pixel_aspect_ratio
    
    @property
    def propagation_distance_m(self) -> float:
        return self.propagation_distance_wlu * self.wavelength_m

    def get_spatial_coordinates(self) -> tuple[RealTensor, RealTensor]:
        ii = torch.arange(self.width_px)
        jj = torch.arange(self.height_px)
        JJ, II = torch.meshgrid(jj, ii, indexing="ij")
        XX = II - self.width_px // 2
        YY = JJ - self.height_px // 2
        XX = utils.auto_transfer_to_device(XX)
        YY = utils.auto_transfer_to_device(YY)
        return YY, XX

    def get_frequency_coordinates(self) -> tuple[RealTensor, RealTensor]:
        fx = fftfreq(self.width_px)
        fy = fftfreq(self.height_px)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")
        FY = utils.auto_transfer_to_device(FY)
        FX = utils.auto_transfer_to_device(FX)
        return FY.detach(), FX.detach()
    
    def is_fresnel_transform_preferrable(self) -> bool:
        n_eff = math.sqrt(float(self.height_px) * float(self.width_px))
        return self.fresnel_number < 1 / (2 * n_eff)


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

    @timer()
    def propagate_forward(self, wavefield: ComplexTensor) -> ComplexTensor:
        return pmath.fft2_precise(wavefield, norm=self.norm)

    @timer()
    def propagate_backward(self, wavefield: ComplexTensor) -> ComplexTensor:
        return pmath.ifft2_precise(wavefield, norm=self.norm)

class AngularSpectrumPropagator(WavefieldPropagator):
    def __init__(self, parameters: WavefieldPropagatorParameters) -> None:
        super().__init__()

        _transfer_function = self.get_transfer_function(parameters)
        
        # Separate registered buffer into real and imaginary parts to prevent it
        # from breaking in DataParallel.
        self._transfer_function_real = _transfer_function.real
        self._transfer_function_imag = _transfer_function.imag

    def update(self, parameters: WavefieldPropagatorParameters) -> None:
        _transfer_function = self.get_transfer_function(parameters)
        self._transfer_function_real = _transfer_function.real
        self._transfer_function_imag = _transfer_function.imag

    def get_transfer_function(self, parameters: WavefieldPropagatorParameters) -> ComplexTensor:
        ar = parameters.pixel_aspect_ratio

        i2piz = 2j * torch.pi * parameters.propagation_distance_wlu

        FY, FX = parameters.get_frequency_coordinates()
        FY, FX = FY.double(), FX.double()
        F2 = torch.square(FX) + torch.square(ar * FY)
        self.register_buffer('F2', F2)

        ratio = self.F2 / (parameters.pixel_width_wlu**2)
        tf = torch.exp(i2piz * torch.sqrt(1 - ratio))
        tf = torch.where(ratio < 1, tf, 1)
        tf = tf.to(get_default_complex_dtype())
        return tf

    def propagate_forward(self, wavefield: ComplexTensor) -> ComplexTensor:
        tf = self._transfer_function_real + 1j * self._transfer_function_imag
        return pmath.ifft2_precise(tf * pmath.fft2_precise(wavefield))

    def propagate_backward(self, wavefield: ComplexTensor) -> ComplexTensor:
        tf = self._transfer_function_real + 1j * self._transfer_function_imag
        return pmath.ifft2_precise(pmath.fft2_precise(wavefield) / tf)


class FresnelTransformPropagator(WavefieldPropagator):
    def __init__(self, parameters: WavefieldPropagatorParameters) -> None:
        super().__init__()

        _C0, _C1C2, _B = self.get_kernels(parameters)
        self._C0 = _C0
        self.register_buffer('_C1C2', _C1C2)
        self.register_buffer('_B', _B)
        
    def get_kernels(
        self, 
        parameters: WavefieldPropagatorParameters
    ) -> tuple[ComplexTensor, ComplexTensor, ComplexTensor]:
        ipi = 1j * torch.pi
        Fr = float(parameters.fresnel_number)
        ar = float(parameters.pixel_aspect_ratio)
        N = float(parameters.width_px)
        M = float(parameters.height_px)
        YY, XX = parameters.get_spatial_coordinates()
        YY, XX = YY.double(), XX.double()
        
        C0 = Fr / (1j * ar)
        C1 = cmath.exp(2j * cmath.pi * parameters.propagation_distance_wlu)
        C2 = torch.exp((torch.square(XX / N) + torch.square(ar * YY / M)) * ipi / Fr)
        C1C2 = C1 * C2
        B = torch.exp(ipi * Fr * (torch.square(XX) + torch.square(YY / ar)))
        
        C1C2 = C1C2.to(get_default_complex_dtype())
        B = B.to(get_default_complex_dtype())
        return C0, C1C2, B

    def propagate_forward(self, wavefield: ComplexTensor) -> ComplexTensor:
        A = self._C1C2 * self._C0
        return (A * pmath.fft2_precise(wavefield * self._B)).to(get_default_complex_dtype())

    def propagate_backward(self, wavefield: ComplexTensor) -> ComplexTensor:
        A = self._C1C2 / self._C0
        return (self._B * pmath.ifft2_precise(wavefield * A)).to(get_default_complex_dtype())


class FraunhoferPropagator(WavefieldPropagator):
    def __init__(self, parameters: WavefieldPropagatorParameters) -> None:
        super().__init__()
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
        return A * pmath.fft2_precise(wavefield)

    def propagate_backward(self, wavefield: ComplexTensor) -> ComplexTensor:
        A = self._C2 * self._C1 / self._C0
        return pmath.ifft2_precise(wavefield * A)
