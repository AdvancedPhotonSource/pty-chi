# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias
import cmath

import torch
from torch.fft import fftfreq

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
        """Signed *per-pixel* Fresnel number, ``dx^2 / (lambda z)``.

        Not the full-aperture Fresnel number ``W H / (lambda z)``; the two differ by a
        factor of ``width_px * height_px * (dy / dx)``. Width-only, because the propagator
        algebra carries the y-axis separately through :attr:`pixel_aspect_ratio`.

        Signed, because the propagator phase terms must conjugate when the propagation
        direction reverses. Take :func:`abs` wherever only the magnitude is meant.

        Raises
        ------
        ZeroDivisionError
            When the propagation distance is zero, where the conjugate-plane pixel size
            ``lambda |z| / (N dx)`` vanishes and the single-FFT propagators are undefined.
        """
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
        return YY, XX

    def get_frequency_coordinates(self) -> tuple[RealTensor, RealTensor]:
        fx = fftfreq(self.width_px)
        fy = fftfreq(self.height_px)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")
        return FY.detach(), FX.detach()
    
    def is_fresnel_transform_preferrable(self) -> bool:
        """Whether the Fresnel transform samples this geometry better than angular spectrum.

        Angular spectrum keeps the source pixel pitch, so it is correctly sampled only
        while the conjugate-plane pitch ``lambda |z| / (N dx)`` is no coarser than the
        source pitch. Tested per axis and resolved conservatively, so an anisotropic
        geometry is never aliased on its narrow axis. For square pixels this reduces to
        ``abs(fresnel_number) < 1 / N``.

        Uses the magnitude of the distance: the sampling requirement does not depend on
        which way the wavefield travels.
        """
        distance_wlu = abs(self.propagation_distance_wlu)
        return (
            self.pixel_width_wlu**2 / distance_wlu < 1 / self.width_px
            or self.pixel_height_wlu**2 / distance_wlu < 1 / self.height_px
        )


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

        # Stored as separate real and imaginary parts, and as plain attributes rather than
        # registered buffers, so that `update` can rebind them without an in-place write
        # (which would need `retain_graph` when the distance is optimizable).
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

        ratio = F2 / (parameters.pixel_width_wlu**2)
        # Clamp before the square root: `sqrt` of a negative gives NaN, and `torch.where`
        # discards the NaN *value* but not the NaN gradient it contributes through `exp`,
        # which poisons d(loss)/d(distance) when slice spacings are optimizable.
        tf = torch.exp(i2piz * torch.sqrt(torch.clamp(1 - ratio, min=0.0)))
        # Evanescent modes decay; blocking them is what makes the transfer function
        # physical, and it keeps `propagate_backward` a projector rather than a division
        # by a vanishing number.
        tf = torch.where(ratio < 1, tf, 0)
        tf = tf.to(utils.get_default_complex_dtype())
        return tf

    def propagate_forward(self, wavefield: ComplexTensor) -> ComplexTensor:
        tf = self._transfer_function_real + 1j * self._transfer_function_imag
        return pmath.ifft2_precise(tf * pmath.fft2_precise(wavefield))

    def propagate_backward(self, wavefield: ComplexTensor) -> ComplexTensor:
        tf = self._transfer_function_real + 1j * self._transfer_function_imag
        # Conjugate rather than divide: identical where |tf| == 1, but well defined on the
        # evanescent band where tf is 0.
        return pmath.ifft2_precise(torch.conj(tf) * pmath.fft2_precise(wavefield))


class FresnelTransformPropagator(WavefieldPropagator):
    """Direct Fresnel transform; centered in, centered out.

    The output plane has pixel pitch ``lambda |z| / (N dx)`` rather than the source pitch,
    unlike :class:`AngularSpectrumPropagator`, which preserves it. Callers that care about
    the physical scale of the output must account for that themselves; it is not reported.
    Output coordinates increase along both axes for either sign of the distance.
    """

    def __init__(self, parameters: WavefieldPropagatorParameters) -> None:
        super().__init__()

        _C0, _C1C2, _B = self.get_kernels(parameters)
        self._negative_distance = parameters.propagation_distance_wlu < 0
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
        
        C1C2 = C1C2.to(utils.get_default_complex_dtype())
        B = B.to(utils.get_default_complex_dtype())
        return C0, C1C2, B

    @timer()
    def propagate_forward(self, wavefield: ComplexTensor) -> ComplexTensor:
        # The kernels live on a centered grid (`XX = II - width // 2`), so the transform
        # must be centered too; a bare `fft2` would return the result in corner order.
        A = self._C1C2 * self._C0
        g = torch.fft.ifftshift(wavefield * self._B, dim=(-2, -1))
        # Negative distance reverses the Fourier exponent. norm="forward" makes the
        # inverse FFT unnormalized, preserving the diffraction amplitude prefactor.
        transformed = (
            pmath.ifft2_precise(g, norm="forward") if self._negative_distance
            else pmath.fft2_precise(g)
        )
        out = A * torch.fft.fftshift(transformed, dim=(-2, -1))
        return out.to(utils.get_default_complex_dtype())

    @timer()
    def propagate_backward(self, wavefield: ComplexTensor) -> ComplexTensor:
        # Divide out the same A the forward multiplied in, and conjugate B. Using
        # `C1C2 / C0` and `B` here would apply both phase screens twice in the forward
        # direction instead of undoing them.
        A = self._C1C2 * self._C0
        g = torch.fft.ifftshift(wavefield / A, dim=(-2, -1))
        transformed = (
            pmath.fft2_precise(g, norm="forward") if self._negative_distance
            else pmath.ifft2_precise(g)
        )
        out = torch.conj(self._B) * torch.fft.fftshift(transformed, dim=(-2, -1))
        return out.to(utils.get_default_complex_dtype())


class FraunhoferPropagator(WavefieldPropagator):
    """Far-field propagator: :class:`FresnelTransformPropagator` without the input
    quadratic phase ``exp(i pi Fr (X^2 + Y^2))``.

    Shares that class's pitch and centering conventions. The dropped term is evaluated at
    ``X_max = N / 2``, so the condition for it to be negligible is ``N^2 Fr << 1``, not
    ``Fr << 1``.
    """

    def __init__(self, parameters: WavefieldPropagatorParameters) -> None:
        super().__init__()
        ipi = 1j * torch.pi

        Fr = float(parameters.fresnel_number)
        ar = float(parameters.pixel_aspect_ratio)
        N = float(parameters.width_px)
        M = float(parameters.height_px)
        YY, XX = parameters.get_spatial_coordinates()
        # Double precision: the phase argument is of order 1 / Fr, which loses all
        # significance in float32 for a far-field geometry.
        YY, XX = YY.double(), XX.double()

        C2 = torch.exp((torch.square(XX / N) + torch.square(ar * YY / M)) * ipi / Fr)

        self._C0 = Fr / (1j * ar)
        self._negative_distance = parameters.propagation_distance_wlu < 0
        self._C1 = cmath.exp(2j * cmath.pi * parameters.propagation_distance_wlu)
        self.register_buffer("_C2", C2.to(utils.get_default_complex_dtype()))

    @timer()
    def propagate_forward(self, wavefield: ComplexTensor) -> ComplexTensor:
        A = self._C2 * self._C1 * self._C0
        g = torch.fft.ifftshift(wavefield, dim=(-2, -1))
        # Use the unnormalized inverse FFT for increasing output coordinates at z < 0.
        transformed = (
            pmath.ifft2_precise(g, norm="forward") if self._negative_distance
            else pmath.fft2_precise(g)
        )
        out = A * torch.fft.fftshift(transformed, dim=(-2, -1))
        return out.to(utils.get_default_complex_dtype())

    @timer()
    def propagate_backward(self, wavefield: ComplexTensor) -> ComplexTensor:
        A = self._C2 * self._C1 * self._C0
        g = torch.fft.ifftshift(wavefield / A, dim=(-2, -1))
        transformed = (
            pmath.fft2_precise(g, norm="forward") if self._negative_distance
            else pmath.ifft2_precise(g)
        )
        out = torch.fft.fftshift(transformed, dim=(-2, -1))
        return out.to(utils.get_default_complex_dtype())
