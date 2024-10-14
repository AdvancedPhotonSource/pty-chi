from typing import Sequence
import logging

import torch

from ..device import Device
from ..maths import orthogonalize_gs
from .api import CorrectionPlan, DataProduct, DetectorData, IterativeAlgorithm
from .support import squared_modulus, ObjectPatchInterpolator

logger = logging.getLogger(__name__)


class RelaxedAveragedAlternatingReflections(IterativeAlgorithm):
    def __init__(self, device: Device, detector_data: DetectorData, product: DataProduct) -> None:
        self._good_pixels = torch.logical_not(detector_data.bad_pixels).to(device.torch_device)
        self._detected_amplitude = torch.sqrt(
            detector_data.diffraction_patterns.to(device.torch_device)
        )
        self._positions_px = product.positions_px.to(device.torch_device)
        self._probe = product.probe[0].to(device.torch_device)  # TODO support OPR modes
        self._object = product.object_[0].to(device.torch_device)  # TODO support multislice
        self._propagators = [propagator for propagator in product.propagators]

        self._iteration = 0
        self._data_error_norm = torch.sum(
            detector_data.diffraction_patterns.sum(dim=0)[self._good_pixels]
        )

        self._relaxation = 0.75
        self._probe_power = 0.0

    def set_relaxation(self, value: float) -> None:
        self._relaxation = value

    def get_relaxation(self) -> float:
        return self._relaxation

    def set_probe_power(self, value: float) -> None:
        self._probe_power = value

    def get_probe_power(self) -> float:
        return self._probe_power

    def iterate(self, plan: CorrectionPlan) -> Sequence[float]:
        number_of_positions = self._positions_px.shape[0]
        iteration_data_error = list()
        layer = 0  # TODO support multislice

        psi = self._probe.unsqueeze(0).repeat(number_of_positions, 1, 1, 1)

        if self._probe_power > 0.0:
            # calculate probe power correction
            propagated_probe = self._propagators[layer].propagate_forward(self._probe)
            propagated_probe_power = torch.sum(squared_modulus(propagated_probe))
            power_correction = torch.sqrt(self._probe_power / propagated_probe_power)

            # apply power correction
            self._probe = self._probe * power_correction
            self._object = self._object / power_correction

        for iteration in range(plan.number_of_iterations):
            data_error = 0.0

            for idx in range(number_of_positions):
                interpolator = ObjectPatchInterpolator(
                    self._object, self._positions_px[idx], self._probe.size()
                )
                object_patch = interpolator.get_patch()

                # exit wave is the outcome of the probe-object interation
                exit_wave = self._probe * object_patch

                foo = 2 * exit_wave - psi[idx]  # FIXME

                # propagate exit wave to the detector plane
                wavefield = self._propagators[layer].propagate_forward(foo)
                # propagated wavefield intensity is incoherent sum of mixed-state modes
                wavefield_intensity = torch.sum(squared_modulus(wavefield), dim=-3)

                # calculate data error
                detected_amplitude = self._detected_amplitude[idx]
                amplitude_diff_sq = torch.square(wavefield_intensity.sqrt() - detected_amplitude)
                data_error += torch.sum(amplitude_diff_sq[self._good_pixels]).item()

                # intensity correction
                corrected_wavefield = torch.where(
                    self._good_pixels,
                    detected_amplitude * torch.exp(1j * torch.angle(wavefield)),
                    wavefield,
                )

                # propagate corrected wavefield to object plane
                corrected_exit_wave = self._propagators[layer].propagate_backward(
                    corrected_wavefield
                )

                # calculate the relaxed reflection
                tempEW = torch.lerp(foo, corrected_exit_wave, 2 * self._relaxation)  # FIXME

                # update exit wave
                psi[idx] += (tempEW + psi[idx]) / 2

            if plan.probe_correction.is_enabled(iteration):
                probe_upper = torch.zeros_like(self._probe)
                probe_lower = torch.zeros_like(self._probe[0], dtype=torch.float32)

                for idx in range(number_of_positions):
                    interpolator = ObjectPatchInterpolator(
                        self._object, self._positions_px[idx], self._probe.size()
                    )
                    object_patch = interpolator.get_patch()

                    probe_upper += object_patch.conj() * psi[idx]
                    probe_lower += squared_modulus(object_patch)

                self._probe = probe_upper / (probe_lower + 1e-7)

            if plan.object_correction.is_enabled(iteration):
                object_upper = torch.zeros_like(self._object)
                object_lower = torch.zeros_like(self._object, dtype=torch.float32)

                for idx in range(number_of_positions):
                    interpolator_upper = ObjectPatchInterpolator(
                        object_upper, self._positions_px[idx], self._probe.size()
                    )
                    interpolator_upper.update_patch(
                        torch.sum(self._probe.conj() * psi[idx], dim=-3)
                    )

                    interpolator_lower = ObjectPatchInterpolator(
                        object_lower, self._positions_px[idx], self._probe.size()
                    )
                    interpolator_lower.update_patch(torch.sum(squared_modulus(self._probe), dim=-3))

                self._object = object_upper / (object_lower + 1e-7)

            if plan.probe_correction.is_enabled(iteration):
                self._probe = orthogonalize_gs(self._probe)

            iteration_data_error.append(data_error / self._data_error_norm)
            self._iteration += 1
            logger.info(f"iteration={self._iteration} error={data_error:.6e}")

        return iteration_data_error

    def get_product(self) -> DataProduct:
        return DataProduct(
            self._positions_px.cpu(),
            torch.unsqueeze(self._probe.cpu(), 0),
            torch.unsqueeze(self._object.cpu(), 0),
            [propagator.cpu() for propagator in self._propagators],
        )
