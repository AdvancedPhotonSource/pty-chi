# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Optional, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor

from ptychi.api.options.base import DetectorResponseOptions
from ptychi.data_structures.base import ReconstructParameter
from ptychi.utils import to_tensor
from ptychi.metrics import validate_predicted_intensity

if TYPE_CHECKING:
    from ptychi.data_structures.probe import Probe


class DetectorResponse(ReconstructParameter):
    """Shared affine detector response, stored as ``(2, H, W)``: gain, offset.

    Gain is dimensionless and offset has measured-intensity units. The tensor
    uses the internal diffraction-data ordering, after any configured FFT shift.
    Gain is strictly positive; valid pixels are specified by valid_pixel_mask.
    Fixed maps with exactly unit gain and zero offset bypass detector corrections.
    Identity status is cached; use ``set_data`` to change fixed response values.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        data=None,
        options: Optional[DetectorResponseOptions] = None,
        valid_pixel_mask: Optional[Tensor] = None,
        fft_shift: bool = False,
    ):
        options = DetectorResponseOptions() if options is None else options
        if data is None:
            tensor = torch.stack((torch.ones(shape), torch.zeros(shape)))
        else:
            components = [to_tensor(component) for component in data]
            if len(components) != 2 or any(tuple(c.shape) != tuple(shape) for c in components):
                raise ValueError("`detector_response_map` must contain two detector-shaped maps.")
            for name, component in zip(("gain", "offset"), components):
                if torch.is_complex(component):
                    raise ValueError(f"`detector_response_map` {name} must be real-valued.")
                if not torch.isfinite(component).all():
                    raise ValueError(
                        f"`detector_response_map` {name} must contain only finite values "
                        "in the reconstruction dtype."
                    )
            if (components[0] <= 0).any():
                raise ValueError(
                    "`detector_response_map` gain must be strictly positive "
                    "in the reconstruction dtype."
                )
            tensor = torch.stack(components)
            if fft_shift:
                tensor = torch.fft.fftshift(tensor, dim=(-2, -1))
        super().__init__(data=tensor, name="detector_response", options=options, is_complex=False)
        self.tensor.requires_grad_(self.optimizable)
        if valid_pixel_mask is None:
            valid_pixel_mask = torch.ones(shape, device=tensor.device, dtype=torch.bool)
        self.register_buffer(
            "valid_pixel_mask", valid_pixel_mask.to(device=tensor.device, dtype=torch.bool)
        )
        self.register_buffer("least_squares_statistics", None, persistent=False)
        self.register_buffer("minimum_intensity", None, persistent=False)
        self._update_identity_status()

    @property
    def gain(self) -> Tensor:
        return self.tensor[0]

    @property
    def offset(self) -> Tensor:
        return self.tensor[1]

    @torch.no_grad()
    def normalize_gain(self, probe: "Probe") -> None:
        """Fix the gain/probe scale ambiguity without changing predictions.

        Divide every gain by its mean c over valid detector pixels and multiply
        all probe amplitudes by sqrt(c). Detector signal intensity scales by c,
        so gain * intensity + offset is invariant, including with crop and blur.
        Masked gains share this coordinate change but receive no fitted update.
        """
        if not self.options.optimize_gain or not self.valid_pixel_mask.any():
            return
        scale = self.gain[self.valid_pixel_mask].mean()
        self.gain.div_(scale)
        probe.set_data(probe.data * scale.sqrt())

    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
        self._update_identity_status()

    def _update_identity_status(self):
        self.is_identity = not self.optimizable and bool(
            (self.gain == 1).all() & (self.offset == 0).all()
        )

    def set_optimizable(self, optimizable):
        super().set_optimizable(optimizable)
        self._update_identity_status()

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._update_identity_status()

    def _apply(self, *args, **kwargs):
        result = super()._apply(*args, **kwargs)
        self._update_identity_status()
        return result

    def reset_least_squares_statistics(self) -> None:
        """Start a response fit with empty statistics on the response device.

        These temporary detector-grid buffers are excluded from checkpoints.
        Statistics use float64 to reduce cancellation in intensity variance;
        minimum intensity uses the response dtype for its feasibility bound.
        """
        self.least_squares_statistics = torch.zeros(
            (5, *self.gain.shape), dtype=torch.float64, device=self.tensor.device
        )
        self.minimum_intensity = torch.full_like(self.gain, torch.inf)

    @torch.no_grad()
    def accumulate_least_squares_statistics(
        self, intensity: Tensor, measured: Tensor, mask: Tensor
    ) -> None:
        """Accumulate count, sum(I), sum(d), sum(I**2), and sum(I*d) per pixel.

        I is the detector-grid intensity after crop/blur and before response;
        d is measured intensity. Use float64 statistics to reduce cancellation
        when computing variance across positions. Excluded samples contribute zero.
        Track the minimum uncorrected intensity over constrained observations
        for the offset bound. Call ``reset_least_squares_statistics`` first and
        keep the object and probe fixed throughout the accumulation pass.
        """
        self.minimum_intensity = torch.minimum(
            self.minimum_intensity, torch.where(mask, intensity, torch.inf).amin(0)
        )
        statistics = self.least_squares_statistics
        intensity = torch.where(mask, intensity, 0).to(statistics.dtype)
        measured = torch.where(mask, measured, 0).to(statistics.dtype)
        statistics[0] += mask.sum(0)
        statistics[1] += intensity.sum(0)
        statistics[2] += measured.sum(0)
        statistics[3] += intensity.square().sum(0)
        statistics[4] += (intensity * measured).sum(0)

    def calculate_least_squares_update(self) -> Tensor:
        """Return fitted response minus current response for intensity least squares.

        For offset-only refinement, b=mean(d)-gain*mean(I). When both maps
        are refined, gain=cov(I,d)/var(I), then b=mean(d)-gain*mean(I).
        With fixed offset, gain=sum(I*(d-b))/sum(I**2). Gain estimates whose
        denominator is unresolved at the reconstruction dtype's precision retain
        their current value. Pixels with no observations retain both values.

        This is an optimizer update direction, not a likelihood gradient. Store
        its negative in ``tensor.grad``: unit SGD reaches the unconstrained fit.
        ``step_optimizer`` imposes positivity; ``normalize_gain`` normalizes gain
        together with probe amplitude to preserve predictions.
        """
        statistics = self.least_squares_statistics
        count, sum_i, sum_d, sum_ii, sum_id = statistics
        observed = count > 0
        mean_i, mean_d = sum_i / count.clamp_min(1), sum_d / count.clamp_min(1)
        target = self.tensor.detach().to(statistics.dtype).clone()
        if self.options.optimize_gain:
            if self.options.optimize_offset:
                denominator = sum_ii - sum_i * mean_i
                numerator = sum_id - sum_i * mean_d
            else:
                denominator = sum_ii
                numerator = sum_id - self.offset * sum_i
            resolved = observed & (denominator > torch.finfo(self.tensor.dtype).eps * sum_ii)
            fitted_gain = numerator / torch.where(resolved, denominator, 1)
            target[0] = torch.where(resolved, fitted_gain, target[0])
        if self.options.optimize_offset:
            target[1] = torch.where(observed, mean_d - target[0] * mean_i, target[1])
        return target.to(self.tensor.dtype) - self.tensor.detach()

    @torch.no_grad()
    def apply_least_squares_update(self, probe: "Probe") -> None:
        """Synchronize and apply the accumulated fit, constraints, and normalization.

        Call once per completed accumulation pass on every rank, including
        ranks with no local observations. Sum sufficient statistics across ranks
        and take the minimum intensity before fitting the shared response.

        Automatic closures minimize half the squared distance to the fixed
        epoch-wise fit over enabled components and valid pixels. Its gradient
        is the negative fitted displacement, matching the supplied SGD update.
        The fit stays fixed during trial evaluations; gain and offset constraints
        are applied after the optimizer step.

        Bound each fitted offset by eps - gain * min(I) over constrained
        observations. Here eps is machine epsilon times max(1, gain * min(I)),
        in measured-intensity units. This preserves feasible negative offsets
        with a rounding margin. Infeasible gain-only fits restore the previous
        response and raise. Finally normalize gain and the supplied probe
        together. The bound holds for the current object and probe; subsequent
        likelihood evaluations validate their own predictions.
        """
        if dist.is_initialized():
            dist.all_reduce(self.least_squares_statistics)
            dist.all_reduce(self.minimum_intensity, op=dist.ReduceOp.MIN)
        observed = self.least_squares_statistics[0] > 0
        if not observed.any():
            return
        previous = self.data.clone()
        update = self.calculate_least_squares_update()
        self.set_grad(-update)
        closure = None
        if self.uses_optimizer_closure and self.optimizer_closure is None:
            target = self.tensor.detach() + update
            mask = self.get_optimization_mask()
            closure = self.make_optimizer_closure(
                lambda: 0.5 * (self.tensor - target)[mask].square().sum()
            )
        self.step_optimizer(closure=closure)
        self.optimizer.zero_grad()
        minimum_signal = self.gain * self.minimum_intensity
        margin = torch.finfo(self.tensor.dtype).eps * minimum_signal.clamp_min(1)
        lower_bound = margin - minimum_signal
        try:
            if self.options.optimize_offset:
                self.offset[observed] = torch.maximum(
                    self.offset[observed], lower_bound[observed]
                )
            elif (self.offset[observed] < lower_bound[observed]).any():
                raise ValueError(
                    "Detector response update is infeasible with the fixed offset. "
                    "Enable offset refinement or revise the detector response."
                )
            validate_predicted_intensity((minimum_signal + self.offset)[observed])
        except ValueError:
            self.set_data(previous)
            raise
        self.normalize_gain(probe)

    def get_optimization_mask(self) -> Tensor:
        """Return enabled gain/offset components at valid detector pixels."""
        mask = self.valid_pixel_mask.expand_as(self.tensor).clone()
        mask[0] &= self.options.optimize_gain
        mask[1] &= self.options.optimize_offset
        return mask

    def process_optimizer_gradients(self) -> None:
        if self.tensor.grad is not None:
            self.tensor.grad.masked_fill_(~self.get_optimization_mask(), 0)

    @torch.no_grad()
    def step_optimizer(self, *args, **kwargs):
        # Restore fixed components/pixels even for optimizers with momentum or decay.
        previous = self.tensor.clone()
        update_mask = self.get_optimization_mask()
        self.process_optimizer_gradients()
        super().step_optimizer(*args, **kwargs)
        self.tensor.copy_(torch.where(update_mask, self.tensor, previous))
        if self.options.optimize_gain and self.valid_pixel_mask.any():
            gain = self.gain[self.valid_pixel_mask].clamp_min(self.options.gain_floor)
            self.gain[self.valid_pixel_mask] = gain
        if self.options.optimize_offset and self.options.offset_nonnegative:
            self.offset[self.valid_pixel_mask] = self.offset[self.valid_pixel_mask].clamp_min(0)
        self._update_identity_status()
