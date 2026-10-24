# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

"""Options for GM-rPIE and GM-MAGPIE with a single shared coherent probe."""

from dataclasses import field

import torch
from pydantic import Field as PydanticField
from pydantic.dataclasses import dataclass

from ptychi.api import enums
from ptychi.api.options import base, pie

__all__ = [
    "MAGPIECapabilityError",
    "MAGPIEInitializationError",
    "MAGPIEMultimodeError",
    "MAGPIEMultisliceError",
    "MAGPIEOPRError",
    "MAGPIEOPRModeWeightsOptions",
    "MAGPIEObjectOptions",
    "MAGPIEOptions",
    "MAGPIEProbeOptions",
    "MAGPIEProbePositionOptions",
    "MAGPIEReconstructorOptions",
]

_UNSET_TASK_DATA = object()


class MAGPIECapabilityError(ValueError):
    """A requested feature is unsupported by GM-rPIE and GM-MAGPIE."""


class MAGPIEInitializationError(ValueError):
    """An initial object or probe is invalid for GM-rPIE and GM-MAGPIE."""


class MAGPIEMultisliceError(MAGPIECapabilityError):
    """GM-rPIE and GM-MAGPIE require a single object slice."""


class MAGPIEMultimodeError(MAGPIECapabilityError):
    """GM-rPIE and GM-MAGPIE require a single coherent probe mode."""


class MAGPIEOPRError(MAGPIECapabilityError):
    """GM-rPIE and GM-MAGPIE require a shared probe without OPR."""


def _check_data_capabilities(object_data=None, probe_data=None, opr_mode_weights_data=None):
    """Check valid task array shapes before constructing model parameters."""
    if object_data is not None:
        shape = (
            object_data.shape
            if hasattr(object_data, "shape")
            else torch.as_tensor(object_data, device="cpu").shape
        )
        if len(shape) == 3 and shape[0] != 1:
            raise MAGPIEMultisliceError(
                f"GM-rPIE/GM-MAGPIE received {shape[0]} object slices but support "
                "a single slice; supply object_data with shape (1, height, width)."
            )
    if probe_data is not None:
        shape = (
            probe_data.shape
            if hasattr(probe_data, "shape")
            else torch.as_tensor(probe_data, device="cpu").shape
        )
        if len(shape) == 4:
            if shape[0] != 1:
                raise MAGPIEOPRError(
                    f"GM-rPIE/GM-MAGPIE received {shape[0]} OPR modes but do not support "
                    "OPR; supply probe_data with one shared mode, shape (1, 1, height, width)."
                )
            if shape[1] != 1:
                raise MAGPIEMultimodeError(
                    f"GM-rPIE/GM-MAGPIE received {shape[1]} incoherent probe modes but support "
                    "a single coherent mode; supply probe_data with shape (1, 1, height, width)."
                )
    if opr_mode_weights_data is not None:
        weights = (
            opr_mode_weights_data
            if isinstance(opr_mode_weights_data, torch.Tensor)
            else torch.as_tensor(opr_mode_weights_data, device="cpu")
        )
        if weights.ndim in (1, 2) and (
            weights.shape[-1] != 1 or not torch.all(weights == 1).item()
        ):
            raise MAGPIEOPRError(
                f"GM-rPIE/GM-MAGPIE received OPR weights with shape {tuple(weights.shape)} "
                "and multiple modes or non-unity values. OPR and scan-dependent intensity "
                "variation are unsupported; omit the weights or supply one mode fixed at unity."
            )


def _check_initial_factors(object_data=None, probe_data=None):
    """Require finite factors, a nonzero object everywhere, and a nonzero probe."""
    for name, data in (("object", object_data), ("probe", probe_data)):
        if data is None:
            continue
        if isinstance(data, (list, tuple)):
            # Python values must not inherit a lower process-wide precision
            # before task setup selects its configured dtype.
            values = torch.as_tensor(data, dtype=torch.complex128, device="cpu")
        else:
            values = data if isinstance(data, torch.Tensor) else torch.as_tensor(data, device="cpu")
        if not torch.isfinite(values).all().item():
            raise MAGPIEInitializationError(
                f"GM-rPIE/GM-MAGPIE received NaN or infinity in {name}_data. "
                f"Supply an initial {name} guess containing only finite values."
            )
        if not torch.any(values != 0).item():
            raise MAGPIEInitializationError(
                f"GM-rPIE/GM-MAGPIE received all-zero {name}_data. "
                f"Supply a finite initial {name} guess with nonzero amplitude; "
                "geometric-mean updates cannot recover an entirely zero factor."
            )
        if name == "object" and torch.any(values == 0).item():
            raise MAGPIEInitializationError(
                "GM-rPIE/GM-MAGPIE received zero entries in object_data. "
                "Supply a finite initial object guess with nonzero amplitude at every pixel "
                "(for example, an array of complex ones); exact-zero object pixels cannot "
                "be updated by the geometric mean."
            )


def _check_direct_updates(options: base.ParameterOptions, name: str):
    """Reject settings that would change or bypass the synthesized update."""
    if not options.optimizable:
        raise MAGPIECapabilityError(
            f"GM-rPIE/GM-MAGPIE require both object and probe to be updated; "
            f"set {name}_options.optimizable=True."
        )
    plan = options.optimization_plan
    if plan.start != 0 or plan.stop is not None or plan.stride != 1:
        raise MAGPIECapabilityError(
            f"GM-rPIE/GM-MAGPIE update {name} on every minibatch: "
            "optimization_plan must have start=0, stop=None, and stride=1."
        )
    if plan.step_size_scheduler_class is not None or plan.step_size_scheduler_options:
        raise MAGPIECapabilityError(
            f"GM-rPIE/GM-MAGPIE apply synthesized {name} updates directly and "
            "do not support object/probe step-size schedulers."
        )
    if options.step_size != 1 or options.optimizer != enums.Optimizers.SGD:
        raise MAGPIECapabilityError(
            f"GM-rPIE/GM-MAGPIE require {name}_options.step_size=1 and "
            "optimizer=SGD because synthesized updates are applied directly."
        )
    required = {
        "lr": 1,
        "momentum": 0,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
        "maximize": False,
        "differentiable": False,
    }
    for key, value in options.optimizer_params.items():
        if key in ("foreach", "fused"):
            continue
        if key not in required or value != required[key]:
            raise MAGPIECapabilityError(
                f"GM-rPIE/GM-MAGPIE require unit SGD without momentum or "
                f"weight decay; unsupported {name}_options.optimizer_params "
                f"entry {key!r}={value!r}."
            )


@dataclass
class MAGPIEReconstructorOptions(pie.PIEReconstructorOptions):
    """Settings for GM-rPIE (one level) or GM-MAGPIE (multiple levels).

    Both algorithms use random minibatches, one object slice, one coherent
    shared probe, and matching object/probe/detector sampling. Resampling,
    OPR, and detector blur are not supported. Object and probe updates are
    synthesized together from their pre-update state and applied directly.
    Native gradient position correction and post-epoch constraints remain available.
    """

    multigrid_levels: int = PydanticField(default=1, strict=True, ge=1)
    """Number of object grid levels, including the original resolution.

    Use 1 for GM-rPIE or a larger value for GM-MAGPIE. Each additional level
    halves each spatial dimension using 2-by-2 block averaging. The
    requested depth must fit the probe shape; excessive depths raise an
    error rather than being reduced silently.
    """

    forward_model_options: base.ForwardModelOptions = field(
        default_factory=lambda: base.ForwardModelOptions(pad_for_shift=1)
    )
    """Native forward-model options, with one-pixel subpixel-shift padding."""

    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.MAGPIE

    def check(self, options):
        super().check(options)
        if self.forward_model_options.pad_for_shift != 1:
            raise MAGPIECapabilityError(
                "GM-rPIE/GM-MAGPIE require forward_model_options.pad_for_shift=1 "
                "to preserve the one-pixel boundary padding of their local updates."
            )
        if self.batching_mode != enums.BatchingModes.RANDOM:
            raise MAGPIECapabilityError(
                "GM-rPIE/GM-MAGPIE support random minibatches only; "
                "set batching_mode=BatchingModes.RANDOM."
            )
        if self.default_dtype not in (enums.Dtypes.FLOAT32, enums.Dtypes.FLOAT64):
            raise MAGPIECapabilityError(
                "GM-rPIE/GM-MAGPIE support FLOAT32/complex64 and FLOAT64/complex128 precision."
            )


@dataclass
class MAGPIEObjectOptions(pie.PIEObjectOptions):
    """A single directly represented object slice, updated every minibatch.

    The inherited ``alpha`` is the nonnegative rPIE regularization factor
    used by the local object proposals. ``step_size`` must be 1, with plain
    SGD and no scheduler, since the synthesized update already incorporates
    its weights. Multislice propagation and slice-spacing refinement are
    unsupported.
    """

    def check(self, options):
        _check_direct_updates(self, "object")
        if self.patch_interpolation_method not in (
            enums.PatchInterpolationMethods.FOURIER,
            enums.PatchInterpolationMethods.BILINEAR,
        ):
            raise MAGPIECapabilityError(
                "GM-rPIE/GM-MAGPIE use native subpixel shifts; "
                "patch_interpolation_method must be FOURIER or BILINEAR."
            )
        if self.slice_spacing_options.optimizable or self.multislice_regularization.enabled:
            raise MAGPIEMultisliceError(
                "GM-rPIE/GM-MAGPIE support a single object slice; "
                "disable multislice regularization and slice-spacing refinement."
            )
        super().check(options)


@dataclass
class MAGPIEProbeOptions(pie.PIEProbeOptions):
    """One directly represented coherent probe shared by all scan positions.

    The inherited ``alpha`` is the nonnegative rPIE regularization factor
    used by local probe proposals. The probe is updated on every minibatch
    with unit SGD settings and no scheduler. Incoherent multimode probes,
    OPR, and synthesis-dictionary probe representations are unsupported.

    Local probe corrections use the native shift adjoint before being averaged
    with the intensity of each reverse-shifted local object patch. These weights
    are nonnegative without a floor or clipping. Zero total weight selects the
    ordinary mean; a singleton gives exactly the direct adjoint when the
    intermediates are finite. This coordinate-transport approximation differs
    from the reference synthesis and does not guarantee descent of its weighted
    residual objective for discrete padded shifts.
    """

    def check(self, options):
        _check_direct_updates(self, "probe")
        if self.experimental.sdl_probe_options.enabled:
            raise MAGPIECapabilityError(
                "GM-rPIE/GM-MAGPIE require a directly represented probe; "
                "disable experimental.sdl_probe_options.enabled."
            )
        super().check(options)


@dataclass
class MAGPIEProbePositionOptions(pie.PIEProbePositionOptions):
    """Native gradient position correction, including its optimization schedule.

    Cross-correlation position correction is not supported by this initial
    implementation.
    """

    def check(self, options):
        if (
            self.optimizable
            and self.correction_options.correction_type != enums.PositionCorrectionTypes.GRADIENT
        ):
            raise MAGPIECapabilityError(
                "GM-rPIE/GM-MAGPIE support gradient position correction only; "
                "set correction_options.correction_type=PositionCorrectionTypes.GRADIENT."
            )
        super().check(options)


@dataclass
class MAGPIEOPRModeWeightsOptions(pie.PIEOPRModeWeightsOptions):
    """OPR and scan-dependent intensity weights are unsupported.

    Keep weights fixed at unity and disable their optimization and smoothing.
    A single shared probe is used for every diffraction pattern.
    """

    def check(self, options):
        if self.optimizable or self.optimize_intensity_variation or self.smoothing.enabled:
            raise MAGPIEOPRError(
                "GM-rPIE/GM-MAGPIE require a shared probe without OPR or "
                "intensity variation; disable OPR weight optimization and smoothing."
            )
        super().check(options)


@dataclass
class MAGPIEOptions(pie.PIEOptions):
    """Native task options for GM-rPIE and GM-MAGPIE.

    Set ``reconstructor_options.multigrid_levels`` to 1 for GM-rPIE or a
    larger valid depth for GM-MAGPIE. Supply data and initial guesses as
    keyword arguments to :class:`ptychi.api.task.PtychographyTask`.

    Initial object and probe arrays must contain only finite values. Every
    object pixel must have nonzero amplitude because exact-zero object pixels
    cannot be updated by the geometric mean. The probe must contain at least
    one nonzero value; individual zero probe pixels are allowed. Invalid
    arrays raise :class:`MAGPIEInitializationError` during setup.
    """

    reconstructor_options: MAGPIEReconstructorOptions = field(
        default_factory=MAGPIEReconstructorOptions
    )
    object_options: MAGPIEObjectOptions = field(default_factory=MAGPIEObjectOptions)
    probe_options: MAGPIEProbeOptions = field(default_factory=MAGPIEProbeOptions)
    probe_position_options: MAGPIEProbePositionOptions = field(
        default_factory=MAGPIEProbePositionOptions
    )
    opr_mode_weight_options: MAGPIEOPRModeWeightsOptions = field(
        default_factory=MAGPIEOPRModeWeightsOptions
    )

    def check(
        self,
        *args,
        object_data=_UNSET_TASK_DATA,
        probe_data=_UNSET_TASK_DATA,
        opr_mode_weights_data=_UNSET_TASK_DATA,
        **kwargs,
    ):
        # Run before task construction so unsupported capabilities produce
        # their own errors, rather than missing-slice-spacing/weight errors.
        if object_data is _UNSET_TASK_DATA:
            object_data = self.object_options.initial_guess
        if probe_data is _UNSET_TASK_DATA:
            probe_data = self.probe_options.initial_guess
        if opr_mode_weights_data is _UNSET_TASK_DATA:
            opr_mode_weights_data = self.opr_mode_weight_options.initial_weights
        _check_data_capabilities(object_data, probe_data, opr_mode_weights_data)
        _check_initial_factors(object_data, probe_data)
        super().check(*args, **kwargs)
