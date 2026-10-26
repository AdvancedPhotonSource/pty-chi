# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

"""Options for GM-rPIE and GM-MAGPIE with a single shared coherent probe."""

from dataclasses import field

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
    """Native forward-model options, with one-pixel subpixel-shift padding.

    Only ``ShiftOperators.PATCH`` sampling is supported. ``AUTO`` is accepted
    when it resolves to ``PATCH`` for the constructed geometry.
    Detector blur is disabled; a zero blur sigma is normalized to ``None``.
    """

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
        if self.forward_model_options.diffraction_pattern_blur_sigma not in (None, 0):
            raise MAGPIECapabilityError(
                "GM-rPIE/GM-MAGPIE does not support diffraction-pattern blur. "
                "Disable diffraction_pattern_blur_sigma or select an autodiff reconstructor."
            )
        if self.forward_model_options.diffraction_pattern_blur_sigma == 0:
            # The native forward model uses None to bypass the blur operation.
            self.forward_model_options.diffraction_pattern_blur_sigma = None


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
    r"""Native task options for GM-rPIE and GM-MAGPIE.

    Both algorithms use ``MAGPIEOptions``. Set
    ``reconstructor_options.multigrid_levels = 1`` for GM-rPIE or a larger
    valid depth for GM-MAGPIE. The initial implementation supports one object
    slice and one coherent shared probe, random minibatches, and matching
    object/probe/detector sampling. Object and probe must both be optimized
    on every minibatch, with unit SGD settings and no step-size schedulers.
    Native gradient position correction and post-epoch constraints remain available.

    Object sampling must resolve to ``ShiftOperators.PATCH``. The default
    ``AUTO`` selection is accepted when it resolves to ``PATCH``; near-field
    geometries with a probe approaching the object size can instead select
    ``FULL_OBJECT`` and are rejected during setup. To use patch sampling for
    such a geometry, explicitly set
    ``reconstructor_options.forward_model_options.shift_operator`` to
    ``enums.ShiftOperators.PATCH``. Full-object sampling is unsupported because
    the synthesis assumes integer object-patch placement and fractional probe shifts.

    .. code-block:: python

       from ptychi.api import enums
       from ptychi.api.options import MAGPIEOptions
       from ptychi.api.task import PtychographyTask

       options = MAGPIEOptions()
       options.reconstructor_options.default_device = enums.Devices.CPU
       options.reconstructor_options.multigrid_levels = 1  # GM-rPIE
       # Supply your measured intensities, initial guesses, and positions.
       task = PtychographyTask(
           options,
           diffraction_data=intensities,
           object_data=object_guess,
           probe_data=probe_guess,
           probe_position_x_px=positions_x,
           probe_position_y_px=positions_y,
       )
       task.run()

    Probe synthesis approximates the reference algorithm's weighted residual in
    common probe coordinates. Let :math:`S_k` be the native padded probe shift,
    :math:`d_k=\widetilde P_k-S_kP` the local geometric-mean probe correction, and
    :math:`\widetilde O_k` the local geometric-mean object patch. Define :math:`T_k`
    as the native coordinate shift with the opposite fractional displacement,
    using the same interpolation, replicate padding, and cropping. It differs
    from the actual padded adjoint :math:`S_k^*`.

    The shared probe correction minimizes a nonnegative weighted consensus:

    .. math::

       a_k &= S_k^*d_k, \\
       c_k &= |T_k\widetilde O_k|^2, \\
       \Delta P &= \underset{\delta}{\operatorname{argmin}}\;
           \frac12\sum_{k\in B}\left\|
           (T_k\widetilde O_k)\odot(\delta-a_k)\right\|_2^2
           = \frac{\sum_{k\in B}c_k\odot a_k}{\sum_{k\in B}c_k}, \\
       P_{\mathrm{new}} &= P+\Delta P.

    Multiplication and division are elementwise; the displayed ratio applies where
    the total confidence is positive. The confidence has no floor or clipping.
    Zero total confidence uses the ordinary mean of the adjoint
    corrections. With finite intermediates, a singleton minibatch gives exactly
    its direct-adjoint correction, including a singleton at the end of a larger
    batch.

    For ideal coordinate translations on a full or periodic domain,
    :math:`T_k=S_k^{-1}=S_k^*`, and translation preserves both the norm and pointwise
    products. Consequently,

    .. math::

       \left\|\widetilde O_k\odot(S_k\delta-d_k)\right\|_2^2
       =\left\|(T_k\widetilde O_k)\odot(\delta-S_k^*d_k)\right\|_2^2.

    Discrete interpolation does not generally preserve products, and padding makes
    the reverse coordinate shift different from the adjoint. The implemented
    consensus is therefore an approximation; it does not guarantee exact
    minimization or monotone decrease of the original weighted residual objective.

    Initial object and probe arrays must contain only finite values. Every
    object pixel must have nonzero amplitude because exact-zero object pixels
    cannot be updated by the geometric mean. The probe must contain at least
    one nonzero value; individual zero probe pixels are allowed. The constructed
    arrays are checked after dtype conversion, before reconstruction starts.
    Invalid values raise ``MAGPIEInitializationError`` during setup.

    :class:`ptychi.api.task.PtychographyTask` validates the complete options
    configuration before constructing parameters. When constructing parameters
    and ``MAGPIEReconstructor`` directly,
    call ``options.check()`` on a complete ``MAGPIEOptions`` configuration
    first, use its corresponding parameter and reconstructor options, and call
    ``reconstructor.build()`` before reconstruction. The reconstructor's
    ``check_inputs()`` validates constructed data and forward-model compatibility;
    it does not repeat the options-validation hierarchy.
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
