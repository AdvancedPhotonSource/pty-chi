# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Optional, Union, TYPE_CHECKING, Sequence, get_origin, get_args
import dataclasses
from dataclasses import field, fields
import logging
import enum
import warnings

import numpy as np
import torch
from pydantic import ConfigDict, Field as PydanticField, field_validator, model_validator
from pydantic.dataclasses import dataclass

import ptychi.api.enums as enums
import ptychi.utils as utils

if TYPE_CHECKING:
    import ptychi.api.options.task as task_options
    from ptychi.api.options.task import PtychographyTaskOptions


logger = logging.getLogger(__name__)

SerializableArray = list | tuple
DataArray = torch.Tensor | np.ndarray | list | tuple

OPTIONS_CONFIG = ConfigDict(
    validate_assignment=True,
    extra="forbid",
    arbitrary_types_allowed=True,
)


def _get_validation_values(data):
    """Return mutable field values from Pydantic dataclass validator input."""
    if isinstance(data, dict):
        return data
    return getattr(data, "kwargs", None)


def _as_serializable_array(value):
    """Convert array-like option values to JSON-native containers."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


@dataclass(config=OPTIONS_CONFIG)
class Options:

    def check(self, *args, **kwargs) -> None:
        """Check if options values are valid.
        """
        return
    
    def resolve_type(self, ann_type) -> type:
        """Resolve annotation to underlying type (handles Optional, etc.)."""
        origin = get_origin(ann_type)
        if origin is Union:
            args = get_args(ann_type)
            # Drop NoneType from Optional[...]
            return next((arg for arg in args if arg is not type(None)), None)
        return ann_type
    
    def get_non_data_fields(self) -> dict:
        """Get fields that do not contain large arrays or tensors."""
        d = self.__dict__.copy()
        return d
    
    def get_dict(self) -> dict:
        """Get a dictionary representation of the options."""
        d = self.get_non_data_fields()
        for k, v in d.items():
            if isinstance(v, Options):
                d[k] = v.get_dict()
            else:
                d[k] = utils.jsonize(v)
        return d
    
    def load_from_dict(self, d: dict) -> "Options":
        """Load options from a dictionary."""
        for k, v in d.items():
            field_type = self.resolve_type(self.get_field_type(k))
            if isinstance(field_type, type) and issubclass(field_type, Options):
                self.__setattr__(k, self.resolve_type(self.get_field_type(k))().load_from_dict(v))
            elif isinstance(field_type, type) and issubclass(field_type, enum.StrEnum) and isinstance(v, str):
                self.__setattr__(k, field_type(v))
            else:
                self.__setattr__(k, v)
        return self
                
    def get_field_type(self, name: str) -> type:
        """Get the type of a field."""
        for f in fields(self):
            if f.name == name:
                return f.type
        raise ValueError(f"Field {name} not found in {self.__class__.__name__}.")


@dataclass
class OptimizationPlan(Options):
    """
    When a `ReconstructParameter` has `optimizable == True`, this class is used to specify
    the start, stop, and stride epochs of the optimization for that parameter. This class is
    also used by `FeatureOptions`.
    """
    start: int = PydanticField(default=0, ge=0)
    """
    The starting epoch.
    """

    stop: Optional[int] = PydanticField(default=None, ge=0)
    """
    The starting epoch. If None, optimization will run to the last epoch if the parameter
    is optimizable.
    """

    stride: int = PydanticField(default=1, ge=1)
    """
    The stride in epochs. Optimization will run every `stride` epochs.
    """

    step_size_scheduler_class: Optional[str] = None
    """
    Name of the step-size scheduler class in ``torch.optim.lr_scheduler``.
    If None, no scheduler is used.
    """

    step_size_scheduler_options: dict = dataclasses.field(default_factory=dict)
    """
    Keyword arguments passed to the step-size scheduler constructor other than
    ``optimizer``.
    """

    def is_enabled(self, epoch: int) -> bool:
        if self.start is not None and epoch < self.start:
            return False
        if self.stop is not None and epoch >= self.stop:
            return False
        if self.start is None:
            return True
        return (epoch - self.start) % self.stride == 0

    def is_in_optimization_interval(self, epoch: int) -> bool:
        if self.start is not None and epoch < self.start:
            return False
        if self.stop is not None and epoch >= self.stop:
            return False
        return True


@dataclass
class ParameterOptions(Options):
    optimizable: bool = True
    """
    Whether the parameter is optimizable.
    """

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)
    """
    Optimization plan for the parameter.
    """

    optimizer: enums.Optimizers = enums.Optimizers.SGD
    """
    Name of the optimizer.
    """

    step_size: float = 1
    """
    Step size of the optimizer. This will be the learning rate `lr` in
    `optimizer_params`.
    """

    optimizer_params: dict = dataclasses.field(default_factory=dict)
    """
    Settings for the optimizer of the parameter. For additional information on
    optimizer parameters, see: https://pytorch.org/docs/stable/optim.html
    """
    
    def check(self, options: "task_options.PtychographyTaskOptions"):
        return super().check(options)


@dataclass
class FeatureOptions(Options):
    """
    Abstract base class that is inherited by sub-feature dataclasses. This class is used to
    determining if/when a feature is used.
    """

    enabled: bool
    "Turns execution of the feature on and off."

    optimization_plan: OptimizationPlan
    "Schedules when the feature is executed."

    def is_enabled_on_this_epoch(self, current_epoch: int):
        if self.enabled and self.optimization_plan.is_enabled(current_epoch):
            return True
        else:
            return False


@dataclass
class ObjectMultisliceRegularizationOptions(FeatureOptions):
    """Settings for multislice regularization of the object."""

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    weight: float = 0
    """
    The weight for multislice regularization. Disabled if 0, or if `type != ObjectTypes.MULTISLICE`. 
    When enabled, multislice objects are regularized using cross-slice smoothing.
    """

    unwrap_phase: bool = True
    """Whether to unwrap the phase of the object during multislice regularization."""

    unwrap_image_grad_method: enums.ImageGradientMethods = (
        enums.ImageGradientMethods.FOURIER_DIFFERENTIATION
    )
    """
    The method for calculating the phase gradient during phase unwrapping.
    
        - FOURIER_SHIFT: Use Fourier shift to perform shift.
        - NEAREST: Use nearest neighbor to perform shift.
        - FOURIER_DIFFERENTIATION: Use Fourier differentiation.
    """

    unwrap_image_integration_method: enums.ImageIntegrationMethods = (
        enums.ImageIntegrationMethods.FOURIER
    )
    """
    The method for integrating the phase gradient during phase unwrapping.
    
        - FOURIER: Use Fourier integration as implemented in PtychoShelves.
        - DECONVOLUTION: Deconvolve a ramp filter.
        - DISCRETE: Use cumulative sum.
    """

@dataclass
class ObjectHardLimitsMagnitudePhase(FeatureOptions):
    """Settings for the hard constraint on sample mangitude and phase limits."""

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    abs_lim: Optional[SerializableArray] = None
    """Hard constraint for object magnitude: abs_lim[0] <= abs(object) <= abs_lim[1]."""
    
    phase_lim: Optional[SerializableArray] = None
    """Hard constraint for object phase: phase_lim[0] <= angle(object) <= phase_lim[1]."""

    @field_validator("abs_lim", "phase_lim", mode="before")
    @classmethod
    def _convert_limits_to_serializable_arrays(cls, value):
        return _as_serializable_array(value)


@dataclass
class ObjectL1NormConstraintOptions(FeatureOptions):
    """Settings for the L1 norm constraint."""

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    weight: float = 0
    """The weight of the L1 norm constraint. Disabled if equal or less than 0."""
    

@dataclass
class ObjectL2NormConstraintOptions(FeatureOptions):
    """Settings for the L2 norm constraint."""

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    weight: float = 0
    """The weight of the L2 norm constraint. Disabled if equal or less than 0."""


@dataclass
class ObjectSmoothnessConstraintOptions(FeatureOptions):
    """Settings for smoothing of the magnitude (but not the phase) of the object"""

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    alpha: float = PydanticField(default=0, ge=0, le=1.0 / 8)
    """
    The relaxation smoothing constant. This value should be in the range  0 < alpha <= 1/8.

    Smoothing is done by constructing a 3x3 kernel of

    ..  code-block::

        alpha, alpha,         alpha
        alpha, 1 - 8 * alpha, alpha
        alpha, alpha,         alpha

    and convolve it with the object magnitude. When `alpha == 1 / 8`, the smoothing power
    is maximal. The value of alpha should not be larger than 1 / 8.
    """


@dataclass
class ObjectTotalVariationOptions(FeatureOptions):
    """Settings for total variation constraint on the object."""

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    weight: float = 0
    """The weight of the total variation constraint. Disabled if equal or less than 0."""


@dataclass
class RemoveGridArtifactsOptions(FeatureOptions):
    """Settings for grid artifact removal in the object's phase, applied at the end of an epoch"""

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    period_x_m: float = PydanticField(default=1e-7, gt=0)
    """The horizontal period of grid artifacts in meters."""

    period_y_m: float = PydanticField(default=1e-7, gt=0)
    """The vertical period of grid artifacts in meters."""

    window_size: int = PydanticField(default=5, ge=1)
    """The window size for grid artifact removal in pixels."""

    direction: enums.Directions = enums.Directions.XY
    """The direction of grid artifact removal."""
    
    component: enums.MagPhaseComponents = enums.MagPhaseComponents.PHASE
    """The component of the object to remove grid artifacts from."""
    

@dataclass
class RemoveObjectProbeAmbiguityOptions(FeatureOptions):
    """Settings for removing the object-probe ambiguity, where the object is scaled by its norm
    so that the mean transmission is kept around 1, and the probe is scaled accordingly.
    """

    enabled: bool = True

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=lambda: OptimizationPlan(stride=10))
    
    
@dataclass
class SliceSpacingOptions(ParameterOptions):
    
    optimizable: bool = False
    """Whether the slice spacings are optimizable.
    
    Known issue: slice spacing optimization only works with AutodiffPtychography, and we
    have to use `loss.backward(retain_graph=True)` to make it work with AD. This might
    result in growing per-epoch walltime and memory usage. We are working on a better
    solution.
    """
    
    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)
    
    optimizer: enums.Optimizers = enums.Optimizers.SGD
    """The optimizer to use for optimizing the slice spacings."""
    
    step_size: float = 1e-10
    """The step size for the optimizer. As a recommendation, start with 1e-10 for SGD
    optimizer, and 1e-7 for ADAM optimizer.
    """
    
    def check(self, options: "task_options.PtychographyTaskOptions"):
        super().check(options)
        
        if (self.optimizable 
            and options.reconstructor_options.get_reconstructor_type() != enums.Reconstructors.AD_PTYCHO
        ):
            raise ValueError("Slice spacing optimization is only supported for AD Ptychography.")
    
    

@dataclass
class ObjectOptions(ParameterOptions):
    initial_guess: Optional[DataArray] = None
    """A (h, w) complex tensor of the object initial guess."""

    slice_spacings_m: Optional[SerializableArray] = None
    """Slice spacings in meters. This should be provided if the object is multislice.
    
    If the slice spacings need to be optimized, set `slice_spacing_options.optimizable` to `True`.
    In that case, the slice spacings provided here are supposed to be the initial guess.
    """
    
    slice_spacing_options: SliceSpacingOptions = field(default_factory=SliceSpacingOptions)

    pixel_size_m: float = PydanticField(default=1.0, gt=0)
    """The pixel size in meters. When pixel size is non-square, this should be the width (x)
    of the pixel size."""
    
    pixel_size_aspect_ratio: float = PydanticField(default=1.0, gt=0)
    """The aspect ratio of the pixel size, defined as width (x) / height (y).
    """

    l1_norm_constraint: ObjectL1NormConstraintOptions = field(
        default_factory=ObjectL1NormConstraintOptions
    )
    
    l2_norm_constraint: ObjectL2NormConstraintOptions = field(
        default_factory=ObjectL2NormConstraintOptions
    )

    smoothness_constraint: ObjectSmoothnessConstraintOptions = field(
        default_factory=ObjectSmoothnessConstraintOptions
    )

    total_variation: ObjectTotalVariationOptions = field(
        default_factory=ObjectTotalVariationOptions
    )
    
    hard_limits_magnitude_phase: ObjectHardLimitsMagnitudePhase = field(
        default_factory=ObjectHardLimitsMagnitudePhase
    )

    remove_grid_artifacts: RemoveGridArtifactsOptions = field(
        default_factory=RemoveGridArtifactsOptions
    )

    multislice_regularization: ObjectMultisliceRegularizationOptions = field(
        default_factory=ObjectMultisliceRegularizationOptions
    )

    patch_interpolation_method: enums.PatchInterpolationMethods = (
        enums.PatchInterpolationMethods.FOURIER
    )
    """
    Selects the interpolation method used for extracting and updating 
    patches of the object IF patch extraction/placement is done using 
    the object's methods `extract_patches_function` or 
    `place_patches_function`.
    """
    
    remove_object_probe_ambiguity: RemoveObjectProbeAmbiguityOptions = field(
        default_factory=RemoveObjectProbeAmbiguityOptions
    )
    
    build_preconditioner_with_all_modes: bool = False
    """If True, the probe illumination map used for the preconditioner is 
    built using the sum of intensities of all probe modes. This may help address
    some issues if some probe modes contain highly localized high-intensity anomalies,
    if the selected reconstructor uses preconditioner to regularize object updates.
    However, it might lead to slower convergence speed.
    """
    
    determine_position_origin_coords_by: enums.ObjectPosOriginCoordsMethods = enums.ObjectPosOriginCoordsMethods.SUPPORT
    """The method to determine the pixel coordinates of the object that corresponds 
    to the origin of the probe positions. 
    
    Probe positions are given as a list of coordinates that can be either positive
    or negative and have arbitrary offsets, while the object buffer is a discrete
    tensor where the pixel indices are 0-based and the origin is at the top left corner.
    The position origin coordinates are used to determine how the given probe positions
    are mapped to the object buffer: the origin of the probe positions (0, 0) is mapped
    to the pixel indices given by the position origin coordinates, and as such, the
    pixel indices of all probe positions are calculated as
    ```
    positions_pxind = positions + position_origin_coords
    ```
    
    - `POSITIONS`: the origin coordinates are determined as 
      `buffer_center - (positions.max() + positions.min()) / 2`. This puts the mid-point
      of the position range at the center of the buffer. It is more adaptive; however,
      in the case that one initializes a reconstruction with a previously reconstructed object
      and corrected probe positions, the center coordinates are not necessarily the same, 
      which can cause the positions to mismatch between both reconstructions. 
      
    - `SUPPORT`: the origin coordinates are determined as the center of the support of the 
      object. This is helpful to keep the center coordinates consistent between consecutive
      reconstructions, but the probe positions given should (at least approximately) zero-centered,
      i.e., `-postitions.min() ~ positions.max()` to prevent out-of-bound errors.
      
    - `SPECIFIED`: the origin coordinates are specified by the user. To make this setting effective,
      `position_origin_coords` should be specified.
    """
    
    position_origin_coords: Optional[SerializableArray] = None
    """The user-specified origin coordinates of the object. To make this setting effective,
    `determine_position_origin_coords_by` should be set to `SPECIFIED`. 
    
    Probe positions are given as a list of coordinates that can be either positive
    or negative and have arbitrary offsets, while the object buffer is a discrete
    tensor where the pixel indices are 0-based and the origin is at the top left corner.
    The position origin coordinates are used to determine how the given probe positions
    are mapped to the object buffer: the origin of the probe positions (0, 0) is mapped
    to the pixel indices given by the position origin coordinates, and as such, the
    pixel indices of all probe positions are calculated as
    ```
    positions_pxind = positions + position_origin_coords
    ```
    """

    def get_non_data_fields(self) -> dict:
        d = super().get_non_data_fields()
        del d["initial_guess"]
        return d

    @field_validator("slice_spacings_m", "position_origin_coords", mode="before")
    @classmethod
    def _convert_array_like_fields_to_serializable_arrays(cls, value):
        return _as_serializable_array(value)
    
    def check(self, options: "PtychographyTaskOptions"):
        super().check(options)
        if self.determine_position_origin_coords_by == enums.ObjectPosOriginCoordsMethods.SPECIFIED:
            if self.position_origin_coords is None:
                raise ValueError("`object_options.center_coords` should be specified when "
                                 "`object_options.determine_center_coords_by` is set to "
                                 "`SPECIFIED`.")
        if self.position_origin_coords is not None:
            if self.determine_position_origin_coords_by != enums.ObjectPosOriginCoordsMethods.SPECIFIED:
                logging.warning(
                    "`object_options.center_coords` will be disregarded when "
                    "`object_options.determine_center_coords_by` is not set to "
                    "`SPECIFIED`."
                )

        if self.optimizer == enums.Optimizers.LBFGS and "Autodiff" not in options.__class__.__name__:
            raise ValueError("LBFGS optimizer is currently only supported for Autodiff reconstructors.")

@dataclass
class ProbePowerConstraintOptions(FeatureOptions):
    """
    Settings for scaling the probe and object intensity.
    """

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    probe_power: float = PydanticField(default=0.0, ge=0)
    """
    The target probe power. The intensity of the probe and optionally the object will be 
    scaled such that the power of the probe itself is `probe_power`.
    """

    scale_object: bool = True
    """
    If True, scale the object inversely when the probe power is constrained.
    If False, only the probe is rescaled.
    """

@dataclass
class ProbeOrthogonalizeIncoherentModesOptions(FeatureOptions):
    """
    Settings for orthogonalizing incoherent probe modes.
    """

    enabled: bool = True

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    method: enums.OrthogonalizationMethods = enums.OrthogonalizationMethods.SVD
    """The method to use for incoherent_mode orthogonalization."""
    
    sort_by_occupancy: bool = False
    """If True, keep the probes sorted so that mode with highest occupancy is the 0th shared mode."""

@dataclass
class ProbeOrthogonalizeOPRModesOptions(FeatureOptions):
    """
    Settings for orthogonalizing OPR modes.
    """

    enabled: bool = True

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)


@dataclass
class ProbeSupportConstraintOptions(FeatureOptions):
    """
    Settings for probe support constraint. The constraint applies shrinkwrapping, 
    where small values below a threshold are set to 0. It can also optionally
    apply a probe support mask before shrinkwrapping.
    """
        
    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)
    
    fixed_probe_support: enums.ProbeSupportMethods = enums.ProbeSupportMethods.NONE
    """
    If not `NONE`, a fixed probe support mask is generated and applied before shrinkwrapping.
    The mask is applied to each incoherent probe mode. Choices are: `ELLIPSE`, `RECTANGLE`.
    """
    
    fixed_probe_support_params: Optional[SerializableArray] = None
    """
    If using the use_fixed_probe_support option, define the center, widths, and heights
    for the ellipse/rectangle, format is:
    [center (rows), center (columns), side length (rows), side length (columns)]
    """
    
    threshold: float = PydanticField(default=0.005, ge=0)
    """
    The threshold for shrinkwrapping. The value of a pixel (x, y) is set to 0
    if `p(x, y) < [max(blur(p)) * `threshold`](x, y)`.
    """

    @field_validator("fixed_probe_support_params", mode="before")
    @classmethod
    def _convert_fixed_probe_support_params_to_serializable_array(cls, value):
        return _as_serializable_array(value)


@dataclass
class ProbeCenterConstraintOptions(FeatureOptions):
    """
    Settings for constraining the probe's center of mass to the center of the probe array.
    """

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)
    
    use_total_intensity_for_com: bool = False
    """
    Whether to use the magnitude of the dominant shared probe 
    mode for computing the center of mass of the probe in order 
    to keep it centered, or to use the total probe intensity.
    """

    use_intensity_for_com: bool = False
    """
    Deprecated alias for `use_total_intensity_for_com`.
    """

    center_modes_individually: bool = False
    """
    If True, each mode is shifted individually based on their own center of mass.
    """

    @model_validator(mode="before")
    @classmethod
    def _validate_center_options(cls, data):
        values = _get_validation_values(data)
        if values is not None:
            use_intensity_for_com = values.get("use_intensity_for_com", False)
            use_total_intensity_for_com = values.get("use_total_intensity_for_com", False)
            center_modes_individually = values.get("center_modes_individually", False)
            if use_intensity_for_com:
                warnings.warn(
                    "`probe_options.center_constraint.use_intensity_for_com` is deprecated; "
                    "use `probe_options.center_constraint.use_total_intensity_for_com` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                values["use_total_intensity_for_com"] = True
                use_total_intensity_for_com = True
            if center_modes_individually and use_total_intensity_for_com:
                raise ValueError(
                    "`probe_options.center_constraint.use_total_intensity_for_com` must be False when "
                    "`probe_options.center_constraint.center_modes_individually` is True."
                )
        return data

    @model_validator(mode="after")
    def _validate_assignment_center_options(self):
        if self.center_modes_individually and self.use_total_intensity_for_com:
            raise ValueError(
                "`probe_options.center_constraint.use_total_intensity_for_com` must be False when "
                "`probe_options.center_constraint.center_modes_individually` is True."
            )
        return self

    def check(self, options: "task_options.PtychographyTaskOptions"):
        super().check(options)
        if self.use_intensity_for_com and not self.use_total_intensity_for_com:
            warnings.warn(
                "`probe_options.center_constraint.use_intensity_for_com` is deprecated; "
                "use `probe_options.center_constraint.use_total_intensity_for_com` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.use_total_intensity_for_com = True


@dataclass
class ProbeOptions(ParameterOptions):
    """
    The probe configuration.

    The first OPR mode of all incoherent modes are always optimized aslong as
    `optimizable` is `True`. In addition to thtat, eigenmodes (of the first
    incoherent mode) are optimized when:

    - The probe has multiple OPR modes;
    - `OPRModeWeightsConfig` is given.
    """

    initial_guess: Optional[DataArray] = None
    """A (n_opr_modes, n_modes, h, w) complex tensor of the probe initial guess."""

    power_constraint: ProbePowerConstraintOptions = field(
        default_factory=ProbePowerConstraintOptions
    )

    orthogonalize_incoherent_modes: ProbeOrthogonalizeIncoherentModesOptions = field(
        default_factory=ProbeOrthogonalizeIncoherentModesOptions
    )

    orthogonalize_opr_modes: ProbeOrthogonalizeOPRModesOptions = field(
        default_factory=ProbeOrthogonalizeOPRModesOptions
    )

    support_constraint: ProbeSupportConstraintOptions = field(
        default_factory=ProbeSupportConstraintOptions
    )

    center_constraint: ProbeCenterConstraintOptions = field(
        default_factory=ProbeCenterConstraintOptions
    )

    eigenmode_update_relaxation: float = PydanticField(default=1.0, ge=0, le=1)
    """
    A separate step size for eigenmode update.
    """

    def check(self, options: "task_options.PtychographyTaskOptions"):
        super().check(options)
        self.center_constraint.check(options)
        if self.power_constraint.enabled and options.object_options.remove_object_probe_ambiguity.enabled:
            logger.warning(
                "`ObjectOptions.remove_object_probe_ambiguity` and `ProbeOptions.power_constraint` "
                "are both enabled, which may lead to unexpected results."
            )
        if self.optimizer == enums.Optimizers.LBFGS and "Autodiff" not in options.__class__.__name__:
            raise ValueError("LBFGS optimizer is currently only supported for Autodiff reconstructors.")

    def get_non_data_fields(self) -> dict:
        d = super().get_non_data_fields()
        del d["initial_guess"]
        return d


@dataclass
class SynthesisDictLearnProbeOptions(Options):
    
    d_mat: Optional[SerializableArray] = None
    """The synthesis sparse dictionary matrix; contains the basis functions 
    that will be used to represent the probe via the sparse code weights."""
    
    d_mat_conj_transpose: Optional[SerializableArray] = None
    """Conjugate transpose of the synthesis sparse dictionary matrix."""
    
    d_mat_pinv: Optional[SerializableArray] = None
    """Moore-Penrose pseudoinverse of the synthesis sparse dictionary matrix."""
    
    probe_sparse_code: Optional[SerializableArray] = None
    """Sparse code weights vector."""
    
    probe_sparse_code_nnz: Optional[float] = None
    """Number of non-zeros we will keep when enforcing sparsity constraint on
    the sparse code weights vector probe_sparse_code."""
    
    enabled: bool = False

    @field_validator(
        "d_mat",
        "d_mat_conj_transpose",
        "d_mat_pinv",
        "probe_sparse_code",
        mode="before",
    )
    @classmethod
    def _convert_array_like_fields_to_serializable_arrays(cls, value):
        return _as_serializable_array(value)

    def get_non_data_fields(self) -> dict:
        d = super().get_non_data_fields()
        del d["d_mat"]
        del d["d_mat_conj_transpose"]
        del d["d_mat_pinv"]
        del d["probe_sparse_code"]
        return d

@dataclass
class PositionCorrectionOptions(Options):
    """Options used for specifying the position correction function."""

    correction_type: enums.PositionCorrectionTypes = enums.PositionCorrectionTypes.GRADIENT
    """Type of algorithm used to calculate the position correction update."""
    
    differentiation_method: enums.ImageGradientMethods = enums.ImageGradientMethods.FOURIER_DIFFERENTIATION
    """The method for calculating the gradient of the object. Only used when `correction_type` 
    is `GRADIENT`. `"FOURIER_DIFFERENTIATION"` is usually the fastest, but it might be less
    stable when the object is noisy or non-smooth, under which circumstance `"GAUSSIAN"` or
    `"FOURIER_SHIFT"` may offer better stability. `"NEAREST"` is not recommended.
    """

    cross_correlation_scale: int = PydanticField(default=20000, ge=1)
    """The upsampling factor of the cross-correlation in real space."""

    cross_correlation_real_space_width: float = PydanticField(default=0.01, gt=0)
    """The width of the cross-correlation in real-space"""

    cross_correlation_probe_threshold: float = PydanticField(default=0.1, ge=0)
    """The probe intensity threshold used to calculate the probe mask."""
    
    slice_for_correction: Optional[int] = PydanticField(default=None, ge=0)
    """The object slice for which the position correction is calculated. If None, the middle slice
    is chosen.
    """
    
    clip_update_magnitude_by_mad: bool = True
    """If True, the update magnitude is eventually clipped by 10 times the mean absolute deviation (MAD)
    of the updates. When `update_magnitude_limit` is set, the limit will be set to the smaller of them,
    i.e., `min(update_magnitude_limit, 10 * MAD)`.
    """
    
    update_magnitude_limit: Optional[float] = PydanticField(default=0.1, gt=0)
    """The maximum allowed magnitude of position update in each axis. Updates larger than this value 
    are clipped. Set to None or inf to disable the constraint. When `clip_update_magnitude_by_mad` is
    `True`, the actual limit will be set to the smaller of `update_magnitude_limit` and `10 * MAD`.
    """
    

@dataclass
class PositionAffineTransformConstraintOptions(FeatureOptions):
    """Settings for imposing an affine transformation constraint on the probe positions.
    """

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)
    
    degrees_of_freedom: Sequence[enums.AffineDegreesOfFreedom] = (
        enums.AffineDegreesOfFreedom.ROTATION,
        enums.AffineDegreesOfFreedom.SCALE,
        enums.AffineDegreesOfFreedom.SHEAR,
        enums.AffineDegreesOfFreedom.ASYMMETRY,
    )
    """The degrees of freedom to include in the affine transformation."""
    
    position_weight_update_interval: int = PydanticField(default=10, ge=1)
    """The number of epochs between position weight updates.
    """
    
    apply_constraint: bool = True
    """Constraint is applied to probe positions only when this is `True`. When `False`,
    probe position weights and affine transformation matrix are still computed and
    stored in the `ProbePositions` object so that they can be logged and analyzed 
    externally, but the positions are not altered.
    """
    
    max_expected_error: float = PydanticField(default=1.0, gt=0)
    """The maximum expected position error, given in pixels. Note that this is different
    from `update_magnitude_limit`, and is only used in the estimation of friction in
    affine transformation constraint.
    """
    
    override_update_flexibility: Optional[float] = PydanticField(default=None, ge=0, le=1)
    """If set, the update flexibility will be set to this value instead of being
    determined by the actual errors and max expected error. The value should be betweem
    0 and 1. If affine constraint is causing instability, setting this to a smaller value
    may help.
    """
    
    def is_position_weight_update_enabled_on_this_epoch(self, current_epoch: int):
        if not self.enabled:
            return False
        if (current_epoch - self.optimization_plan.start) % self.position_weight_update_interval == 0:
            return True
        else:
            return False

    @field_validator("override_update_flexibility")
    @classmethod
    def _warn_override_update_flexibility(cls, value: Optional[float]) -> Optional[float]:
        if value is not None:
            logging.warning(
                f"`override_update_flexibility` is set to {value}. "
                f"`max_expected_error` will be ignored."
            )
        return value
        
    def check(self, options: "task_options.PtychographyTaskOptions"):
        super().check(options)


@dataclass
class ProbePositionOptions(ParameterOptions):
    optimizable: bool = False
    
    step_size: float = 0.3
    """The step size for probe position update."""
    
    position_x_px: Optional[DataArray] = None
    """The x position in pixel."""

    position_y_px: Optional[DataArray] = None
    """The y position in pixel."""

    constrain_position_mean: bool = False
    """
    Whether to subtract the mean from positions after updating positions.
    """

    correction_options: PositionCorrectionOptions = dataclasses.field(
        default_factory=PositionCorrectionOptions
    )
    """
    Detailed options for position correction.
    """
    
    affine_transform_constraint: PositionAffineTransformConstraintOptions = dataclasses.field(
        default_factory=PositionAffineTransformConstraintOptions
    )
    """When enabled, an affine transformation from initial positions to current positions
    is fit, and positions deviating from the expected positions given by the affine
    transformation are penalized.
    """

    def get_non_data_fields(self) -> dict:
        d = super().get_non_data_fields()
        del d["position_x_px"]
        del d["position_y_px"]
        return d

    def check(self, options: "task_options.PtychographyTaskOptions"):
        super().check(options)
        self.affine_transform_constraint.check(options)
        
        if self.optimizer == enums.Optimizers.LBFGS and "Autodiff" not in options.__class__.__name__:
            raise ValueError("LBFGS optimizer is currently only supported for Autodiff reconstructors.")


@dataclass
class OPRModeWeightsSmoothingOptions(FeatureOptions):
    """Settings for smoothing OPR mode weights."""

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)
    
    method: enums.OPRWeightSmoothingMethods = enums.OPRWeightSmoothingMethods.MEDIAN
    """
    The method for smoothing OPR mode weights. 
    
    MEDIAN: applying a median filter to the weights of each mode. 
    
    POLYNOMIAL: fit the weights of each mode with a polynomial of selected degree.
    """

    polynomial_degree: int = PydanticField(default=4, ge=0)
    """
    The degree of the polynomial used for smoothing OPR mode weights.
    """


@dataclass
class OPRModeWeightsOptions(ParameterOptions):
    initial_weights: Optional[DataArray] = None
    """
    The initial weight(s) of the eigenmode(s). Acceptable values include the following:

    - a (n_scan_points, n_opr_modes) array of initial weights for every point.
    - a (n_opr_modes,) array that gives the weights of each OPR mode. These weights
      will be duplicated for every point.
    """
    
    optimizable: bool = False
    """
    The master switch of optimizability of OPR mode weights. This option must be set
    to True for either `optimize_eigenmode_weights` or `optimize_intensity_variation`
    to take effect.
    """

    optimize_eigenmode_weights: bool = True
    """
    Whether to optimize eigenmode weights, i.e., the weights of the second and
    following OPR modes.

    At least one of `optimize_eigenmode_weights` and `optimize_intensity_variation`
    should be set to `True` if `optimizable` is `True`.
    """

    optimize_intensity_variation: bool = False
    """
    Whether to optimize intensity variation, i.e., the weight of the first OPR mode.

    At least one of `optimize_eigenmode_weights` and `optimize_intensity_variation`
    should be set to `True` if `optimizable` is `True`.
    """

    primary_mode_weight_floor: Optional[float] = PydanticField(default=None, ge=0)
    """
    Lower bound for the weight of the primary (0th) OPR mode. If `None`, no
    lower bound is applied.
    """

    smoothing: OPRModeWeightsSmoothingOptions = dataclasses.field(
        default_factory=OPRModeWeightsSmoothingOptions
    )

    update_relaxation: float = PydanticField(default=1.0, ge=0, le=1)
    """
    A separate step size for eigenmode weight update.
    """

    @model_validator(mode="before")
    @classmethod
    def _validate_optimization_switches(cls, data):
        values = _get_validation_values(data)
        if values is not None:
            optimizable = values.get("optimizable", False)
            optimize_intensity_variation = values.get("optimize_intensity_variation", False)
            optimize_eigenmode_weights = values.get("optimize_eigenmode_weights", True)
            if optimizable and not (optimize_intensity_variation or optimize_eigenmode_weights):
                raise ValueError(
                    "When OPRModeWeights is optimizable, at least 1 of "
                    "optimize_intensity_variation and optimize_eigenmode_weights "
                    "should be set to True."
                )
        return data

    @model_validator(mode="after")
    def _validate_assignment_optimization_switches(self):
        if self.optimizable and not (
            self.optimize_intensity_variation or self.optimize_eigenmode_weights
        ):
            raise ValueError(
                "When OPRModeWeights is optimizable, at least 1 of "
                "optimize_intensity_variation and optimize_eigenmode_weights "
                "should be set to True."
            )
        return self

    def check(self, options: "task_options.PtychographyTaskOptions"):
        super().check(options)
        if self.optimizable:
            if not (self.optimize_intensity_variation or self.optimize_eigenmode_weights):
                raise ValueError(
                    "When OPRModeWeights is optimizable, at least 1 of "
                    "optimize_intensity_variation and optimize_eigenmode_weights "
                    "should be set to True."
                )
        if self.optimizer == enums.Optimizers.LBFGS and "Autodiff" not in options.__class__.__name__:
            raise ValueError("LBFGS optimizer is currently only supported for Autodiff reconstructors.")

    def get_non_data_fields(self) -> dict:
        d = super().get_non_data_fields()
        del d["initial_weights"]
        return d
    
    
@dataclass
class ForwardModelOptions(Options):
    low_memory_mode: bool = False
    """If True, forward propagation of ptychography will be done using less vectorized code.
    This reduces the speed, but also lowers memory usage.
    """
    
    pad_for_shift: Optional[int] = 0
    """If not None, the image is padded with border values by this amount before shifting."""
    
    diffraction_pattern_blur_sigma: Optional[float] = None
    """If not None, simulated diffraction patterns are blurred with a Gaussian kernel of
    this sigma. This is useful to mitigate the effect of the detector's point spread function.
    """


@dataclass
class ReconstructorOptions(Options):
    # This should be superseded by CorrectionPlan in ParameterConfig when it is there.
    num_epochs: int = PydanticField(default=100, ge=1)
    """The number of epochs to run."""

    batch_size: int = PydanticField(default=100, ge=1)
    """The number of data to process in each minibatch."""

    batching_mode: enums.BatchingModes = enums.BatchingModes.RANDOM
    """
    The batching mode to use. 
    
    - `enums.BatchingModes.RANDOM`: load a random set of data in each minibatch.
    - `enums.BatchingModes.COMPACT`: load a spatially close cluster of data in each minibatch.
      This is equivalent to the "compact" mode in PtychoShelves.
    - `enums.BatchingModes.UNIFORM`: load a random set of data in each minibatch, but the
      indices across batches are manipulated so that points in each batch are more uniformly
      spread out in the scan space. This is equivalent to the "sparse" mode in PtychoShelves.
    """

    compact_mode_update_clustering: bool = False
    """
    If True, clusters are updated after each probe position update when `batching_mode` is
    `COMPACT`.
    """

    compact_mode_update_clustering_stride: int = PydanticField(default=1, ge=1)
    """
    The number of epochs between updating clusters when `batching_mode` is `COMPACT` and
    `compact_mode_update_clustering` is `True`.
    """

    default_device: enums.Devices = enums.Devices.GPU
    """The default device to use for computation."""

    default_dtype: enums.Dtypes = enums.Dtypes.FLOAT32
    """The default data type to use for computation."""
    
    use_double_precision_for_fft: bool = False
    """If True, use double precision for critical FFT operations. When set to `True`,
    this option overrides `default_dtype`: even if `default_dtype` is set to `FLOAT32`,
    the FFTs will still be performed using double precision. If `False`,
    the FFTs will be performed using the precision specified by `default_dtype`.
    """

    allow_nondeterministic_algorithms: bool = True
    """If True, allow nondeterministic algorithms to be used. Non-deterministic algorithms
    include `scatter_add_` and `scatter_`. They can be faster, but may produce larger
    run-to-run variations.
    """

    random_seed: Optional[int] = None
    """The random seed to use for reproducibility. If None, no seed will be set."""

    displayed_loss_function: Optional[enums.LossFunctions] = enums.LossFunctions.MSE_SQRT
    """
    The function that computes the displayed cost. Different from the `loss_function`
    argument in some reconstructors, this function is only used for cost displaying
    and is not involved in the reconstruction math.
    """

    exclude_measured_pixels_below: Optional[float] = None
    """
    If not None, gradients corresponding to measured diffraction pixels whose intensity
    is less than or equal to this value are set to 0 in reconstructors that support it.
    """

    forward_model_options: ForwardModelOptions = dataclasses.field(
        default_factory=ForwardModelOptions
    )
    

    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.base


@dataclass
class TaskOptions(Options):
    pass
