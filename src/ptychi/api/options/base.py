from typing import Optional, Union, TYPE_CHECKING, Sequence
import dataclasses
from dataclasses import field
import logging
from math import inf, ceil

from numpy import ndarray
from torch import Tensor
import torch
import numpy as np

import ptychi.api.enums as enums
from ptychi.api.options.plan import OptimizationPlan

if TYPE_CHECKING:
    import ptychi.api.options.task as task_options
    from ptychi.api.options.task import PtychographyTaskOptions


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Options:
    
    def __setattr__(self, name, value):
        # Check if the attribute already exists in the class fields.
        if name not in {f.name for f in dataclasses.fields(self)}:
            raise AttributeError(f"{name} is not a valid field in {self.__class__.__name__}.")
        # If it exists, allow setting the value.
        super().__setattr__(name, value)
        
    def check(self, *args, **kwargs) -> None:
        """Check if options values are valid.
        """
        return


@dataclasses.dataclass
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

    def get_non_data_fields(self) -> dict:
        d = self.__dict__.copy()
        return d


@dataclasses.dataclass
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


@dataclasses.dataclass
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


@dataclasses.dataclass
class ObjectL1NormConstraintOptions(FeatureOptions):
    """Settings for the L1 norm constraint."""

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    weight: float = 0
    """The weight of the L1 norm constraint. Disabled if equal or less than 0."""
    

@dataclasses.dataclass
class ObjectL2NormConstraintOptions(FeatureOptions):
    """Settings for the L2 norm constraint."""

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    weight: float = 0
    """The weight of the L2 norm constraint. Disabled if equal or less than 0."""


@dataclasses.dataclass
class ObjectSmoothnessConstraintOptions(FeatureOptions):
    """Settings for smoothing of the magnitude (but not the phase) of the object"""

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    alpha: float = 0
    """
    The relaxation smoothing constant. This value should be in the range  0 < alpha <= 1/8.

    Smoothing is done by constructing a 3x3 kernel of
    ```
        alpha, alpha,         alpha
        alpha, 1 - 8 * alpha, alpha
        alpha, alpha,         alpha
    ```
    and convolve it with the object magnitude. When `alpha == 1 / 8`, the smoothing power
    is maximal. The value of alpha should not be larger than 1 / 8.
    """


@dataclasses.dataclass
class ObjectTotalVariationOptions(FeatureOptions):
    """Settings for total variation constraint on the object."""

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    weight: float = 0
    """The weight of the total variation constraint. Disabled if equal or less than 0."""


@dataclasses.dataclass
class RemoveGridArtifactsOptions(FeatureOptions):
    """Settings for grid artifact removal in the object's phase, applied at the end of an epoch"""

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    period_x_m: float = 1e-7
    """The horizontal period of grid artifacts in meters."""

    period_y_m: float = 1e-7
    """The vertical period of grid artifacts in meters."""

    window_size: int = 5
    """The window size for grid artifact removal in pixels."""

    direction: enums.Directions = enums.Directions.XY
    """The direction of grid artifact removal."""
    

@dataclasses.dataclass
class RemoveObjectProbeAmbiguityOptions(FeatureOptions):
    """Settings for removing the object-probe ambiguity, where the object is scaled by its norm
    so that the mean transmission is kept around 1, and the probe is scaled accordingly.
    """

    enabled: bool = True

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=lambda: OptimizationPlan(stride=10))
    
    
@dataclasses.dataclass
class SliceSpacingOptions(ParameterOptions):
    
    optimizable: bool = False
    """Whether the slice spacings are optimizable."""
    
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
    
    

@dataclasses.dataclass
class ObjectOptions(ParameterOptions):
    initial_guess: Union[ndarray, Tensor] = None
    """A (h, w) complex tensor of the object initial guess."""

    slice_spacings_m: Optional[ndarray] = None
    """Slice spacings in meters. This should be provided if the object is multislice.
    
    If the slice spacings need to be optimized, set `slice_spacing_options.optimizable` to `True`.
    In that case, the slice spacings provided here are supposed to be the initial guess.
    """
    
    slice_spacing_options: SliceSpacingOptions = field(default_factory=SliceSpacingOptions)

    pixel_size_m: float = 1.0
    """The pixel size in meters. When pixel size is non-square, this should be the width (x)
    of the pixel size."""
    
    pixel_size_aspect_ratio: float = 1.0
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
    
    position_origin_coords: Optional[ndarray] = None
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
    
    def check(self, options: "PtychographyTaskOptions"):
        super().check(options)
        pos_y = options.probe_position_options.position_y_px
        pos_x = options.probe_position_options.position_x_px
        probe_shape = options.probe_options.initial_guess.shape[-2:]
        obj_shape = options.object_options.initial_guess.shape[-2:]
        min_size = [
            int(ceil((pos_y.max() - pos_y.min() + probe_shape[-2]).item())) + 2,
            int(ceil((pos_x.max() - pos_x.min() + probe_shape[-1]).item())) + 2,
        ]
        if any([min_size[i] > obj_shape[i] for i in range(2)]):
            logging.warning(
                f"An object tensor with a lateral size of at least {min_size} is "
                "required to avoid padding when extracting/placing patches, but the provided "
                f"object size is {list(options.object_options.initial_guess.shape[-2:])}."
            )
        if self.determine_position_origin_coords_by == enums.ObjectPosOriginCoordsMethods.SUPPORT:
            buffer_center = torch.tensor(
                [np.round(x / 2) + 0.5 for x in obj_shape]
            )
            if (
                pos_y.max() + buffer_center[0] + probe_shape[-2] // 2 > obj_shape[-2]
                or pos_y.min() + buffer_center[0] - probe_shape[-2] // 2 < 0
                or pos_x.max() + buffer_center[1] + probe_shape[-1] // 2 > obj_shape[-1]
                or pos_x.min() + buffer_center[1] - probe_shape[-1] // 2 < 0
            ):
                logging.warning(
                    "`object_options.determine_center_coords_by` is set to `SUPPORT`. This assumes "
                    "that the probe positions are approximately zero-centered, i.e., "
                    "`-pos_y.min() ~ pos_y.max()` and `-pos_x.min() ~ pos_x.max()`. "
                    "However, the given probe positions will cause the reconstructor to access pixels "
                    "out of the object support. Please provide probe positions that are approximately "
                    "zero-centered, or set `object_options.determine_center_coords_by` to `POSITIONS`."
                )
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


@dataclasses.dataclass
class ProbePowerConstraintOptions(FeatureOptions):
    """
    Settings for scaling the probe and object intensity.
    """

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    probe_power: float = 0.0
    """
    The target probe power. The probe and object intensity will be scaled such that 
    the power of the far-field probe is `probe_power`.
    """


@dataclasses.dataclass
class ProbeOrthogonalizeIncoherentModesOptions(FeatureOptions):
    """
    Settings for orthogonalizing incoherent probe modes.
    """

    enabled: bool = True

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    method: enums.OrthogonalizationMethods = enums.OrthogonalizationMethods.SVD
    """The method to use for incoherent_mode orthogonalization."""


@dataclasses.dataclass
class ProbeOrthogonalizeOPRModesOptions(FeatureOptions):
    """
    Settings for orthogonalizing OPR modes.
    """

    enabled: bool = True

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)


@dataclasses.dataclass
class ProbeSupportConstraintOptions(FeatureOptions):
    """
    Settings for probe shrinkwrapping, where small values are set to 0.
    """
        
    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    threshold: float = 0.005
    """
    The threshold for the probe support constraint. The value of a pixel (x, y) is set to 0
    if `p(x, y) < [max(blur(p)) * `threshold`](x, y)`.
    """


@dataclasses.dataclass
class ProbeCenterConstraintOptions(FeatureOptions):
    """
    Settings for constraining the probe's center of mass to the center of the probe array.
    """

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

@dataclasses.dataclass
class ProbeOptions(ParameterOptions):
    """
    The probe configuration.

    The first OPR mode of all incoherent modes are always optimized aslong as
    `optimizable` is `True`. In addition to thtat, eigenmodes (of the first
    incoherent mode) are optimized when:

    - The probe has multiple OPR modes;
    - `OPRModeWeightsConfig` is given.
    """

    initial_guess: Union[ndarray, Tensor] = None
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

    eigenmode_update_relaxation: float = 1.0
    """
    A separate step size for eigenmode update.
    """
    
    def check(self, options: "task_options.PtychographyTaskOptions"):
        super().check(options)
        if not (self.initial_guess is not None and self.initial_guess.ndim == 4):
            raise ValueError("Probe initial_guess must be a (n_opr_modes, n_modes, h, w) tensor.")
        if self.power_constraint.enabled and options.object_options.remove_object_probe_ambiguity.enabled:
            logger.warning(
                "`ObjectOptions.remove_object_probe_ambiguity` and `ProbeOptions.power_constraint` "
                "are both enabled, which may lead to unexpected results."
            )

    def get_non_data_fields(self) -> dict:
        d = super().get_non_data_fields()
        del d["initial_guess"]
        return d


@dataclasses.dataclass
class PositionCorrectionOptions(Options):
    """Options used for specifying the position correction function."""

    correction_type: enums.PositionCorrectionTypes = enums.PositionCorrectionTypes.GRADIENT
    """Type of algorithm used to calculate the position correction update."""
    
    differentiation_method: enums.ImageGradientMethods = enums.ImageGradientMethods.GAUSSIAN
    """The method for calculating the gradient of the object. Only used when `correction_type` 
    is `GRADIENT`. `"FOURIER_DIFFERENTIATION"` is usually the fastest, but it might be less
    stable when the object is noisy or non-smooth, under which circumstance `"GAUSSIAN"` or
    `"FOURIER_SHIFT"` may offer better stability. `"NEAREST"` is not recommended.
    """

    cross_correlation_scale: int = 20000
    """The upsampling factor of the cross-correlation in real space."""

    cross_correlation_real_space_width: float = 0.01
    """The width of the cross-correlation in real-space"""

    cross_correlation_probe_threshold: float = 0.1
    """The probe intensity threshold used to calculate the probe mask."""
    
    slice_for_correction: int = None
    """The object slice for which the position correction is calculated. If None, the middle slice
    is chosen.
    """
    
    update_magnitude_limit: Optional[float] = inf
    """The maximum allowed magnitude of position update in each axis. Updates larger than this value 
    are clipped. Set to None or inf to disable the constraint.
    """
    

@dataclasses.dataclass
class PositionAffineTransformConstraintOptions(FeatureOptions):
    """Settings for imposing an affine transformation constraint on the probe positions.
    """

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)
    
    degrees_of_freedom: Sequence[enums.AffineDegreesOfFreedom] = (
        enums.AffineDegreesOfFreedom.ROTATION,
        enums.AffineDegreesOfFreedom.SCALE,
        enums.AffineDegreesOfFreedom.SHEAR,
        enums.AffineDegreesOfFreedom.ASSYMETRY,
    )
    """The degrees of freedom to include in the affine transformation."""
    
    position_weight_update_interval: int = 10
    """The number of epochs between position weight updates.
    """
    
    apply_constraint: bool = True
    """Constraint is applied to probe positions only when this is `True`. When `False`,
    probe position weights and affine transformation matrix are still computed and
    stored in the `ProbePositions` object so that they can be logged and analyzed 
    externally, but the positions are not altered.
    """
    
    max_expected_error: float = 1.0
    """The maximum expected position error, given in pixels. Note that this is different
    from `update_magnitude_limit`, and is only used in the estimation of friction in
    affine transformation constraint.
    """
    
    def is_position_weight_update_enabled_on_this_epoch(self, current_epoch: int):
        if not self.enabled:
            return False
        if (current_epoch - self.optimization_plan.start) % self.position_weight_update_interval == 0:
            return True
        else:
            return False


@dataclasses.dataclass
class ProbePositionMagnitudeLimitOptions(FeatureOptions):
    """Settings for imposing a magnitude limit on the probe position update."""

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    limit: Optional[float] = 0


@dataclasses.dataclass
class ProbePositionOptions(ParameterOptions):
    optimizable: bool = False
    
    position_x_px: Union[ndarray, Tensor] = None
    """The x position in pixel."""

    position_y_px: Union[ndarray, Tensor] = None
    """The y position in pixel."""

    magnitude_limit: ProbePositionMagnitudeLimitOptions = dataclasses.field(
        default_factory=ProbePositionMagnitudeLimitOptions
    )

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
        if self.magnitude_limit.enabled or self.magnitude_limit.limit > 0:
            raise ValueError(
                "`probe_position_options.magnitude_limit` is depreciated "
                "and will be removed in the future. Please use "
                "`probe_position_options.correction_options.update_magnitude_limit` instead. "
                "To avoid this error, set `enabled` to `False` and `limit` to 0 in "
                "`probe_position_options.magnitude_limit`."
            )
        if self.correction_options.update_magnitude_limit == 0:
            raise ValueError(
                "`probe_position_options.correction_options.update_magnitude_limit` is "
                "set to 0. This will prevent the optimizer from performing any updates, but "
                "will at the same time produce unstability. If you want to disable "
                "position correction, set `optimizable` to `False`. To disable update "
                "magnitude limit, set `update_magnitude_limit` to None or inf."
            )
        if (
            self.correction_options.update_magnitude_limit is not None
            and self.correction_options.update_magnitude_limit < inf
            and options.reconstructor_options.get_reconstructor_type() == enums.Reconstructors.AD_PTYCHO
        ):
            raise NotImplementedError(
                "Update magnitude limit is not supported for AutodiffPtychography."
            )


@dataclasses.dataclass
class OPRModeWeightsSmoothingOptions(FeatureOptions):
    """Settings for smoothing OPR mode weights."""

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)
    
    method: Optional[enums.OPRWeightSmoothingMethods] = None
    """
    The method for smoothing OPR mode weights. 
    
    MEDIAN: applying a median filter to the weights of each mode. 
    
    POLYNOMIAL: fit the weights of each mode with a polynomial of selected degree.
    """

    polynomial_degree: int = 4
    """
    The degree of the polynomial used for smoothing OPR mode weights.
    """


@dataclasses.dataclass
class OPRModeWeightsOptions(ParameterOptions):
    initial_weights: Union[ndarray] = None
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

    smoothing: OPRModeWeightsSmoothingOptions = dataclasses.field(
        default_factory=OPRModeWeightsSmoothingOptions
    )

    update_relaxation: float = 1.0
    """
    A separate step size for eigenmode weight update.
    """

    def check(self, options: "task_options.PtychographyTaskOptions"):
        super().check(options)
        if self.optimizable:
            if not (self.optimize_intensity_variation or self.optimize_eigenmode_weights):
                raise ValueError(
                    "When OPRModeWeights is optimizable, at least 1 of "
                    "optimize_intensity_variation and optimize_eigenmode_weights "
                    "should be set to True."
                )
        n_opr_modes_in_probe = options.probe_options.initial_guess.shape[0]
        if n_opr_modes_in_probe > 1:
            if self.initial_weights is None:
                raise ValueError(
                    f"You have {n_opr_modes_in_probe} OPR modes in the probe initial guess, "
                    "but initial OPR weights are not provided."
                )
            elif self.initial_weights.shape[-1] != n_opr_modes_in_probe:
                raise ValueError(
                    f"You have {n_opr_modes_in_probe} OPR modes in the probe initial guess, "
                    f"but the number of modes in your provided OPR weights is {self.initial_weights.shape[-1]}."
                )
        else:
            if self.initial_weights is None:
                logging.info(
                    "Unspecified OPR weight initial guess will be automatically populated with 1s."
                )
            elif self.initial_weights.shape[-1] != n_opr_modes_in_probe:
                raise ValueError(
                    f"You have {n_opr_modes_in_probe} OPR modes in the probe initial guess, "
                    f"but the number of modes in your provided OPR weights is {self.initial_weights.shape[-1]}."
                )
        if self.initial_weights is not None and self.optimizable:
            logging.warning(
                "The default value of OPRModeWeightsOptions has been changed to False. "
                "You have provided initial OPR weights, but optimizable is set to False. "
                "Is this intended?"            
            )

    def get_non_data_fields(self) -> dict:
        d = super().get_non_data_fields()
        del d["initial_weights"]
        return d
    
    
@dataclasses.dataclass
class ForwardModelOptions(Options):
    low_memory_mode: bool = False
    """If True, forward propagation of ptychography will be done using less vectorized code.
    This reduces the speed, but also lowers memory usage.
    """
    
    pad_for_shift: Optional[int] = 0
    """If not None, the image is padded with border values by this amount before shifting."""


@dataclasses.dataclass
class ReconstructorOptions(Options):
    # This should be superseded by CorrectionPlan in ParameterConfig when it is there.
    num_epochs: int = 100
    """The number of epochs to run."""

    batch_size: int = 100
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

    compact_mode_update_clustering_stride: int = 1
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

    forward_model_options: ForwardModelOptions = dataclasses.field(
        default_factory=ForwardModelOptions
    )
    

    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.base


@dataclasses.dataclass
class TaskOptions(Options):
    pass
