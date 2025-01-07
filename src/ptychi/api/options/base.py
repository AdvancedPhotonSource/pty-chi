from abc import ABC
from typing import Optional, Union, Sequence
import dataclasses
from dataclasses import field
import logging

from numpy import ndarray
from torch import Tensor

import ptychi.api.enums as enums
from ptychi.api.options.plan import OptimizationPlan


@dataclasses.dataclass
class Options:
    
    def __setattr__(self, name, value):
        # Check if the attribute already exists in the class fields.
        if name not in {f.name for f in dataclasses.fields(self)}:
            raise AttributeError(f"{name} is not a valid field in {self.__class__.__name__}.")
        # If it exists, allow setting the value.
        super().__setattr__(name, value)
    
    def uninherited_fields(self) -> dict:
        """
        Find fields that are not inherited from the generic options parent
        class (typically the direct subclass of `ParameterOptions` or
        `Options`), and return them as a dictionary.
        """
        parent_classes = [
            ObjectOptions,
            ProbeOptions,
            ReconstructorOptions,
            ProbePositionOptions,
            OPRModeWeightsOptions,
        ]
        parent_class = [
            parent_class for parent_class in parent_classes if isinstance(self, parent_class)
        ][0]
        if parent_class == object:
            return self.__dict__
        parent_fields = [f.name for f in dataclasses.fields(parent_class)]
        d = {}
        for k, v in self.__dict__.items():
            if k not in parent_fields:
                d[k] = v
        return d


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

    def get_non_data_fields(self) -> dict:
        d = self.__dict__.copy()
        return d


@dataclasses.dataclass
class FeatureOptions(ABC):
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
        enums.ImageGradientMethods.FOURIER_SHIFT
    )
    """
    The method for calculating the phase gradient during phase unwrapping.
    
        - FOURIER_SHIFT: Use Fourier shift to perform shift.
        - NEAREST: Use nearest neighbor to perform shift.
        - FOURIER_DIFFERENTIATION: Use Fourier differentiation.
    """

    unwrap_image_integration_method: enums.ImageIntegrationMethods = (
        enums.ImageIntegrationMethods.DECONVOLUTION
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
class ObjectOptions(ParameterOptions):
    initial_guess: Union[ndarray, Tensor] = None
    """A (h, w) complex tensor of the object initial guess."""

    slice_spacings_m: Optional[ndarray] = None
    """Slice spacing in meters. This should be provided if the object is multislice."""

    pixel_size_m: float = 1.0
    """The pixel size in meters."""

    l1_norm_constraint: ObjectL1NormConstraintOptions = field(
        default_factory=ObjectL1NormConstraintOptions
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
    """The interpolation method used for extracting and updating patches of the object."""

    def get_non_data_fields(self) -> dict:
        d = super().get_non_data_fields()
        del d["initial_guess"]
        return d


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

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    method: enums.OrthogonalizationMethods = enums.OrthogonalizationMethods.GS
    """The method to use for incoherent_mode orthogonalization."""


@dataclasses.dataclass
class ProbeOrthogonalizeOPRModesOptions(FeatureOptions):
    """
    Settings for orthogonalizing OPR modes.
    """

    enabled: bool = False

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


    eigenmode_update_relaxation: float = 1.0
    """
    A separate step size for eigenmode update.
    """
    
    center_constraint: bool = False
    """
    Whether to constrain the probe's center of mass to the center of the probe array.
    """
    
    center_constraint_stride: int = 1
    """
    The number of epochs between probe center constraint updates.
    """

    def check(self):
        if not (self.initial_guess is not None and self.initial_guess.ndim == 4):
            raise ValueError("Probe initial_guess must be a (n_opr_modes, n_modes, h, w) tensor.")

    def get_non_data_fields(self) -> dict:
        d = super().get_non_data_fields()
        del d["initial_guess"]
        return d


@dataclasses.dataclass
class PositionCorrectionOptions:
    """Options used for specifying the position correction function."""

    correction_type: enums.PositionCorrectionTypes = enums.PositionCorrectionTypes.GRADIENT
    """Type of algorithm used to calculate the position correction update."""

    cross_correlation_scale: int = 20000
    """The upsampling factor of the cross-correlation in real space."""

    cross_correlation_real_space_width: float = 0.01
    """The width of the cross-correlation in real-space"""

    cross_correlation_probe_threshold: float = 0.1
    """The probe intensity threshold used to calculate the probe mask."""


@dataclasses.dataclass
class ProbePositionMagnitudeLimitOptions(FeatureOptions):
    """Settings for imposing a magnitude limit on the probe position update."""

    enabled: bool = False

    optimization_plan: OptimizationPlan = dataclasses.field(default_factory=OptimizationPlan)

    limit: Optional[float] = 0


@dataclasses.dataclass
class ProbePositionOptions(ParameterOptions):
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

    def get_non_data_fields(self) -> dict:
        d = super().get_non_data_fields()
        del d["position_x_px"]
        del d["position_y_px"]
        return d


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

    def check(self):
        if self.optimizable:
            if not (self.optimize_intensity_variation or self.optimize_eigenmode_weights):
                raise ValueError(
                    "When OPRModeWeights is optimizable, at least 1 of "
                    "optimize_intensity_variation and optimize_eigenmode_weights "
                    "should be set to True."
                )

    def get_non_data_fields(self) -> dict:
        d = super().get_non_data_fields()
        del d["initial_weights"]
        return d


@dataclasses.dataclass
class ReconstructorOptions(Options):
    # This should be superseded by CorrectionPlan in ParameterConfig when it is there.
    num_epochs: int = 100
    """The number of epochs to run."""

    batch_size: int = 1
    """The number of data to process in each minibatch."""

    batching_mode: enums.BatchingModes = enums.BatchingModes.RANDOM
    """
    The batching mode to use. 
    
    - `enums.BatchingModes.RANDOM`: load a random set of data in each minibatch.
    - `enums.BatchingModes.COMPACT`: load a spatially close cluster of data in each minibatch.
      This is equivalent to the compact mode in PtychoSheleves.
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

    random_seed: Optional[int] = None
    """The random seed to use for reproducibility. If None, no seed will be set."""

    displayed_loss_function: Optional[enums.LossFunctions] = enums.LossFunctions.MSE_SQRT
    """
    The function that computes the displayed cost. Different from the `loss_function`
    argument in some reconstructors, this function is only used for cost displaying
    and is not involved in the reconstruction math.
    """

    use_low_memory_forward_model: bool = False
    """
    If True, forward propagation of ptychography will be done using less vectorized code.
    This reduces the speed, but also lowers memory usage.
    """

    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.base


@dataclasses.dataclass
class TaskOptions(Options):
    pass
