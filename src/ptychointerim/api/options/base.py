from typing import Optional, Union, Sequence
import dataclasses
import logging

from numpy import ndarray
from torch import Tensor

import ptychointerim.api.enums as enums
from ptychointerim.api.options.plan import OptimizationPlan

@dataclasses.dataclass
class Options:

    def uninherited_fields(self) -> dict:
        """
        Find fields that are not inherited from the generic options parent
        class (typically the direct subclass of `ParameterOptions` or
        `Options`), and return them as a dictionary.
        """
        parent_classes = [ObjectOptions, ProbeOptions, ReconstructorOptions, ProbePositionOptions, OPRModeWeightsOptions]
        parent_class = [parent_class for parent_class in parent_classes if isinstance(self, parent_class)][0]
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
class ObjectOptions(ParameterOptions):

    initial_guess: Union[ndarray, Tensor] = None
    """A (h, w) complex tensor of the object initial guess."""

    type: enums.ObjectTypes = enums.ObjectTypes.TWO_D
    """Type of the object."""

    slice_spacings_m: Optional[ndarray] = None
    """Slice spacing in meters. Only required if `type == ObjectTypes.MULTISLICE`."""

    pixel_size_m: float = 1.0
    """The pixel size in meters."""

    l1_norm_constraint_weight: float = 0
    """The weight of the L1 norm constraint. Disabled if equal or less than 0."""

    l1_norm_constraint_stride: int = 1
    """The number of epochs between L1 norm constraint updates."""

    smoothness_constraint_alpha: float = 0
    """
    The relaxation smoothing constant. If greater than 0, the magnitude (but not phase)
    of the object will be smoothed every `smoothness_constraint_stride` epochs.

    Smoothing is done by constructing a 3x3 kernel of
    ```
        alpha, alpha,         alpha
        alpha, 1 - 8 * alpha, alpha
        alpha, alpha,         alpha
    ```
    and convolve it with the object magnitude. When `alpha == 1 / 8`, the smoothing power
    is maximal. The value of alpha should not be larger than 1 / 8.
    """

    smoothness_constraint_stride: int = 1
    """The number of epochs between smoothness constraint updates."""
    
    total_variation_weight: float = 0
    """The weight of the total variation constraint. Disabled if equal or less than 0."""

    total_variation_stride: int = 1
    """The number of epochs between total variation constraint updates."""
    
    def get_non_data_fields(self) -> dict:
        d = super().get_non_data_fields()
        del d["initial_guess"]
        return d


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

    probe_power: float = 0.0
    """
    The target probe power. If greater than 0, probe power constraint
    is run every `probe_power_constraint_stride` epochs, where it scales the probe
    and object intensity such that the power of the far-field probe is `probe_power`.
    """

    probe_power_constraint_stride: int = 1
    """The number of epochs between probe power constraint updates."""

    orthogonalize_incoherent_modes: bool = False
    """Whether to orthogonalize incoherent probe modes. If True, the incoherent probe
    modes are orthogonalized every `orthogonalize_incoherent_modes_stride` epochs.
    """

    orthogonalize_incoherent_modes_stride: int = 1
    """The number of epochs between orthogonalizing the incoherent probe modes."""

    orthogonalize_incoherent_modes_method: enums.OrthogonalizationMethods = enums.OrthogonalizationMethods.GS
    """The method to use for incoherent_mode orthogonalization."""

    orthogonalize_opr_modes: bool = False
    """Whether to orthogonalize OPR modes. If True, the OPR modes are orthogonalized
    every `orthogonalize_opr_modes_stride` epochs.
    """

    orthogonalize_opr_modes_stride: int = 1
    """The number of epochs between orthogonalizing the OPR modes."""

    def check(self):
        if not (self.initial_guess is not None and self.initial_guess.ndim == 4):
            raise ValueError('Probe initial_guess must be a (n_opr_modes, n_modes, h, w) tensor.')

    def get_non_data_fields(self) -> dict:
        d = super().get_non_data_fields()
        del d["initial_guess"]
        return d


@dataclasses.dataclass
class ProbePositionOptions(ParameterOptions):

    position_x_m: Union[ndarray, Tensor] = None
    """The x position in meters."""

    position_y_m: Union[ndarray, Tensor] = None
    """The y position in meters."""

    pixel_size_m: float = 1.0
    """The pixel size in meters."""

    update_magnitude_limit: Optional[float] = 0
    """Magnitude limit of the probe update. No limit is imposed if it is 0."""
    
    def get_non_data_fields(self) -> dict:
        d = super().get_non_data_fields()
        del d["position_x_m"]
        del d["position_y_m"]
        return d

    @dataclasses.dataclass
    class CorrectionOptions:
        """Options used for specifying the position correction function."""

        correction_type: enums.PositionCorrectionTypes = (
            enums.PositionCorrectionTypes.GRADIENT
        )
        """Type of algorithm used to calculate the position correction update."""

        @dataclasses.dataclass
        class CrossCorrelationOptions:
            scale: int = 20000
            """The upsampling factor of the cross-correlation in real space."""

            real_space_width: float = 0.01
            """The width of the cross-correlation in real-space"""

            probe_threshold: float = 0.1
            """The probe intensity threshold used to calculate the probe mask."""

        @dataclasses.dataclass
        class GradientOptions:
            pass

        cross_correlation_options: CrossCorrelationOptions = dataclasses.field(
            default_factory=CrossCorrelationOptions
        )
        """Options used with cross correlation position correction."""

        gradient_options: GradientOptions = dataclasses.field(
            default_factory=GradientOptions
        )
        """Options used with gradient based position correction."""

    correction_options: CorrectionOptions = dataclasses.field(
        default_factory=CorrectionOptions
    )


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

    def check(self):
        if self.optimizable:
            if not (self.optimize_intensity_variation or self.optimize_eigenmode_weights):
                raise ValueError('When OPRModeWeights is optimizable, at least 1 of '
                                 'optimize_intensity_variation and optimize_eigenmode_weights '
                                 'should be set to True.')
                
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

    default_device: enums.Devices = enums.Devices.GPU
    """The default device to use for computation."""

    gpu_indices: Sequence[int] = ()
    """The GPU indices to use for computation. If empty, use all available GPUs."""

    default_dtype: enums.Dtypes = enums.Dtypes.FLOAT32
    """The default data type to use for computation."""

    random_seed: Optional[int] = None
    """The random seed to use for reproducibility. If None, no seed will be set."""

    metric_function: Optional[enums.LossFunctions] = None
    """
    The function that computes the tracked cost. Different from the `loss_function`
    argument in some reconstructors, this function is only used for cost tracking
    and is not involved in the reconstruction math.
    """

    log_level: int | str = logging.INFO
    """The log level to use for logging."""

    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.base


@dataclasses.dataclass
class TaskOptions(Options):

    pass
