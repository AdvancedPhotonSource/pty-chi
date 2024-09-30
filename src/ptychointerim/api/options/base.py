from typing import Optional, Literal, Union, Sequence, Any
import dataclasses
from dataclasses import field
import json
import logging

from torch import Tensor
from numpy import ndarray

import ptychointerim.api.enums as enums


@dataclasses.dataclass
class Options:
        
    def uninherited_fields(self) -> dict:
        """
        Find fields that are not inherited from the parent class, and return
        them as a dictionary. 
        """
        parent_class = self.__class__.__bases__[0]
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
    
    optimizer: enums.Optimizers = enums.Optimizers.SGD
    """
    Name of the optimizer.
    """
    
    step_size: float = 1
    """
    Step size of the optimizer.
    """
    

@dataclasses.dataclass
class ObjectOptions(ParameterOptions):
    
    initial_guess: Union[ndarray, Tensor] = None
    """A (h, w) complex tensor of the object initial guess."""
    
    pixel_size_m: float = 1.0
    """The pixel size in meters."""
    
    
@dataclasses.dataclass
class ProbeOptions(ParameterOptions):
    """
    The probe configuration.
    
    The update behavior of eigenmodes (the second and following OPR modes) is currently
    different between LSQMLReconstructor and other reconstructors.
    
    LSQMLReconstructor:
        - The first OPR mode is always optimized as long as `optimizable == True`.
        - The eigenmodes are optimized only when
            - The probe has multiple OPR modes;
            - `optimizable == True`;
            - `OPRModeWeightsConfig` is given;
            - `OPRModeWeightsConfig` is optimizable.
    
    Other reconstructors:
        - The first OPR mode is always optimized as long as `optimizable == True`.
        - The eigenmodes are optimized when
            - The probe has multiple OPR modes;
            - `optimizable == True`;
            - `OPRModeWeightsConfig` is given.
    """
    
    initial_guess: Union[ndarray, Tensor] = None
    """A (n_opr_modes, n_modes, h, w) complex tensor of the probe initial guess."""
    
    def check(self):
        if not (self.initial_guess is not None and self.initial_guess.ndim == 4):
            raise ValueError('Probe initial_guess must be a (n_opr_modes, n_modes, h, w) tensor.')
    

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


@dataclasses.dataclass
class OPRModeWeightsOptions(ParameterOptions):
    
    initial_eigenmode_weights: Union[list[float], float] = 0.1
    """
    The initial weight(s) of the eigenmode(s). If it is a scaler, the weights of
    all eigenmodes (i.e., the second and following OPR modes) are set to this value.
    If it is a list, the length should be the number of eigenmodes.
    """
    
    optimize_intensity_variation: bool = False
    """
    Whether to optimize intensity variation, i.e., the weight of the first OPR mode.
    
    The behavior of this parameter is currently different between LSQMLReconstructor and
    other reconstructors.
    
    LSQMLReconstructor:
        - If `optimizable == True` but `optimize_intensity_variation == False`: only
            the weights of eigenmodes (2nd and following OPR modes) are optimized.
        - If `optimizable == True` and `optimize_intensity_variation == True`: both
            the weights of eigenmodes and the weight of the first OPR mode are optimized.
        - If `optimizable == False`: nothing is optimized.
    Other reconstructors:
        - This parameter is ignored.
        - If `optimizable == True`: both the weights of eigenmodes and the weight of 
            the first OPR mode are optimized. 
        - If `optimizable == False`: nothing is optimized.
    """
    
    def check(self):
        if self.optimizable:
            if not (self.optimize_intensity_variation or self.optimize_eigenmode_weights):
                raise ValueError('When OPRModeWeights is optimizable, at least 1 of '
                                 'optimize_intensity_variation and optimize_eigenmode_weights '
                                 'should be set to True.')

    
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
    
    default_dtype: Literal['float32', 'float64'] = 'float32'
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
class DataOptions(Options):
    
    pass
    
    
@dataclasses.dataclass
class PtychographyDataOptions(DataOptions):
    
    data: Union[ndarray, Tensor] = None
    """The data."""
    
    propagation_distance_m: float = 1.0
    """The propagation distance in meters."""
    
    wavelength_m: float = 1e-9
    """The wavelength in meters."""
    
    detector_pixel_size_m: float = 1e-8
    """The detector pixel size in meters."""
    
    valid_pixel_mask: Optional[Union[ndarray, Tensor]] = None
    """A 2D boolean mask where valid pixels are True."""


class TaskOptions(Options):
    pass


@dataclasses.dataclass
class PtychographyTaskOptions(TaskOptions):
    
    data_options: PtychographyDataOptions = field(default_factory=PtychographyDataOptions)
    
    reconstructor_options: ReconstructorOptions = field(default_factory=ReconstructorOptions)
    
    object_options: ObjectOptions = field(default_factory=ObjectOptions)
    
    probe_options: ProbeOptions = field(default_factory=ProbeOptions)
    
    probe_position_options: ProbePositionOptions = field(default_factory=ProbePositionOptions)
    
    opr_mode_weight_options: OPRModeWeightsOptions = field(default_factory=OPRModeWeightsOptions)

