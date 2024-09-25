import dataclasses
from typing import Literal, Optional

from .base import Config


@dataclasses.dataclass
class ReconstructorConfig(Config):
    
    # This should be superseded by CorrectionPlan in ParameterConfig when it is there. 
    num_epochs: int = 100
    """The number of epochs to run."""
    
    batch_size: int = 1
    """The number of data to process in each minibatch."""
    
    default_device: Literal['cpu', 'gpu'] = 'gpu'
    """The default device to use for computation."""
    
    gpu_indices: Optional[list[int]] = None
    """The GPU indices to use for computation. If None, use all available GPUs."""
    
    default_dtype: Literal['float32', 'float64'] = 'float32'
    """The default data type to use for computation."""
    
    random_seed: Optional[int] = None
    """The random seed to use for reproducibility. If None, no seed will be set."""
    
    metric_function: Optional[Literal['mse_sqrt', 'poisson', 'mse']] = None
    """
    The function that computes the tracked cost. Different from the `loss_function` 
    argument in some reconstructors, this function is only used for cost tracking
    and is not involved in the reconstruction math.
    """
    
    log_level: Literal['debug', 'info', 'warning', 'error', 'critical'] = 'info'
    """The log level to use for logging."""


@dataclasses.dataclass
class LSQMLReconstructorConfig(ReconstructorConfig):
    
    noise_model: Literal['gaussian', 'poisson'] = 'gaussian'
    
    noise_model_params: Optional[dict] = None
    """
    Noise model parameters. Depending on the choice of `noise_model`, the dictionary can contain the 
    following keys:
    
    Gaussian noise model:
        - 'gaussian_noise_std': The standard deviation of the gaussian noise.
    Poisson noise model:
        (None)
    """
    

@dataclasses.dataclass
class PIEReconstructorConfig(ReconstructorConfig):
    
    pass


@dataclasses.dataclass
class DMReconstructorConfig(ReconstructorConfig):
    
    pass
    
    
@dataclasses.dataclass
class AutodiffPtychographyReconstructorConfig(ReconstructorConfig):
    
    loss_function: Literal['mse_sqrt', 'poisson', 'mse'] = 'mse_sqrt'
    """
    The loss function.
    """