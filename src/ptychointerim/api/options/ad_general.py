from typing import Optional, Type, Union
import dataclasses
from dataclasses import field


import ptychointerim.api.options.base as base
import ptychointerim.api.options.task as task_options
import ptychointerim.api.enums as enums
import ptychointerim.forward_models as fm


@dataclasses.dataclass
class AutodiffReconstructorOptions(base.ReconstructorOptions):
    loss_function: enums.LossFunctions = enums.LossFunctions.MSE_SQRT
    """
    The loss function.
    """

    forward_model_class: Union[enums.ForwardModels, Type[fm.ForwardModel]] = enums.ForwardModels.base
    """
    The forward model class.
    """

    forward_model_params: Optional[dict] = None
    """
    The forward model parameters.
    """

    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.AD_GENERAL


@dataclasses.dataclass
class AutodiffOptions(task_options.PtychographyTaskOptions):
    reconstructor_options: AutodiffReconstructorOptions = field(
        default_factory=AutodiffReconstructorOptions
    )
