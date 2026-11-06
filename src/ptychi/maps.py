# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Type
import inspect

import torch

import ptychi.api.enums as enums
from ptychi.metrics import MSELossOfSqrt
import ptychi.reconstructors as reconstructors
import ptychi.forward_models as fm
from ptychi.reconstructors.base import Reconstructor
import ptychi.reconstructors.nn as pnn
from ptychi.device import AcceleratorModuleWrapper

def get_complex_dtype_by_enum(key: enums.Dtypes) -> torch.dtype:
    return {enums.Dtypes.FLOAT32: torch.complex64, enums.Dtypes.FLOAT64: torch.complex128}[key]


# Map each optimizer enum to (optimizer class, default OptimizerClosureModes value).
# NONE uses supplied gradients; LOSS evaluates only the objective;
# LOSS_AND_GRADIENTS reevaluates the objective and replaces the optimizer's gradients.
_OPTIMIZERS = {
    enums.Optimizers.SGD: (torch.optim.SGD, enums.OptimizerClosureModes.NONE),
    enums.Optimizers.ADAM: (torch.optim.Adam, enums.OptimizerClosureModes.NONE),
    enums.Optimizers.RMSPROP: (torch.optim.RMSprop, enums.OptimizerClosureModes.NONE),
    enums.Optimizers.ADAGRAD: (torch.optim.Adagrad, enums.OptimizerClosureModes.NONE),
    enums.Optimizers.ADADELTA: (torch.optim.Adadelta, enums.OptimizerClosureModes.NONE),
    enums.Optimizers.LBFGS: (torch.optim.LBFGS, enums.OptimizerClosureModes.LOSS_AND_GRADIENTS),
    enums.Optimizers.ASGD: (torch.optim.ASGD, enums.OptimizerClosureModes.NONE),
    enums.Optimizers.SPARSE_ADAM: (torch.optim.SparseAdam, enums.OptimizerClosureModes.NONE),
    enums.Optimizers.ADAMAX: (torch.optim.Adamax, enums.OptimizerClosureModes.NONE),
    enums.Optimizers.RADAM: (torch.optim.RAdam, enums.OptimizerClosureModes.NONE),
    enums.Optimizers.ADAMW: (torch.optim.AdamW, enums.OptimizerClosureModes.NONE),
}
_OPTIMIZER_CLOSURE_POLICY = dict(_OPTIMIZERS.values())


def get_optimizer_by_enum(key: enums.Optimizers) -> Type[torch.optim.Optimizer]:
    return _OPTIMIZERS[key][0]


def get_optimizer_closure_mode(optimizer: torch.optim.Optimizer) -> enums.OptimizerClosureModes:
    """Select automatic closure behavior from the optimizer's contract.

    A class's ``closure_mode`` selects loss-only or loss-and-gradient evaluation.
    It takes precedence over the legacy ``requires_closure`` boolean. Required
    closure arguments always receive a closure, falling back to loss and gradients
    for a NONE declaration. Optimizers without a closure argument use NONE.
    Otherwise, known built-ins use the registry and custom classes default to
    loss and gradients, including subclasses of built-ins.
    """
    argument = inspect.signature(optimizer.step).parameters.get("closure")
    if argument is None:
        return enums.OptimizerClosureModes.NONE
    mode = getattr(type(optimizer), "closure_mode", None)
    if mode is not None:
        mode = enums.OptimizerClosureModes(mode)
        if (mode == enums.OptimizerClosureModes.NONE
                and argument.default is inspect.Parameter.empty):
            return enums.OptimizerClosureModes.LOSS_AND_GRADIENTS
        return mode
    if argument.default is inspect.Parameter.empty:
        return enums.OptimizerClosureModes.LOSS_AND_GRADIENTS
    declared = getattr(type(optimizer), "requires_closure", None)
    if declared is not None:
        return (enums.OptimizerClosureModes.LOSS_AND_GRADIENTS
                if declared else enums.OptimizerClosureModes.NONE)
    return _OPTIMIZER_CLOSURE_POLICY.get(
        type(optimizer), enums.OptimizerClosureModes.LOSS_AND_GRADIENTS
    )


def get_loss_function_by_enum(key: enums.LossFunctions) -> torch.nn.Module:
    return {
        enums.LossFunctions.MSE: torch.nn.MSELoss,
        enums.LossFunctions.POISSON: torch.nn.PoissonNLLLoss,
        enums.LossFunctions.MSE_SQRT: MSELossOfSqrt,
    }[key]


def get_forward_model_by_enum(key: enums.ForwardModels) -> Type["fm.ForwardModel"]:
    return {
        enums.ForwardModels.PLANAR_PTYCHOGRAPHY: fm.PlanarPtychographyForwardModel,
    }[key]


def get_reconstructor_by_enum(key: enums.Reconstructors) -> Type["Reconstructor"]:
    return {
        enums.Reconstructors.base: Reconstructor,
        enums.Reconstructors.LSQML: reconstructors.LSQMLReconstructor,
        enums.Reconstructors.AD_PTYCHO: reconstructors.AutodiffPtychographyReconstructor,
        enums.Reconstructors.PIE: reconstructors.PIEReconstructor,
        enums.Reconstructors.EPIE: reconstructors.EPIEReconstructor,
        enums.Reconstructors.RPIE: reconstructors.RPIEReconstructor,
        enums.Reconstructors.MAGPIE: reconstructors.MAGPIEReconstructor,
        enums.Reconstructors.DM: reconstructors.DMReconstructor,
        enums.Reconstructors.RAAR: reconstructors.RAARReconstructor,
        enums.Reconstructors.BH: reconstructors.BHReconstructor,
    }[key]


def get_multiprocess_reconstructor_by_enum(key: enums.Reconstructors) -> Type["Reconstructor"]:
    d = {
        enums.Reconstructors.LSQML: reconstructors.MultiprocessLSQMLReconstructor,
        enums.Reconstructors.AD_PTYCHO: reconstructors.AutodiffPtychographyReconstructor,
    }
    reconstructor_class = d.get(key, None)
    if reconstructor_class is None:
        raise ValueError(
            f"{key} does not support multiprocessing. Either launch the task in "
            f"single-process mode or use a reconstructor that supports multiprocessing."
        )
    return reconstructor_class


def get_noise_model_by_enum(key: enums.NoiseModels) -> str:
    return {enums.NoiseModels.GAUSSIAN: "gaussian", enums.NoiseModels.POISSON: "poisson"}[key]


def get_device_by_enum(key: enums.Devices) -> str:
    return {enums.Devices.CPU: "cpu", enums.Devices.GPU: AcceleratorModuleWrapper.get_to_device_string()}[key]


def get_dtype_by_enum(key: enums.Dtypes) -> torch.dtype:
    return {
        enums.Dtypes.FLOAT16: torch.float16,
        enums.Dtypes.FLOAT32: torch.float32,
        enums.Dtypes.FLOAT64: torch.float64,
    }[key]


def get_nn_model_by_enum(key: enums.DIPModels) -> Type["torch.nn.Module"]:
    return {
        enums.DIPModels.UNET: pnn.models.unet.UNet,
        enums.DIPModels.AUTOENCODER: pnn.models.autoencoder.Autoencoder,
    }[key]
