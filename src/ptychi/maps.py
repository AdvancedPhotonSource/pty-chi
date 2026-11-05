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


_OPTIMIZERS = {
    enums.Optimizers.SGD: (torch.optim.SGD, False),
    enums.Optimizers.ADAM: (torch.optim.Adam, False),
    enums.Optimizers.RMSPROP: (torch.optim.RMSprop, False),
    enums.Optimizers.ADAGRAD: (torch.optim.Adagrad, False),
    enums.Optimizers.ADADELTA: (torch.optim.Adadelta, False),
    enums.Optimizers.LBFGS: (torch.optim.LBFGS, True),
    enums.Optimizers.ASGD: (torch.optim.ASGD, False),
    enums.Optimizers.SPARSE_ADAM: (torch.optim.SparseAdam, False),
    enums.Optimizers.ADAMAX: (torch.optim.Adamax, False),
    enums.Optimizers.RADAM: (torch.optim.RAdam, False),
    enums.Optimizers.ADAMW: (torch.optim.AdamW, False),
}
_OPTIMIZER_CLOSURE_POLICY = dict(_OPTIMIZERS.values())


def get_optimizer_by_enum(key: enums.Optimizers) -> Type[torch.optim.Optimizer]:
    return _OPTIMIZERS[key][0]


def optimizer_uses_closure(optimizer: torch.optim.Optimizer) -> bool:
    """Select automatic closure behavior from the optimizer's contract.

    Required closure arguments always receive a closure. For optional arguments,
    a custom class's ``requires_closure`` declaration overrides detection. Known
    built-in classes use the registered policy; custom classes (including subclasses
    of built-ins) receive a closure by default when their step accepts one.
    """
    argument = inspect.signature(optimizer.step).parameters.get("closure")
    if argument is None:
        return False
    if argument.default is inspect.Parameter.empty:
        return True
    declared = getattr(type(optimizer), "requires_closure", None)
    if declared is not None:
        return declared
    return _OPTIMIZER_CLOSURE_POLICY.get(type(optimizer), True)


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
