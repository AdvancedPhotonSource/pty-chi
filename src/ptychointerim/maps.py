from typing import Type

import torch

from ptychointerim.api.enums import (
    Dtypes,
    Optimizers,
    LossFunctions,
    Reconstructors,
    Devices,
    NoiseModels,
    ForwardModels,
)
from ptychointerim.metrics import MSELossOfSqrt
from ptychointerim.ptychotorch.reconstructors import (
    AutodiffPtychographyReconstructor,
    PIEReconstructor,
    EPIEReconstructor,
    RPIEReconstructor,
    LSQMLReconstructor,
)
import ptychointerim.forward_models as fm
from ptychointerim.ptychotorch.reconstructors.base import Reconstructor


def get_complex_dtype_by_enum(key: Dtypes) -> torch.dtype:
    return {Dtypes.FLOAT32: torch.complex64, Dtypes.FLOAT64: torch.complex128}[key]


def get_optimizer_by_enum(key: Optimizers) -> torch.optim.Optimizer:
    return {
        Optimizers.SGD: torch.optim.SGD,
        Optimizers.ADAM: torch.optim.Adam,
        Optimizers.RMSPROP: torch.optim.RMSprop,
        Optimizers.ADAGRAD: torch.optim.Adagrad,
        Optimizers.ADADELTA: torch.optim.Adadelta,
        Optimizers.LBFGS: torch.optim.LBFGS,
        Optimizers.ASGD: torch.optim.ASGD,
        Optimizers.SPARSE_ADAM: torch.optim.SparseAdam,
        Optimizers.ADAMAX: torch.optim.Adamax,
        Optimizers.RADAM: torch.optim.RAdam,
        Optimizers.ADAMW: torch.optim.AdamW,
    }[key]


def get_loss_function_by_enum(key: LossFunctions) -> torch.nn.Module:
    return {
        LossFunctions.MSE: torch.nn.MSELoss,
        LossFunctions.POISSON: torch.nn.PoissonNLLLoss,
        LossFunctions.MSE_SQRT: MSELossOfSqrt,
    }[key]


def get_forward_model_by_enum(key: ForwardModels) -> Type["fm.ForwardModel"]:
    return {
        ForwardModels.PTYCHOGRAPHY_2D: fm.Ptychography2DForwardModel,
        ForwardModels.MULTISLICE_PTYCHOGRAPHY: fm.MultislicePtychographyForwardModel,
    }[key]


def get_reconstructor_by_enum(key: Reconstructors) -> Type["Reconstructor"]:
    return {
        Reconstructors.base: Reconstructor,
        Reconstructors.LSQML: LSQMLReconstructor,
        Reconstructors.AD_PTYCHO: AutodiffPtychographyReconstructor,
        Reconstructors.PIE: PIEReconstructor,
        Reconstructors.EPIE: EPIEReconstructor,
        Reconstructors.RPIE: RPIEReconstructor,
    }[key]


def get_noise_model_by_enum(key: NoiseModels) -> str:
    return {NoiseModels.GAUSSIAN: "gaussian", NoiseModels.POISSON: "poisson"}[key]


def get_device_by_enum(key: Devices) -> str:
    return {Devices.CPU: "cpu", Devices.GPU: "cuda"}[key]


def get_dtype_by_enum(key: Dtypes) -> torch.dtype:
    return {Dtypes.FLOAT32: torch.float32, Dtypes.FLOAT64: torch.float64}[key]
