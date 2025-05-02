# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Type

import torch

import ptychi.api.enums as enums
from ptychi.metrics import MSELossOfSqrt
import ptychi.reconstructors as reconstructors
import ptychi.forward_models as fm
from ptychi.reconstructors.base import Reconstructor
import ptychi.image_proc as ip
import ptychi.reconstructors.nn as pnn
from functools import partial


def get_complex_dtype_by_enum(key: enums.Dtypes) -> torch.dtype:
    return {enums.Dtypes.FLOAT32: torch.complex64, enums.Dtypes.FLOAT64: torch.complex128}[key]


def get_optimizer_by_enum(key: enums.Optimizers) -> torch.optim.Optimizer:
    return {
        enums.Optimizers.SGD: torch.optim.SGD,
        enums.Optimizers.ADAM: torch.optim.Adam,
        enums.Optimizers.RMSPROP: torch.optim.RMSprop,
        enums.Optimizers.ADAGRAD: torch.optim.Adagrad,
        enums.Optimizers.ADADELTA: torch.optim.Adadelta,
        enums.Optimizers.LBFGS: torch.optim.LBFGS,
        enums.Optimizers.ASGD: torch.optim.ASGD,
        enums.Optimizers.SPARSE_ADAM: torch.optim.SparseAdam,
        enums.Optimizers.ADAMAX: torch.optim.Adamax,
        enums.Optimizers.RADAM: torch.optim.RAdam,
        enums.Optimizers.ADAMW: torch.optim.AdamW,
    }[key]


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
        enums.Reconstructors.DM: reconstructors.DMReconstructor,
        enums.Reconstructors.BH: reconstructors.BHReconstructor,
    }[key]


def get_noise_model_by_enum(key: enums.NoiseModels) -> str:
    return {enums.NoiseModels.GAUSSIAN: "gaussian", enums.NoiseModels.POISSON: "poisson"}[key]


def get_device_by_enum(key: enums.Devices) -> str:
    return {enums.Devices.CPU: "cpu", enums.Devices.GPU: "cuda"}[key]


def get_dtype_by_enum(key: enums.Dtypes) -> torch.dtype:
    return {enums.Dtypes.FLOAT32: torch.float32, enums.Dtypes.FLOAT64: torch.float64}[key]


def get_patch_placer_function_by_name(
    key: enums.PatchInterpolationMethods,
) -> "ip.PlacePatchesProtocol":
    return {
        enums.PatchInterpolationMethods.FOURIER: ip.place_patches_fourier_shift,
        enums.PatchInterpolationMethods.BILINEAR: partial(
            ip.place_patches_bilinear_shift, round_positions=False
        ),
        enums.PatchInterpolationMethods.NEAREST: partial(
            ip.place_patches_bilinear_shift, round_positions=True
        ),
    }[key]


def get_patch_extractor_function_by_name(
    key: enums.PatchInterpolationMethods,
) -> "ip.ExtractPatchesProtocol":
    return {
        enums.PatchInterpolationMethods.FOURIER: ip.extract_patches_fourier_shift,
        enums.PatchInterpolationMethods.BILINEAR: partial(
            ip.extract_patches_bilinear_shift, round_positions=False
        ),
        enums.PatchInterpolationMethods.NEAREST: partial(
            ip.extract_patches_bilinear_shift, round_positions=True
        ),
    }[key]


def get_nn_model_by_enum(key: enums.DIPModels) -> Type["torch.nn.Module"]:
    return {
        enums.DIPModels.UNET: pnn.models.unet.UNet,
        enums.DIPModels.AUTOENCODER: pnn.models.autoencoder.Autoencoder,
    }[key]
