from ptychointerim.api.enums import Devices, Dtypes, LossFunctions, NoiseModels, Optimizers, Reconstructors


import torch

from ptychointerim.metrics import MSELossOfSqrt
from ptychointerim.ptychotorch.reconstructors import AutodiffPtychographyReconstructor, EPIEReconstructor, LSQMLReconstructor
from ptychointerim.ptychotorch.reconstructors.base import Reconstructor


complex_dtype_dict = {
    Dtypes.FLOAT32: torch.complex64,
    Dtypes.FLOAT64: torch.complex128
}
optimizer_dict = {
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
}
loss_function_dict = {
    LossFunctions.MSE: torch.nn.MSELoss,
    LossFunctions.POISSON: torch.nn.PoissonNLLLoss,
    LossFunctions.MSE_SQRT: MSELossOfSqrt
}
reconstructor_dict = {
    Reconstructors.base: Reconstructor,
    Reconstructors.LSQML: LSQMLReconstructor,
    Reconstructors.AD_PTYCHO: AutodiffPtychographyReconstructor,
    Reconstructors.PIE: EPIEReconstructor
}
noise_model_dict = {
    NoiseModels.GAUSSIAN: 'gaussian',
    NoiseModels.POISSON: 'poisson'
}
device_dict = {
    Devices.CPU: 'cpu',
    Devices.GPU: 'cuda'
}
dtype_dict = {
    Dtypes.FLOAT32: torch.float32,
    Dtypes.FLOAT64: torch.float64
}