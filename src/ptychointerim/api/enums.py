from enum import StrEnum, auto


class Optimizers(StrEnum):
    SGD = auto()
    ADAM = auto()
    RMSPROP = auto()
    ADAGRAD = auto()
    ADADELTA = auto()
    LBFGS = auto()
    ASGD = auto()
    SPARSE_ADAM = auto()
    ADAMAX = auto()
    RADAM = auto()
    ADAMW = auto()


class LossFunctions(StrEnum):
    MSE = auto()
    POISSON = auto()
    MSE_SQRT = auto()


class Reconstructors(StrEnum):
    base = auto()
    LSQML = auto()
    AD_PTYCHO = auto()
    PIE = auto()
    EPIE = auto()
    RPIE = auto()


class ObjectTypes(StrEnum):
    TWO_D = auto()
    MULTISLICE = auto()


class PositionCorrectionTypes(StrEnum):
    GRADIENT = auto()
    CROSS_CORRELATION = auto()


class NoiseModels(StrEnum):
    GAUSSIAN = auto()
    POISSON = auto()


class OrthogonalizationMethods(StrEnum):
    GS = auto()
    SVD = auto()


class Devices(StrEnum):
    CPU = auto()
    GPU = auto()


class Dtypes(StrEnum):
    FLOAT32 = auto()
    FLOAT64 = auto()
