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
    AD_GENERAL = auto()
    AD_PTYCHO = auto()
    LSQML = auto()
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
    

class ForwardModels(StrEnum):
    base = auto()
    PTYCHOGRAPHY_2D = auto()
    MULTISLICE_PTYCHOGRAPHY = auto()


class Directions(StrEnum):
    X = auto()
    Y = auto()
    XY = auto()


class Devices(StrEnum):
    CPU = auto()
    GPU = auto()


class Dtypes(StrEnum):
    FLOAT32 = auto()
    FLOAT64 = auto()


class ImageGradientMethods(StrEnum):
    FOURIER_DIFFERENTIATION = auto()
    FOURIER_SHIFT = auto()
    NEAREST = auto()
    