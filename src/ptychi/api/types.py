# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import TypeAlias, Any
import numpy
import torch


Numeric: TypeAlias = int | float | complex

BooleanArray: TypeAlias = numpy.typing.NDArray[numpy.bool_]
IntegerArray: TypeAlias = numpy.typing.NDArray[numpy.integer[Any]]
RealArray: TypeAlias = numpy.typing.NDArray[numpy.floating[Any]]
ComplexArray: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]

BooleanTensor: TypeAlias = torch.Tensor
ComplexTensor: TypeAlias = torch.Tensor
RealTensor: TypeAlias = torch.Tensor
