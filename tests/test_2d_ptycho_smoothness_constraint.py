import pytest
import torch

from ptychi.api.options.base import ObjectOptions
from ptychi.data_structures.object import PlanarObject


@pytest.mark.parametrize("alpha", [0.0, 0.01, 0.05, 1.0 / 8])
def test_smoothness_impulse_response(alpha):
    options = ObjectOptions(optimizable=False, slice_spacings_m=[1e-6])
    options.smoothness_constraint.alpha = alpha
    data = torch.zeros((2, 7, 7), dtype=torch.complex64)
    data[0, 3, 3] = 1
    data[1, 2, 4] = -2 + 3j
    object_ = PlanarObject(data=data, options=options)

    object_.constrain_smoothness()

    # An interior impulse exposes the kernel weights and their spatial alignment.
    kernel = torch.tensor([
        [alpha, alpha, alpha],
        [alpha, 1 - 8 * alpha, alpha],
        [alpha, alpha, alpha],
    ])
    expected = torch.zeros_like(data)
    expected[0, 2:5, 2:5] = kernel
    expected[1, 1:4, 3:6] = (-2 + 3j) * kernel
    torch.testing.assert_close(object_.data, expected)


def test_smoothness_complex_values_with_replicated_boundaries():
    options = ObjectOptions(optimizable=False)
    options.smoothness_constraint.alpha = 1.0 / 8
    data = torch.tensor([[[1 + 4j, 2 - 2j], [3 + 0j, 4 + 6j]]], dtype=torch.complex64)
    object_ = PlanarObject(data=data, options=options)

    object_.constrain_smoothness()

    # At alpha=1/8, each pixel is the mean of its eight replicated neighbors.
    expected = torch.tensor(
        [[[17 + 14j, 19 + 14j], [21 + 18j, 23 + 18j]]], dtype=torch.complex64
    ) / 8
    torch.testing.assert_close(object_.data, expected)
