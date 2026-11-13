"""Position update directions in internal (y, x) pixel coordinates."""

import pytest
import torch

import ptychi.api as api
from ptychi.position_correction import PositionCorrection

pytestmark = pytest.mark.usefixtures("numerical_cpu")


def test_gradient_position_update_complex_chain_rule_and_axis_order():
    options = api.LSQMLOptions().probe_position_options.correction_options
    correction = PositionCorrection(options)
    # Secondary modes are deliberately large: this correction uses mode zero.
    chi = torch.tensor([[[[2 + 2j, 4]], [[99, 99]]], [[[6j, 8]], [[99, 99]]]])
    probe = torch.tensor([[[[1j, 2]], [[99, 99]]], [[[1j, 2]], [[99, 99]]]])
    dy = torch.tensor([[[1 + 0j, 0]], [[2 + 0j, 0]]])
    dx = torch.tensor([[[0 + 0j, 1]], [[0 + 0j, 2]]])
    result = correction.get_gradient_update(
        chi, torch.zeros((2, 1, 1, 2)), probe, object_gradients=(dy, dx)
    )
    torch.testing.assert_close(result, torch.tensor([[0.4, 0.4], [1.5, 1.0]]))


def test_gradient_position_update_flat_object_is_zero():
    options = api.LSQMLOptions().probe_position_options.correction_options
    result = PositionCorrection(options).get_gradient_update(
        torch.ones((1, 1, 2, 2), dtype=torch.complex128),
        torch.ones((1, 1, 2, 2)),
        torch.ones((1, 1, 2, 2)),
        object_gradients=(torch.zeros((1, 2, 2)),) * 2,
    )
    torch.testing.assert_close(result, torch.zeros((1, 2)))


def test_cross_correlation_position_update_recovers_integer_translation():
    options = api.EPIEOptions().probe_position_options.correction_options
    options.cross_correlation_scale = 1
    options.cross_correlation_real_space_width = 6
    original = torch.zeros((1, 1, 9, 9), dtype=torch.complex128)
    original[0, 0, 3, 4] = 2 + 1j
    original[0, 0, 4, 5] = 1 - 2j
    shifted = original.roll((2, -1), (-2, -1))
    result = PositionCorrection(options).get_cross_correlation_update(
        original, 2 * (shifted - original), torch.ones_like(original), 0.5
    )
    torch.testing.assert_close(result, torch.tensor([[-2.0, 1.0]]))
