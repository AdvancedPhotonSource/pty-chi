"""BH search-direction bookkeeping, complementing the derivative tests."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ptychi.reconstructors.bh import BHReconstructor

pytestmark = pytest.mark.usefixtures("numerical_cpu")


@pytest.mark.parametrize(
    "method,epoch,expected",
    [
        ("GD", 0, [-1 + 1j, -2]),
        ("GD", 1, [-1 + 1j, -2]),
        ("CG", 0, [-1 + 1j, -2]),
        ("CG", 1, [0 + 1j, -0.5]),
    ],
)
def test_bh_search_direction_and_short_batch_normalization(method, epoch, expected):
    recon = BHReconstructor.__new__(BHReconstructor)
    recon.options = SimpleNamespace(method=method)
    recon.current_epoch = epoch
    recon.batch_size = 4
    recon.positions = torch.zeros((2, 2))
    recon.indices = torch.tensor([0, 2])
    recon.eta_o = torch.tensor([4 + 0j, 6])
    recon.eta_op = torch.tensor([[[[4 + 0j, 6]]], [[[99, 99]]], [[[4, 6]]]])
    recon.gradient_o = Mock(
        return_value=(torch.tensor([2 - 2j, 4]), torch.tensor([[[[2 - 2j, 4]]], [[[2 - 2j, 4]]]]))
    )
    recon.refresh_object_direction_patches = lambda direction, pos, patches: patches
    recon.calc_beta_object = Mock(return_value=0.5)
    recon.calc_alpha_object = Mock(return_value=1.0)
    actual = recon.compute_updates_object(torch.zeros((2, 1, 1, 2)), None, None, None, None)
    torch.testing.assert_close(actual, torch.tensor(expected))
    assert recon.calc_beta_object.call_count == (method == "CG" and epoch > 0)
    if method == "CG":
        torch.testing.assert_close(recon.eta_o, 2 * torch.tensor(expected))
        torch.testing.assert_close(recon.eta_op[1], torch.tensor([[[99 + 0j, 99]]]))
