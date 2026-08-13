import torch

from ptychi.api.options.base import OPRModeWeightsOptions
from ptychi.data_structures.opr_mode_weights import OPRModeWeights


def test_primary_mode_weight_floor_is_applied_after_optimizer_step():
    options = OPRModeWeightsOptions(
        optimizable=True,
        step_size=1,
        primary_mode_weight_floor=0.5,
    )
    weights = OPRModeWeights(data=torch.tensor([[0.6, 0.2], [0.4, -0.2]]), options=options)
    weights.set_grad(torch.tensor([[0.3, 0.1], [0.1, -0.1]]))

    weights.step_optimizer()

    expected = torch.tensor([[0.5, 0.1], [0.5, -0.1]])
    assert torch.allclose(weights.data, expected)


def test_primary_mode_weight_floor_none_does_not_clip():
    options = OPRModeWeightsOptions(optimizable=True, step_size=1)
    weights = OPRModeWeights(data=torch.tensor([[0.6, 0.2]]), options=options)
    weights.set_grad(torch.tensor([[0.3, 0.1]]))

    weights.step_optimizer()

    assert torch.allclose(weights.data, torch.tensor([[0.3, 0.1]]))
