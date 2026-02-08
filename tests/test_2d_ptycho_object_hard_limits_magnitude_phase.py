import torch

import ptychi.api as api
from ptychi.data_structures.object import PlanarObject


def test_object_hard_limits_magnitude_phase_plumbing_and_constraints():
    options = api.LSQMLOptions()
    options.object_options.optimizable = False
    options.object_options.hard_limits_magnitude_phase.enabled = True
    options.object_options.hard_limits_magnitude_phase.abs_lim = torch.tensor([0.5, 1.0])
    options.object_options.hard_limits_magnitude_phase.phase_lim = torch.tensor(
        [-0.25 * torch.pi, 0.25 * torch.pi]
    )

    mags = torch.tensor([[[2.0, 0.1], [0.8, 1.2]]])
    phases = torch.tensor(
        [[[0.5 * torch.pi, -0.5 * torch.pi], [0.1 * torch.pi, -0.1 * torch.pi]]]
    )
    data = torch.polar(mags, phases)

    obj = PlanarObject(data=data, options=options.object_options)

    assert obj.options.hard_limits_magnitude_phase.enabled is True
    assert torch.equal(
        obj.options.hard_limits_magnitude_phase.abs_lim,
        options.object_options.hard_limits_magnitude_phase.abs_lim,
    )
    assert torch.equal(
        obj.options.hard_limits_magnitude_phase.phase_lim,
        options.object_options.hard_limits_magnitude_phase.phase_lim,
    )

    obj.constrain_hard_limits_magnitude_phase()

    expected_mags = torch.clamp(
        mags,
        min=options.object_options.hard_limits_magnitude_phase.abs_lim[0],
        max=options.object_options.hard_limits_magnitude_phase.abs_lim[1],
    )
    expected_phases = torch.clamp(
        phases,
        min=options.object_options.hard_limits_magnitude_phase.phase_lim[0],
        max=options.object_options.hard_limits_magnitude_phase.phase_lim[1],
    )
    expected = torch.polar(expected_mags, expected_phases)

    assert torch.allclose(obj.data, expected, atol=1e-6)
