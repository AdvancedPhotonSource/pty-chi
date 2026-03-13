import pytest
import torch
from types import SimpleNamespace

import ptychi.image_proc as ip
from ptychi.api.options import base as obase
from ptychi.data_structures.probe import Probe


def _make_probe(
    data: torch.Tensor,
    *,
    center_modes_individually: bool,
    use_intensity_for_com: bool,
) -> Probe:
    options = obase.ProbeOptions()
    options.optimizable = False
    options.center_constraint.enabled = True
    options.center_constraint.center_modes_individually = center_modes_individually
    options.center_constraint.use_intensity_for_com = use_intensity_for_com
    return Probe(data=data, options=options)


def test_center_probe_can_shift_incoherent_modes_individually():
    data = torch.zeros((2, 2, 7, 7), dtype=torch.complex64)
    data[0, 0, 1, 2] = 1 + 0j
    data[0, 1, 4, 1] = 1 + 0j

    data[1, 0, 0, 0] = 2 + 0j
    data[1, 0, 2, 5] = 3 + 0j
    data[1, 1, 5, 6] = 4 + 0j
    data[1, 1, 6, 1] = 5 + 0j

    secondary_opr_modes_before = data[1:].clone()
    probe = _make_probe(
        data,
        center_modes_individually=True,
        use_intensity_for_com=False,
    )

    probe.center_probe()

    expected_center = torch.tensor([[3.0, 3.0], [3.0, 3.0]])
    centered_mode_com = ip.find_center_of_mass(torch.abs(probe.data[0]) ** 2)

    assert torch.allclose(centered_mode_com, expected_center, atol=1e-4)
    assert torch.allclose(probe.data[1:], secondary_opr_modes_before)


def test_probe_center_constraint_check_rejects_individual_mode_centering_with_intensity_com():
    options = obase.ProbeOptions()
    options.initial_guess = torch.zeros((1, 1, 7, 7), dtype=torch.complex64)
    options.center_constraint.center_modes_individually = True
    options.center_constraint.use_intensity_for_com = True

    task_options = SimpleNamespace(
        object_options=SimpleNamespace(
            remove_object_probe_ambiguity=SimpleNamespace(enabled=False)
        )
    )

    with pytest.raises(ValueError, match="use_intensity_for_com"):
        options.check(task_options)
