import torch

import ptychi.api.options.base as optionsbase
import ptychi.api.enums as enums
from ptychi.data_structures.probe import Probe


def _make_probe(data, fixed_support, fixed_params):
    options = optionsbase.ProbeOptions()
    options.optimizable = False
    options.support_constraint.enabled = True
    options.support_constraint.threshold = 0.0
    options.support_constraint.fixed_probe_support = fixed_support
    options.support_constraint.fixed_probe_support_params = fixed_params
    return Probe(data=data, options=options)


def _ellipse_mask(rows, cols, center_r, center_c, radius_r, radius_c):
    y, x = torch.meshgrid(
        torch.arange(rows),
        torch.arange(cols),
        indexing="ij",
    )
    return ((y - center_r) ** 2 / radius_r ** 2) + ((x - center_c) ** 2 / radius_c ** 2) <= 1


def test_constrain_support_fixed_ellipse():
    data = torch.ones((1, 1, 8, 8), dtype=torch.complex64)
    center_r, center_c = 4.0, 4.0
    radius_r, radius_c = 2.0, 3.0
    probe = _make_probe(
        data,
        enums.ProbeSupportMethods.ELLIPSE,
        torch.tensor([center_r, center_c, radius_r, radius_c]),
    )

    probe.constrain_support()

    result = probe.data[0, 0]
    mask = _ellipse_mask(8, 8, center_r, center_c, radius_r, radius_c)
    assert torch.allclose(result[~mask], torch.zeros_like(result[~mask]))
    assert torch.allclose(result[int(center_r), int(center_c)], torch.tensor(1 + 0j))


def test_constrain_support_fixed_rectangle():
    data = torch.ones((1, 1, 8, 8), dtype=torch.complex64)
    center_r, center_c = 4.0, 4.0
    len_r, len_c = 2.0, 3.0
    probe = _make_probe(
        data,
        enums.ProbeSupportMethods.RECTANGLE,
        torch.tensor([center_r, center_c, len_r, len_c]),
    )

    probe.constrain_support()

    result = probe.data[0, 0]
    mask = torch.zeros((8, 8), dtype=torch.bool)
    mask[int(center_r - len_r) : int(center_r + len_r), int(center_c - len_c) : int(center_c + len_c)] = True
    assert torch.allclose(result[~mask], torch.zeros_like(result[~mask]))
    assert torch.allclose(result[int(center_r), int(center_c)], torch.tensor(1 + 0j))
