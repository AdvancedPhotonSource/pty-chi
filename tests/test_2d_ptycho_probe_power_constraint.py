import torch

from ptychi.api.options import base as obase
from ptychi.data_structures.object import Object
from ptychi.data_structures.probe import Probe


def _make_object(data: torch.Tensor) -> Object:
    options = obase.ObjectOptions()
    options.optimizable = False
    return Object(data=data, options=options)


def _make_probe(data: torch.Tensor, probe_power: float, scale_object: bool = True) -> Probe:
    options = obase.ProbeOptions()
    options.optimizable = False
    options.power_constraint.probe_power = probe_power
    options.power_constraint.scale_object = scale_object
    return Probe(data=data, options=options)


def test_probe_power_constraint_scales_probe_to_target():
    probe_data = torch.ones((1, 1, 2, 2), dtype=torch.complex64)
    object_data = torch.full((1, 2, 2), 2 + 0j, dtype=torch.complex64)

    target_power = 8.0
    probe = _make_probe(probe_data, probe_power=target_power, scale_object=True)
    obj = _make_object(object_data)

    probe.constrain_probe_power(obj)

    probe_power = torch.sum(torch.abs(probe.get_opr_mode(0)) ** 2)
    assert torch.isclose(probe_power, torch.tensor(target_power, dtype=probe_power.dtype))

    scale = torch.sqrt(torch.tensor(target_power, dtype=probe_power.dtype) / 4.0)
    expected_object = object_data / scale
    assert torch.allclose(obj.data, expected_object)


def test_probe_power_constraint_respects_scale_object_option():
    probe_data = torch.ones((1, 1, 2, 2), dtype=torch.complex64)
    object_data = torch.full((1, 2, 2), 3 + 0j, dtype=torch.complex64)

    target_power = 4.0
    probe = _make_probe(probe_data, probe_power=target_power, scale_object=False)
    obj = _make_object(object_data)

    probe.constrain_probe_power(obj)

    probe_power = torch.sum(torch.abs(probe.get_opr_mode(0)) ** 2)
    assert torch.isclose(probe_power, torch.tensor(target_power, dtype=probe_power.dtype))
    assert torch.allclose(obj.data, object_data)
