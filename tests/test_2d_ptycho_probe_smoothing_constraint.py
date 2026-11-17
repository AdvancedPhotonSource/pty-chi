import numpy as np
import pytest
import scipy.ndimage as ndi
import torch
from pydantic import ValidationError
from torchvision.transforms.functional import gaussian_blur

import ptychi.api.enums as enums
import ptychi.image_proc as ip
from ptychi.api.options import base as obase
from ptychi.data_structures.probe import Probe


def _make_probe(data: torch.Tensor, **smoothing_options) -> Probe:
    options = obase.ProbeOptions()
    options.optimizable = False
    options.smoothing_constraint.enabled = True
    for key, value in smoothing_options.items():
        setattr(options.smoothing_constraint, key, value)
    return Probe(data=data, options=options)


def _random_probe() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn((2, 3, 20, 17), dtype=torch.complex64)


def test_median_filter_2d_matches_scipy_mirror():
    torch.manual_seed(0)
    x = torch.rand((3, 25, 30))

    actual = ip.median_filter_2d(x, window_size=5)
    expected = ndi.median_filter(x.numpy(), size=(1, 5, 5), mode="mirror")

    assert np.allclose(actual.numpy(), expected)


def test_median_filter_2d_rejects_even_window():
    with pytest.raises(ValueError, match="odd"):
        ip.median_filter_2d(torch.rand((5, 5)), window_size=4)


@pytest.mark.parametrize(
    ("smoothing_options", "filter_magnitude"),
    [
        (
            {"method": enums.ProbeSmoothingMethods.MEDIAN, "kernel_size": 7},
            lambda mag: ip.median_filter_2d(mag, window_size=7),
        ),
        (
            {"method": enums.ProbeSmoothingMethods.GAUSSIAN, "kernel_size": 3},
            lambda mag: gaussian_blur(mag, kernel_size=3),
        ),
        (
            {"method": enums.ProbeSmoothingMethods.GAUSSIAN, "kernel_size": 5, "sigma": 4.0},
            lambda mag: gaussian_blur(mag, kernel_size=5, sigma=4.0),
        ),
    ],
)
def test_smooth_probe_filters_magnitude_of_first_opr_mode(smoothing_options, filter_magnitude):
    data = _random_probe()
    probe = _make_probe(data.clone(), **smoothing_options)

    probe.smooth_probe()

    assert torch.allclose(probe.data[0].abs(), filter_magnitude(data[0].abs()), atol=1e-6)
    assert torch.allclose(probe.data[0].angle(), data[0].angle(), atol=1e-6)
    assert torch.equal(probe.data[1:], data[1:])


def test_probe_smoothing_constraint_is_disabled_by_default():
    assert obase.ProbeOptions().smoothing_constraint.enabled is False


def test_probe_smoothing_constraint_rejects_even_kernel_size():
    with pytest.raises(ValidationError, match="odd"):
        obase.ProbeSmoothingConstraintOptions(kernel_size=4)
