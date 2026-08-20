import torch

import ptychi.api as api
from ptychi.position_correction import PositionCorrection


def test_gradient_update_uses_undamped_least_squares_projection(monkeypatch):
    dody = torch.tensor([[[1.0, 0.0]], [[0.0, 0.0]]])
    dodx = torch.tensor([[[1.0, 2.0]], [[2.0, 0.0]]])
    monkeypatch.setattr(
        "ptychi.position_correction.ip.fourier_gradient",
        lambda _: (dody, dodx),
    )

    chi = torch.tensor(
        [[[[3.0, 4.0]]], [[[5.0, 6.0]]]], dtype=torch.complex64
    )
    obj_patches = torch.zeros((2, 1, 1, 2), dtype=torch.complex64)
    probe = torch.ones((2, 1, 1, 2), dtype=torch.complex64)
    options = api.LSQMLOptions().probe_position_options.correction_options
    correction = PositionCorrection(options)

    update = correction.get_gradient_update(chi, obj_patches, probe)

    expected = torch.tensor([[3.0, 11.0 / 5.0], [0.0, 10.0 / 4.0]])
    torch.testing.assert_close(update, expected)
