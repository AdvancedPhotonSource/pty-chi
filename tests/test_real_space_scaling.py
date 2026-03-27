import torch

import ptychi.api as api
import ptychi.image_proc as ip
from ptychi.api.task import PtychographyTask
from ptychi.api.options.base import RealSpaceScalingOptions
from ptychi.data_structures.real_space_scaling import RealSpaceScaling


def make_valid_options():
    options = api.LSQMLOptions()
    options.data_options.data = torch.ones((2, 4, 4), dtype=torch.float32)
    options.object_options.initial_guess = torch.ones((1, 8, 8), dtype=torch.complex64)
    options.object_options.pixel_size_m = 1.0
    options.probe_options.initial_guess = torch.ones((1, 1, 4, 4), dtype=torch.complex64)
    options.probe_position_options.position_y_px = torch.tensor([-1.0, 1.0])
    options.probe_position_options.position_x_px = torch.tensor([-1.0, 1.0])
    options.reconstructor_options.default_device = api.Devices.CPU
    return options


def test_real_space_scaling_defaults_and_task_build():
    options = make_valid_options()

    assert options.real_space_scaling_options.initial_guess == 1.0
    assert options.real_space_scaling_options.optimizable is False

    task = PtychographyTask(options)
    scaling = task.reconstructor.parameter_group.real_space_scaling

    assert scaling.shape == (1,)
    assert torch.allclose(scaling.data, torch.tensor([1.0], device=scaling.data.device))
    assert task.get_data("real_space_scaling").shape == (1,)


def test_rescale_images_adjoint():
    torch.manual_seed(0)
    x = torch.randn((3, 7, 5), dtype=torch.complex64)
    y = torch.randn((3, 7, 5), dtype=torch.complex64)
    scale = torch.tensor(1.07)

    ax = ip.rescale_images(x, scale)
    a_star_y = ip.rescale_images(y, scale, adjoint=True)

    lhs = torch.sum(ax.conj() * y)
    rhs = torch.sum(x.conj() * a_star_y)
    assert torch.allclose(lhs, rhs, atol=1e-4, rtol=1e-4)


def test_real_space_scaling_update_matches_formula():
    options = RealSpaceScalingOptions(
        optimizable=True,
        differentiation_method=api.ImageGradientMethods.NEAREST,
    )
    scaling = RealSpaceScaling(data=torch.tensor([1.0]), options=options)

    obj = torch.tensor(
        [[
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ]],
        dtype=torch.complex64,
    ).repeat(2, 1, 1, 1)
    probe = torch.ones((2, 1, 4, 4), dtype=torch.complex64)

    dody, dodx = ip.nearest_neighbor_gradient(obj[:, 0], "backward")
    xgrid = -torch.linspace(-1, 1, 4)
    ygrid = -torch.linspace(-1, 1, 4)
    xgrid = xgrid * ip.tukey_window(4, 0.1)
    ygrid = ygrid * ip.tukey_window(4, 0.1)
    dm_o = dodx * xgrid.view(1, 1, 4) + dody * ygrid.view(1, 4, 1)
    chi = (dm_o * probe[:, 0]).unsqueeze(1)

    delta = scaling.get_update(chi, obj, probe)

    dm_op = dm_o * probe[:, 0]
    nom = torch.real(dm_op.conj() * chi[:, 0]).sum(dim=(-1, -2))
    denom = (dm_op.abs() ** 2).sum(dim=(-1, -2))
    denom_bias = torch.maximum(
        denom.max(),
        torch.tensor(1e-6, device=denom.device, dtype=denom.dtype),
    )
    expected = nom / (denom + denom_bias)
    expected = 0.5 * expected.mean() / 4.0

    assert delta.shape == (1,)
    assert torch.allclose(delta[0], expected)
    assert delta[0] > 0
