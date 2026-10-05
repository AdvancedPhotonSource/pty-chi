import pytest
import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_default_complex_dtype, set_default_complex_dtype


@pytest.fixture
def task(request):
    pixel_size, aspect_ratio, n_modes, optimize_probe = request.param
    old_dtype = torch.get_default_dtype()
    old_device = torch.get_default_device()
    old_complex_dtype = get_default_complex_dtype()
    options = api.BHOptions()
    options.reconstructor_options.default_device = api.Devices.CPU
    options.reconstructor_options.default_dtype = api.Dtypes.FLOAT64
    options.reconstructor_options.method = "CG"
    options.reconstructor_options.batch_size = 2
    options.reconstructor_options.forward_model_options.pad_for_shift = 2
    options.data_options.fft_shift = False
    options.object_options.pixel_size_m = 1.0
    options.probe_options.pixel_size_m = pixel_size
    options.probe_options.pixel_size_aspect_ratio = aspect_ratio
    options.probe_options.optimizable = optimize_probe
    options.probe_options.rho = 0.3
    options.probe_position_options.optimizable = False
    generator = torch.Generator(device="cpu").manual_seed(311)

    def random_complex(shape):
        return torch.randn(shape, generator=generator, dtype=torch.complex128, device="cpu")

    try:
        yield PtychographyTask(
            options,
            diffraction_data=torch.ones(2, 5, 5, device="cpu"),
            object_data=random_complex((1, 17, 19)) + 1,
            probe_data=random_complex((1, n_modes, 5, 5)),
            probe_position_x_px=torch.tensor([-1.3, 2.4], device="cpu"),
            probe_position_y_px=torch.tensor([-2.2, 1.1], device="cpu"),
        )
    finally:
        torch.set_default_dtype(old_dtype)
        torch.set_default_device(old_device)
        set_default_complex_dtype(old_complex_dtype)


@pytest.mark.parametrize(
    "task",
    [
        (pixel_size, aspect_ratio, n_modes, optimize_probe)
        for pixel_size, aspect_ratio in [(1.0, 1.0), (0.8, 1.0), (1.2, 1.0), (1.0, 1.3)]
        for n_modes in [1, 4]
        for optimize_probe in [False, True]
    ],
    indirect=True,
)
def test_bh_resampled_step_matches_autograd_curvature(task):
    reconstructor = task.reconstructor
    # Isolate resampling from the amplitude stabilization approximation.
    reconstructor.eps = 0.0
    indices = torch.arange(2)
    data = reconstructor.dataset.patterns
    for epoch in range(2):
        # The first iteration uses GD; the second exercises the cached CG patches.
        reconstructor.current_epoch = epoch
        with torch.no_grad():
            updates, _ = reconstructor.compute_updates(indices, data)
        parameters = [task.object.tensor.data]
        directions = [torch.view_as_real(reconstructor.eta_o)[None]]
        if task.probe.optimizable:
            parameters.append(task.probe.tensor.data)
            directions.append(
                torch.view_as_real(task.probe.options.rho * reconstructor.eta_p)[None]
            )

        prediction = reconstructor.forward_model(indices)
        loss = (prediction.sqrt() - data.sqrt()).square().sum()
        gradients = torch.autograd.grad(loss, parameters, create_graph=True)
        slope = sum((gradient * direction).sum() for gradient, direction in zip(gradients, directions))
        hessian_products = torch.autograd.grad(slope, parameters)
        curvature = sum(
            (product * direction).sum() for product, direction in zip(hessian_products, directions)
        )
        alpha = -slope.detach() / curvature
        for update, direction in zip(updates, directions):
            torch.testing.assert_close(
                update, alpha * torch.view_as_complex(direction), rtol=1e-9, atol=1e-9
            )
        with torch.no_grad():
            reconstructor.apply_updates(*updates)


@pytest.mark.parametrize(
    "task",
    [
        (pixel_size, aspect_ratio, n_modes, True)
        for pixel_size, aspect_ratio in [(1.0, 1.0), (0.8, 1.0), (1.2, 1.0), (1.0, 1.3)]
        for n_modes in [1, 4]
    ],
    indirect=True,
)
def test_bh_object_and_probe_gradients_match_autograd(task):
    reconstructor = task.reconstructor
    reconstructor.eps = 0.0
    indices = torch.arange(2)
    data = reconstructor.dataset.patterns
    reconstructor.positions = task.probe_positions.tensor[indices]
    mask = reconstructor.get_constrained_pixel_mask(data).clone()
    mask[:, ::2, ::2] = False
    reconstructor.current_constrained_pixel_mask = mask[:, None]

    prediction = reconstructor.forward_model(indices)
    loss = (mask * (prediction.sqrt() - data.sqrt()).square()).sum()
    expected_object, expected_probe = torch.autograd.grad(
        loss, (task.object.tensor.data, task.probe.tensor.data)
    )
    intermediates = reconstructor.forward_model.intermediate_variables
    with torch.no_grad():
        gradF = reconstructor.gradientF(intermediates["psi_far"], data.sqrt()[:, None])
        object_gradient, _ = reconstructor.gradient_o(task.probe.get_opr_mode(0), gradF)
        probe_gradient = reconstructor.gradient_p(intermediates["obj_patches"], gradF)

    torch.testing.assert_close(
        object_gradient, torch.view_as_complex(expected_object.contiguous())[0],
        rtol=1e-11, atol=1e-11,
    )
    torch.testing.assert_close(
        probe_gradient, torch.view_as_complex(expected_probe.contiguous())[0],
        rtol=1e-11, atol=1e-11,
    )
