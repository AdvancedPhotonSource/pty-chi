import pytest
import torch

import ptychi.api as api
import ptychi.maths as pmath
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_default_complex_dtype, set_default_complex_dtype


@pytest.fixture
def task():
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
    options.object_options.remove_object_probe_ambiguity.enabled = False
    options.probe_options.optimizable = True
    options.probe_position_options.optimizable = False
    generator = torch.Generator(device="cpu").manual_seed(42)

    def random_complex(shape):
        return torch.randn(shape, generator=generator, dtype=torch.complex128, device="cpu")

    try:
        yield PtychographyTask(
            options,
            diffraction_data=torch.ones(2, 5, 5, device="cpu"),
            object_data=random_complex((1, 17, 17)),
            probe_data=random_complex((1, 4, 5, 5)),
            probe_position_x_px=torch.tensor([-1.3, 2.4], device="cpu"),
            probe_position_y_px=torch.tensor([-2.2, 1.1], device="cpu"),
        )
    finally:
        torch.set_default_dtype(old_dtype)
        torch.set_default_device(old_device)
        set_default_complex_dtype(old_complex_dtype)


@pytest.mark.parametrize("sort_by_occupancy", [False, True])
@pytest.mark.parametrize("zero_mode", [False, True])
def test_cg_history_preserves_search_path_after_mode_rotation(task, sort_by_occupancy, zero_mode):
    reconstructor = task.reconstructor
    probe = task.probe
    probe.options.orthogonalize_incoherent_modes.sort_by_occupancy = sort_by_occupancy
    with torch.no_grad():
        if zero_mode:
            data = probe.data.clone()
            data[0, -1] = 0
            probe.set_data(data)
        reconstructor.compute_updates(torch.arange(2), reconstructor.dataset.patterns)
        # Include a direction in the zero mode to exercise the full transformation.
        reconstructor.eta_p = torch.randn_like(reconstructor.eta_p)
        before = probe.get_opr_mode(0).clone()
        direction = reconstructor.eta_p.clone()
        object_direction = reconstructor.eta_o.clone()
        patches = reconstructor.forward_model.intermediate_variables["obj_patches"]
        propagator = reconstructor.forward_model.free_space_propagator
        reconstructor.run_post_epoch_hooks()
        after = probe.get_opr_mode(0)
        rotated_direction = reconstructor.eta_p

        assert not torch.allclose(after, before)
        torch.testing.assert_close(reconstructor.eta_o, object_direction)
        for step in [-0.1, 0.0, 0.1]:
            old_field = propagator.propagate_forward(patches * (before + step * direction))
            new_field = propagator.propagate_forward(patches * (after + step * rotated_direction))
            torch.testing.assert_close(
                new_field.abs().square().sum(1), old_field.abs().square().sum(1),
                rtol=1e-10, atol=1e-10,
            )
        reconstructor.current_epoch = 1
        updates, _ = reconstructor.compute_updates(torch.arange(2), reconstructor.dataset.patterns)
        assert all(torch.isfinite(update).all() for update in updates if update is not None)


@pytest.mark.parametrize("method", ["gs", "svd"])
@pytest.mark.parametrize("sort_by_occupancy", [False, True])
def test_probe_returns_complete_mode_transformation(task, method, sort_by_occupancy):
    probe = task.probe
    probe.orthogonalize_incoherent_modes_method = method
    probe.options.orthogonalize_incoherent_modes.sort_by_occupancy = sort_by_occupancy
    with torch.no_grad():
        before = probe.get_opr_mode(0).clone()
        transform = probe.constrain_incoherent_modes_orthogonality(return_transform=True)
        expected = (transform @ before.flatten(1)).reshape_as(before)
        torch.testing.assert_close(probe.get_opr_mode(0), expected)


@pytest.mark.parametrize("method", ["gs", "svd"])
def test_batched_orthogonalization_returns_mode_transformation(method):
    x = torch.randn(2, 3, 4, 5, dtype=torch.complex128, device="cpu")
    function = getattr(pmath, f"orthogonalize_{method}")
    result, transform = function(x, dim=(-2, -1), group_dim=1, return_transform=True)
    torch.testing.assert_close(result, (transform @ x.flatten(2)).reshape_as(x))
    torch.testing.assert_close(result, function(x, dim=(-2, -1), group_dim=1))
