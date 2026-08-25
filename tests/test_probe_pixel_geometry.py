import json

import pytest
import torch
from pydantic import ValidationError

import ptychi.api as api
import ptychi.image_proc as image_proc
from ptychi.api.task import PtychographyTask


def _small_task(options):
    options.reconstructor_options.default_device = api.Devices.CPU
    options.reconstructor_options.num_epochs = 1
    options.reconstructor_options.allow_nondeterministic_algorithms = False
    if hasattr(options.reconstructor_options, "batch_size"):
        options.reconstructor_options.batch_size = 2
    if hasattr(options.reconstructor_options, "chunk_length"):
        options.reconstructor_options.chunk_length = 1
    options.data_options.fft_shift = False
    options.object_options.pixel_size_m = 1.0
    options.object_options.pixel_size_aspect_ratio = 1.0
    options.probe_options.pixel_size_m = 0.8
    options.probe_options.pixel_size_aspect_ratio = 1.2
    options.probe_options.optimizable = False
    options.probe_position_options.optimizable = False

    diffraction = torch.rand(2, 5, 5, device="cpu") + 0.2
    object_data = torch.ones((1, 17, 19), dtype=torch.complex64, device="cpu")
    probe_data = torch.ones((1, 1, 5, 5), dtype=torch.complex64, device="cpu")
    position_x = torch.tensor([-2.0, 2.0], device="cpu")
    position_y = torch.tensor([-1.0, 1.0], device="cpu")
    original_x = position_x.clone()
    original_y = position_y.clone()

    task = PtychographyTask(
        options,
        diffraction_data=diffraction,
        object_data=object_data,
        probe_data=probe_data,
        probe_position_x_px=position_x,
        probe_position_y_px=position_y,
    )
    assert torch.equal(position_x, original_x)
    assert torch.equal(position_y, original_y)
    return task


def test_probe_pixel_options_validate_inherit_and_serialize():
    options = api.LSQMLOptions()
    assert options.probe_options.pixel_size_m is None
    assert options.probe_options.pixel_size_aspect_ratio is None

    options.probe_options.pixel_size_m = 2e-9
    options.probe_options.pixel_size_aspect_ratio = 1.5
    values = options.get_dict()["probe_options"]
    assert values["pixel_size_m"] == 2e-9
    assert values["pixel_size_aspect_ratio"] == 1.5
    json.dumps(options.get_dict())

    with pytest.raises(ValidationError):
        options.probe_options.pixel_size_m = 0
    with pytest.raises(ValidationError):
        options.probe_options.pixel_size_aspect_ratio = -1


@pytest.mark.parametrize(
    ("source_shape", "target_shape"),
    [
        ((5, 6), (8, 3)),
        ((4, 4), (7, 7)),
        ((7, 8), (3, 10)),
        ((5, 5), (4, 4)),
        ((4, 5), (5, 4)),
    ],
)
def test_fourier_resize_preserves_constants_and_has_exact_adjoint(
    source_shape, target_shape
):
    constant = torch.ones(source_shape)
    resized_constant = image_proc.fourier_resize(constant, target_shape)
    assert torch.allclose(resized_constant, torch.ones(target_shape), atol=1e-6)

    source = torch.randn(source_shape, dtype=torch.complex64)
    target = torch.randn(target_shape, dtype=torch.complex64)
    resized = image_proc.fourier_resize(source, target_shape)
    adjoint = image_proc.fourier_resize(target, source_shape, adjoint=True)
    lhs = torch.vdot(resized.reshape(-1), target.reshape(-1))
    rhs = torch.vdot(source.reshape(-1), adjoint.reshape(-1))
    assert torch.allclose(lhs, rhs, atol=2e-5, rtol=2e-5)


def test_anisotropic_geometry_and_versioned_caches(monkeypatch):
    task = _small_task(api.EPIEOptions())
    forward_model = task.reconstructor.forward_model
    expected_shape = (
        round(17 / (0.8 / 1.2)),
        round(19 / 0.8),
    )
    assert forward_model.probe_grid_object_shape == expected_shape
    assert forward_model.probe_grid_scale == pytest.approx(
        (expected_shape[0] / 17, expected_shape[1] / 19)
    )

    resize_calls = 0
    original_resize = image_proc.fourier_resize

    def counted_resize(*args, **kwargs):
        nonlocal resize_calls
        resize_calls += 1
        return original_resize(*args, **kwargs)

    monkeypatch.setattr(image_proc, "fourier_resize", counted_resize)
    task.object.preconditioner = torch.ones(task.object.lateral_shape)
    with torch.no_grad():
        first_object = forward_model.get_probe_grid_object()
        assert forward_model.get_probe_grid_object() is first_object
        first_preconditioner = forward_model.get_probe_grid_preconditioner()
        assert forward_model.get_probe_grid_preconditioner() is first_preconditioner
    assert resize_calls == 2

    with torch.no_grad():
        task.object.set_data(task.object.data + 1)
        forward_model.get_probe_grid_object()
        task.object.preconditioner.add_(1)
        forward_model.get_probe_grid_preconditioner()
    assert resize_calls == 4
    assert torch.all(forward_model.get_probe_grid_preconditioner() >= 0)
    assert not any("resampling" in key for key in forward_model.state_dict())


def test_inherited_probe_geometry_uses_identity_path():
    task = _small_task(api.EPIEOptions())
    forward_model = task.reconstructor.forward_model
    task.probe.options.pixel_size_m = None
    task.probe.options.pixel_size_aspect_ratio = None
    indices = torch.arange(task.probe_positions.n_scan_points)

    actual = forward_model.extract_object_patches(indices)
    expected = task.object.extract_patches(
        task.probe_positions.tensor.round().int(),
        task.probe.get_spatial_shape(),
        integer_mode=True,
    )
    assert not forward_model.resampling_enabled
    assert forward_model.probe_grid_object_shape == task.object.lateral_shape
    assert torch.equal(actual, expected)


def test_object_patch_extraction_accepts_an_explicit_object_array():
    task = _small_task(api.EPIEOptions())
    replacement = torch.full_like(task.object.data, 3 + 2j)
    positions = task.probe_positions.tensor.round().int()

    patches = task.object.extract_patches(
        positions,
        task.probe.get_spatial_shape(),
        integer_mode=True,
        object_array=replacement,
    )

    assert torch.all(patches == 3 + 2j)
    assert torch.all(task.object.data == 1)


def test_autodiff_graph_cache_policy():
    task = _small_task(api.AutodiffPtychographyOptions())
    forward_model = task.reconstructor.forward_model

    first = forward_model.get_probe_grid_object()
    second = forward_model.get_probe_grid_object()
    assert first is not second

    task.object.set_optimizable(False)
    third = forward_model.get_probe_grid_object()
    fourth = forward_model.get_probe_grid_object()
    assert third is fourth


@pytest.mark.parametrize(
    ("options_factory", "integer_mode"),
    [(api.EPIEOptions, True), (api.BHOptions, False)],
)
def test_resampled_patch_extraction_and_placement_are_adjoint(
    options_factory, integer_mode
):
    options = options_factory()
    if not integer_mode:
        options.reconstructor_options.forward_model_options.pad_for_shift = 1
    task = _small_task(options)
    forward_model = task.reconstructor.forward_model
    indices = torch.arange(task.probe_positions.n_scan_points)

    source = torch.randn_like(task.object.data)
    with torch.no_grad():
        task.object.set_data(source)
    patches = forward_model.extract_object_patches(indices)[:, 0]
    cotangent = torch.randn_like(patches)
    placed = forward_model.place_object_patches_on_probe_grid(
        task.probe_positions.tensor,
        cotangent,
        integer_mode=integer_mode,
    )
    native_adjoint = forward_model.object_update_to_native_grid(placed)

    lhs = torch.vdot(patches.reshape(-1), cotangent.reshape(-1))
    rhs = torch.vdot(source.reshape(-1), native_adjoint.reshape(-1))
    assert torch.allclose(lhs, rhs, atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize(
    "options_factory",
    [
        api.AutodiffPtychographyOptions,
        api.EPIEOptions,
        api.BHOptions,
        api.DMOptions,
        api.RAAROptions,
    ],
)
def test_lightweight_mismatched_grid_reconstruction(options_factory):
    options = options_factory()
    if isinstance(options, api.BHOptions):
        options.reconstructor_options.forward_model_options.pad_for_shift = 1
    task = _small_task(options)
    task.run()
    assert torch.isfinite(task.get_data_to_cpu("object")).all()


@pytest.mark.parametrize(
    "batching_mode",
    [api.BatchingModes.RANDOM, api.BatchingModes.UNIFORM, api.BatchingModes.COMPACT],
)
def test_lightweight_mismatched_grid_lsqml(batching_mode):
    options = api.LSQMLOptions()
    options.reconstructor_options.rescale_probe_intensity_in_first_epoch = False
    options.reconstructor_options.batching_mode = batching_mode
    options.reconstructor_options.batch_size = 1
    task = _small_task(options)
    task.run()
    assert torch.isfinite(task.get_data_to_cpu("object")).all()
