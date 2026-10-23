import json

import pytest
import torch

import ptychi.api as api
import ptychi.image_proc as ip
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_default_complex_dtype, set_default_complex_dtype


@pytest.fixture(autouse=True)
def restore_defaults():
    dtype, device = torch.get_default_dtype(), torch.get_default_device()
    complex_dtype = get_default_complex_dtype()
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    set_default_complex_dtype(complex_dtype)


@pytest.mark.parametrize("method", ["fourier", "bilinear", "nearest"])
@pytest.mark.parametrize("source,target", [
    ((7, 8), (5, 6)), ((5, 6), (5, 6)), ((5, 6), (8, 9)),
    ((7, 4), (4, 7)), ((1, 1), (2, 3)),
])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128, torch.float32, torch.complex64])
def test_adjoint(method, source, target, dtype):
    torch.manual_seed(912)
    image = torch.randn(source, dtype=dtype, requires_grad=True)
    positions = torch.tensor([[2.2, 3.7], [-10.4, 19.1], [3., 4.]])
    cotangent = torch.randn((3, *target), dtype=dtype)
    patches = ip.extract_patches_full_object(image, positions, target, method)
    adjoint = ip.place_patches_full_object(torch.zeros_like(image), positions, cotangent, method=method)
    expected = torch.autograd.grad(patches, image, cotangent)[0]
    tolerance = 3e-5 if image.real.dtype == torch.float32 else 1e-11
    torch.testing.assert_close(adjoint, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        (patches.conj() * cotangent).sum(), (image.conj() * adjoint).sum(),
        atol=tolerance, rtol=tolerance,
    )
    torch.testing.assert_close(
        ip.place_patches_full_object(image.detach(), positions, cotangent, method=method),
        image.detach() + adjoint,
    )


@pytest.mark.parametrize("method", ["fourier", "bilinear", "nearest"])
def test_periodic_roll_reference_and_padding(method):
    image = torch.arange(30.).reshape(5, 6).to(torch.complex128)
    shift = torch.tensor([1., -2.]) if method == "fourier" else torch.tensor([1.3, -2.4])
    center = torch.tensor([2., 2.5])
    result = ip.extract_patches_full_object(image, (center - shift)[None], (5, 6), method)[0]
    if method == "bilinear":
        lower = shift.floor().long()
        fraction = shift - lower
        expected = sum(
            (fraction[0] if y else 1 - fraction[0])
            * (fraction[1] if x else 1 - fraction[1])
            * torch.roll(image, (int(lower[0]) + y, int(lower[1]) + x), (0, 1))
            for y in (0, 1) for x in (0, 1)
        )
    else:
        expected = torch.roll(image, tuple(shift.round().long().tolist()), (0, 1))
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
    # Fixed center alignment: source starts at (1, 1) within this larger patch.
    padded = ip.extract_patches_full_object(image, torch.tensor([[2., 2.5]]), (7, 8), method)[0]
    torch.testing.assert_close(padded[1:6, 1:7], image, atol=1e-12, rtol=1e-12)
    assert padded[0].count_nonzero() == 0
    assert padded[:, -1].count_nonzero() == 0
    with pytest.raises(ValueError, match="additive"):
        ip.place_patches_full_object(image, center[None], image, op="set", method=method)


@pytest.mark.parametrize("method", ["fourier", "bilinear", "nearest"])
def test_position_derivatives(method):
    torch.manual_seed(913)
    image = torch.randn(7, 8, dtype=torch.complex128)
    positions = torch.tensor([[2.23, 3.37], [-1.2, 4.1]], requires_grad=True)
    weights = torch.randn(2, 6, 9, dtype=torch.complex128)
    def loss(p):
        return (ip.extract_patches_full_object(image, p, (6, 9), method).conj() * weights).real.sum()
    gradient = torch.autograd.functional.jacobian(loss, positions)
    hessian = torch.autograd.functional.hessian(loss, positions)
    for axis, order in enumerate(((1, 0), (0, 1))):
        derivative = ip.extract_patches_full_object(image, positions, (6, 9), method, derivative=order)
        expected = (derivative.conj() * weights).real.sum((-2, -1))
        torch.testing.assert_close(gradient[:, axis], expected, atol=1e-11, rtol=1e-11)
        for other in range(2):
            second = list(order)
            second[other] += 1
            derivative = ip.extract_patches_full_object(
                image, positions, (6, 9), method, derivative=second,
            )
            expected = (derivative.conj() * weights).real.sum((-2, -1))
            torch.testing.assert_close(
                hessian[torch.arange(2), axis, torch.arange(2), other], expected,
                atol=1e-11, rtol=1e-11,
            )


def make_task(options=None, object_shape=(9, 10), probe_shape=(8, 9), method="fourier",
              n_modes=1, n_slices=1, opr=False, optimize_positions=False, optimize_probe=False,
              optimize_opr=False):
    options = options or api.EPIEOptions()
    options.reconstructor_options.default_device = api.Devices.CPU
    options.reconstructor_options.default_dtype = api.Dtypes.FLOAT64
    options.reconstructor_options.num_epochs = 3
    options.reconstructor_options.batch_size = 3
    if hasattr(options.reconstructor_options, "chunk_length"):
        options.reconstructor_options.chunk_length = 2
    if hasattr(options.reconstructor_options, "rescale_probe_intensity_in_first_epoch"):
        options.reconstructor_options.rescale_probe_intensity_in_first_epoch = False
    options.data_options.fft_shift = False
    options.data_options.wavelength_m = 1e-10
    options.data_options.free_space_propagation_distance_m = 0.01
    options.object_options.pixel_size_m = 1e-6
    options.object_options.patch_interpolation_method = method
    options.object_options.remove_object_probe_ambiguity.enabled = False
    if n_slices > 1:
        options.object_options.slice_spacings_m = [1e-4] * (n_slices - 1)
    options.probe_options.optimizable = optimize_probe
    options.probe_options.orthogonalize_incoherent_modes.enabled = False
    options.probe_position_options.optimizable = optimize_positions
    torch.manual_seed(914)
    obj = torch.ones((n_slices, *object_shape), dtype=torch.complex128)
    probe = (1 + 0.1 * torch.randn((1 + int(opr), n_modes, *probe_shape))).to(torch.complex128)
    positions = torch.tensor([[-0.31, 0.24], [0.42, -0.16], [0.17, 0.36]])
    kwargs = {}
    if opr:
        options.probe_options.orthogonalize_opr_modes.enabled = False
        options.opr_mode_weight_options.optimizable = optimize_opr
        kwargs["opr_mode_weights_data"] = torch.tensor([[1., 0.1], [1., -0.1], [1., 0.2]])
    return PtychographyTask(
        options, diffraction_data=torch.ones((3, *probe_shape)), object_data=obj,
        probe_data=probe, probe_position_y_px=positions[:, 0],
        probe_position_x_px=positions[:, 1], **kwargs,
    )


def test_selection_overrides_and_serialization():
    options = api.EPIEOptions()
    assert options.reconstructor_options.forward_model_options.shift_operator == api.ShiftOperators.AUTO
    json.dumps(options.get_dict())
    with pytest.raises(ValueError):
        options.reconstructor_options.forward_model_options.shift_operator = "invalid"
    for distance, obj, probe, expected in [
        (float("inf"), (10, 10), (12, 12), "patch"),
        (0.1, (10, 10), (8, 9), "patch"),
        (0.1, (10, 10), (9, 8), "patch"),
        (0.1, (10, 10), (9, 9), "full_object"),
        (0.1, (10, 10), (12, 12), "full_object"),
    ]:
        assert ip.resolve_shift_operator("auto", distance, obj, probe) == expected
        for override in ("patch", "full_object"):
            assert ip.resolve_shift_operator(override, distance, obj, probe) == override
    task = make_task()
    model = task.reconstructor.forward_model
    assert model.resolved_shift_operator == api.ShiftOperators.FULL_OBJECT
    task.probe.options.pixel_size_m = 0.5e-6
    assert model.resolved_shift_operator == api.ShiftOperators.PATCH
    model.shift_operator = api.ShiftOperators.FULL_OBJECT
    assert model.resolved_shift_operator == api.ShiftOperators.FULL_OBJECT
    assert model.shift_operator == api.ShiftOperators.FULL_OBJECT


@pytest.mark.parametrize("origin", list(api.ObjectPosOriginCoordsMethods))
@pytest.mark.parametrize("method", ["fourier", "bilinear", "nearest"])
def test_model_adjoint_and_identity_probe_shift(origin, method):
    options = api.EPIEOptions()
    options.object_options.determine_position_origin_coords_by = origin
    options.object_options.position_origin_coords = (3.2, 4.1)
    task = make_task(options, method=method)
    model = task.reconstructor.forward_model
    indices = torch.arange(3)
    image = torch.randn_like(task.object.data, requires_grad=True)
    patches = model.extract_probe_grid_patches(image[0], task.probe_positions.tensor)
    cotangent = torch.randn_like(patches)
    expected = torch.autograd.grad(patches, image, cotangent)[0][0]
    actual = model.place_object_patches_on_probe_grid(task.probe_positions.tensor, cotangent)
    torch.testing.assert_close(actual, expected)
    probe = model.get_unique_probes(indices)
    assert model.shift_unique_probes(indices, probe) is probe
    assert task.reconstructor.adjoint_shift_probe_update_direction(indices, probe) is probe
    torch.testing.assert_close(model.extract_object_patches(indices)[:, 0],
                               model.extract_probe_grid_patches(task.object.data[0], task.probe_positions.tensor))


@pytest.mark.parametrize("factory", [
    api.EPIEOptions, api.RPIEOptions, api.LSQMLOptions,
    api.DMOptions, api.RAAROptions, api.BHOptions, api.AutodiffPtychographyOptions,
])
@pytest.mark.parametrize("method", ["fourier", "bilinear", "nearest"])
def test_reconstructors_reduce_synthetic_error(factory, method):
    options = factory()
    options.object_options.step_size = 0.1
    task = make_task(options, probe_shape=(9, 10), method=method)
    model = task.reconstructor.forward_model
    indices = torch.arange(3)
    initial = task.object.data.clone()
    with torch.no_grad():
        task.probe.set_data(torch.ones_like(task.probe.data))
        truth = (1 + 0.04 * torch.randn_like(initial.real)) * torch.exp(0.05j * torch.randn_like(initial.real))
        task.object.set_data(truth)
        data = model(indices).detach()
        task.reconstructor.dataset.patterns.copy_(data)
        task.object.set_data(initial)
        before = (model(indices).sqrt() - data.sqrt()).square().mean()
    task.run()
    with torch.no_grad():
        after = (model(indices).sqrt() - data.sqrt()).square().mean()
    assert torch.isfinite(task.object.data).all()
    assert after < before


@pytest.mark.parametrize("n_modes,n_slices,opr", [(2, 1, False), (1, 2, False), (1, 1, True)])
def test_modes_slices_opr_and_low_memory(n_modes, n_slices, opr):
    task = make_task(n_modes=n_modes, n_slices=n_slices, opr=opr)
    model = task.reconstructor.forward_model
    indices = torch.arange(3)
    full = model(indices)
    model.low_mem_mode = True
    low = model(indices)
    torch.testing.assert_close(low, full)


def test_full_object_path_does_not_construct_patch_indices(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Full-object sampling constructed patch indices")
    monkeypatch.setattr(ip, "_patch_linear_indices", fail)
    monkeypatch.setattr(ip, "batch_slice", fail)
    monkeypatch.setattr(ip, "batch_put", fail)
    calls = []
    fft = ip.pmath.fft2_precise
    monkeypatch.setattr(ip.pmath, "fft2_precise", lambda x: (calls.append(x.shape), fft(x))[1])
    image = torch.randn(7, 8, dtype=torch.complex128)
    positions = torch.rand(3, 2)
    patches = ip.extract_patches_full_object(image, positions, (6, 7))
    assert calls == [torch.Size([7, 8])]
    ip.place_patches_full_object(torch.zeros_like(image), positions, patches)


@pytest.mark.parametrize("method", ["fourier", "bilinear", "nearest"])
@pytest.mark.parametrize("resample", [False, True])
def test_bh_gradients_and_joint_curvature(method, resample):
    options = api.BHOptions()
    options.reconstructor_options.method = "CG"
    options.reconstructor_options.forward_model_options.shift_operator = api.ShiftOperators.FULL_OBJECT
    if resample:
        options.probe_options.pixel_size_m = 0.8e-6
        options.probe_options.pixel_size_aspect_ratio = 1.2
    task = make_task(options, method=method, n_modes=2, optimize_positions=True, optimize_probe=True)
    reconstructor = task.reconstructor
    reconstructor.eps = 0
    model = reconstructor.forward_model
    indices = torch.arange(3)
    with torch.no_grad():
        task.object.set_data(
            (1 + 0.1 * torch.randn_like(task.object.data.real))
            * torch.exp(0.1j * torch.randn_like(task.object.data.real))
        )
    data = reconstructor.dataset.patterns
    for epoch in range(2):
        reconstructor.current_epoch = epoch
        with torch.no_grad():
            updates, _ = reconstructor.compute_updates(indices, data)
        parameters = [task.object.tensor.data, task.probe.tensor.data, task.probe_positions.tensor]
        directions = [
            torch.view_as_real(reconstructor.eta_o)[None],
            torch.view_as_real(task.probe.options.rho * reconstructor.eta_p)[None],
            model.probe_displacements_to_object_pixels(
                task.probe_positions.options.rho * reconstructor.eta_pos
            ),
        ]
        prediction = model(indices)
        loss = (prediction.sqrt() - data.sqrt()).square().sum()
        gradients = torch.autograd.grad(loss, parameters, create_graph=True)
        slope = sum((g * v).sum() for g, v in zip(gradients, directions))
        products = torch.autograd.grad(slope, parameters)
        curvature = sum((h * v).sum() for h, v in zip(products, directions))
        alpha = -slope.detach() / curvature
        for index, (update, direction) in enumerate(zip(updates, directions)):
            expected = alpha * (torch.view_as_complex(direction) if index < 2 else direction)
            torch.testing.assert_close(update, expected, rtol=1e-8, atol=1e-9)
        with torch.no_grad():
            reconstructor.apply_updates(*updates)


@pytest.mark.parametrize("factory", [
    api.EPIEOptions, api.LSQMLOptions, api.DMOptions, api.RAAROptions,
    api.AutodiffPtychographyOptions,
])
def test_position_refinement_runs(factory):
    task = make_task(factory(), optimize_positions=True)
    with torch.no_grad():
        task.object.set_data(task.object.data * torch.exp(0.1j * torch.randn_like(task.object.data.real)))
    task.run()
    assert torch.isfinite(task.probe_positions.tensor).all()


@pytest.mark.parametrize("factory", [api.EPIEOptions, api.LSQMLOptions, api.AutodiffPtychographyOptions])
@pytest.mark.parametrize("n_modes,n_slices,opr", [(2, 1, False), (1, 2, False), (1, 1, True)])
def test_reconstruction_with_modes_slices_and_opr(factory, n_modes, n_slices, opr):
    task = make_task(factory(), n_modes=n_modes, n_slices=n_slices, opr=opr)
    task.run()
    assert torch.isfinite(task.object.data).all()


@pytest.mark.parametrize("factory", [api.EPIEOptions, api.LSQMLOptions, api.DMOptions, api.RAAROptions, api.BHOptions])
def test_probe_larger_than_object(factory):
    task = make_task(factory(), object_shape=(7, 8), probe_shape=(9, 10))
    indices = torch.arange(3)
    with torch.no_grad():
        # Unilluminated probe margins have exactly zero modeled intensity before propagation.
        task.reconstructor.dataset.patterns.copy_(task.reconstructor.forward_model(indices) * 1.03)
    task.run()
    assert torch.isfinite(task.object.data).all()


def test_mpie_uses_full_object_sampling():
    options = api.RPIEOptions()
    options.object_options.optimizer_params = {"momentum": 0.1, "nesterov": True}
    options.object_options.step_size = 0.1
    task = make_task(options)
    task.run()
    assert torch.isfinite(task.object.data).all()


@pytest.mark.parametrize("factory", [
    api.EPIEOptions, api.LSQMLOptions, api.RAAROptions, api.AutodiffPtychographyOptions,
])
def test_probe_and_opr_optimization(factory):
    options = factory()
    options.object_options.step_size = 0.1
    options.probe_options.step_size = 0.1
    task = make_task(options, n_modes=2, opr=True, optimize_probe=True, optimize_opr=True)
    before = task.probe.data.clone()
    task.run()
    assert torch.isfinite(task.probe.data).all()
    assert torch.isfinite(task.opr_mode_weights.data).all()
    assert not torch.equal(task.probe.data, before)


def test_fixed_resize_parity_uses_absolute_patch_centers():
    for source, target in [((7, 8), (4, 5)), ((6, 7), (5, 4))]:
        image = torch.arange(source[0] * source[1], dtype=torch.float64).reshape(source)
        start = torch.tensor([1., 2.])
        position = start + (torch.tensor(target) - 1) / 2
        expected = image[1:1 + target[0], 2:2 + target[1]]
        actual = ip.extract_patches_full_object(image, position[None], target)[0]
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("method", ["fourier", "bilinear", "nearest"])
@pytest.mark.parametrize("resample", [False, True])
@pytest.mark.parametrize("use_all_modes", [False, True])
def test_illumination_map_matches_selected_adjoint(method, resample, use_all_modes):
    options = api.EPIEOptions()
    options.reconstructor_options.forward_model_options.shift_operator = api.ShiftOperators.FULL_OBJECT
    if resample:
        options.probe_options.pixel_size_m = 0.8e-6
        options.probe_options.pixel_size_aspect_ratio = 1.2
    task = make_task(options, n_modes=2, method=method)
    model = task.reconstructor.forward_model
    # Cross the accumulation chunk boundary, including a partial last chunk.
    positions = task.probe_positions.tensor.repeat(25, 1)
    intensity = (
        task.probe.get_all_mode_intensity(opr_mode=0) if use_all_modes
        else task.probe.get_mode_and_opr_mode(mode=0, opr_mode=0).abs().square()
    )
    expected = model.place_object_patches_on_probe_grid(
        positions, intensity, integer_mode=False,
    )
    expected = model.object_update_to_native_grid(expected, normalized=True).real.clamp_min(0)
    actual = task.object.calculate_illumination_map(
        task.probe, task.probe_positions, use_all_modes=use_all_modes,
        sampler=model.get_object_patch_sampler(),
        sampling_positions=model.get_object_patch_sampling_positions().repeat(25, 1),
        sampling_shape=model.get_object_patch_sampling_shape(),
    )
    torch.testing.assert_close(actual, expected)
    assert (actual >= 0).all()
    assert not actual.is_complex()


def test_standalone_illumination_map_keeps_integer_placement():
    task = make_task()
    intensity = task.probe.get_mode_and_opr_mode(mode=0, opr_mode=0).abs().square()
    expected = ip.place_patches_integer(
        torch.zeros(task.object.lateral_shape),
        task.probe_positions.tensor.round().int() + task.object.pos_origin_coords,
        intensity[None].expand(3, -1, -1),
    )
    actual = task.object.calculate_illumination_map(task.probe, task.probe_positions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
