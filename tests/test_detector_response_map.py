import numpy as np
import pytest
import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.metrics import MSELossOfSqrt
from ptychi.utils import get_default_complex_dtype, set_default_complex_dtype


@pytest.fixture(autouse=True)
def restore_defaults():
    dtype = torch.get_default_dtype()
    device = torch.get_default_device()
    complex_dtype = get_default_complex_dtype()
    yield
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    set_default_complex_dtype(complex_dtype)


def make_task(response=None, options_class=api.LSQMLOptions, fft_shift=False, blur=None,
              valid_pixel_mask=None, noise="gaussian"):
    options = options_class()
    options.reconstructor_options.default_device = api.Devices.CPU
    options.reconstructor_options.batch_size = 2
    if options_class is api.LSQMLOptions:
        options.reconstructor_options.noise_model = noise
    options.reconstructor_options.forward_model_options.diffraction_pattern_blur_sigma = blur
    options.data_options.fft_shift = fft_shift
    options.object_options.pixel_size_m = 1.0
    options.probe_position_options.optimizable = False
    rng = np.random.default_rng(123)
    return PtychographyTask(
        options,
        diffraction_data=np.ones((2, 3, 5), dtype=np.float32),
        object_data=np.ones((1, 12, 12), dtype=np.complex64),
        probe_data=(rng.random((1, 2, 6, 6)) + 1j * rng.random((1, 2, 6, 6))).astype(
            np.complex64
        ),
        probe_position_x_px=np.array([-1.0, 1.0]),
        probe_position_y_px=np.array([-1.0, 1.0]),
        detector_response_map=response,
        valid_pixel_mask=valid_pixel_mask,
    )


@pytest.mark.parametrize(
    "options_class", [api.LSQMLOptions, api.EPIEOptions, api.BHOptions, api.AutodiffPtychographyOptions]
)
@pytest.mark.parametrize("fft_shift", [False, True])
def test_response_after_detector_crop(options_class, fft_shift):
    gain = np.linspace(0.5, 2.0, 15, dtype=np.float32).reshape(3, 5)
    offset = torch.arange(15, dtype=torch.float32, device="cpu").reshape(3, 5)
    reference = make_task(options_class=options_class, fft_shift=fft_shift)
    corrected = make_task((gain, offset), options_class, fft_shift)
    expected_gain = torch.from_numpy(gain)
    expected_offset = offset
    if fft_shift:
        expected_gain = torch.fft.fftshift(expected_gain)
        expected_offset = torch.fft.fftshift(expected_offset)
    for actual, expected in zip(
        corrected.detector_response.data, (expected_gain, expected_offset)
    ):
        torch.testing.assert_close(actual, expected)
    indices = torch.tensor([0, 1])
    expected = reference.reconstructor.forward_model(indices) * expected_gain + expected_offset
    model = corrected.reconstructor.forward_model
    actual, patches = model(indices, return_object_patches=True)
    torch.testing.assert_close(actual, expected)
    if options_class in (api.LSQMLOptions, api.EPIEOptions):
        low_memory, low_memory_patches = model.forward_low_memory(indices, return_object_patches=True)
        torch.testing.assert_close(low_memory, expected)
        torch.testing.assert_close(low_memory_patches, patches)


def test_response_is_applied_after_blur():
    gain = np.linspace(0.5, 2.0, 15, dtype=np.float32).reshape(3, 5)
    offset = np.arange(15, dtype=np.float32).reshape(3, 5)
    reference = make_task(blur=0.2)
    corrected = make_task((gain, offset), blur=0.2)
    indices = torch.tensor([0, 1])
    expected = (
        reference.reconstructor.forward_model(indices) * torch.from_numpy(gain)
        + torch.from_numpy(offset)
    )
    torch.testing.assert_close(corrected.reconstructor.forward_model(indices), expected)
    torch.testing.assert_close(corrected.reconstructor.forward_model.forward_low_memory(indices), expected)


def test_identity_response_preserves_predictions():
    reference = make_task()
    identity = make_task((np.ones((3, 5)), np.zeros((3, 5))))
    assert reference.detector_response.is_identity
    indices = torch.tensor([0, 1])
    torch.testing.assert_close(
        identity.reconstructor.forward_model(indices),
        reference.reconstructor.forward_model(indices),
        rtol=0,
        atol=0,
    )


def test_response_preserves_autograd_chain_rule():
    gain = np.linspace(0.5, 2.0, 15, dtype=np.float32).reshape(3, 5)
    offset = np.ones((3, 5), dtype=np.float32)
    reference = make_task(options_class=api.AutodiffPtychographyOptions)
    corrected = make_task((gain, offset), api.AutodiffPtychographyOptions)
    indices = torch.tensor([0, 1])
    expected = reference.reconstructor.forward_model(indices) * torch.from_numpy(gain)
    expected_gradient = torch.autograd.grad(expected.sum(), reference.object.tensor.data)[0]
    actual = corrected.reconstructor.forward_model(indices)
    actual_gradient = torch.autograd.grad(actual.sum(), corrected.object.tensor.data)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize(
    "response",
    [
        [np.ones((3, 5)), np.zeros((3, 5))],
        (np.ones((3, 5)),),
        (np.ones((3, 5)), np.zeros((3, 5)), np.zeros((3, 5))),
        (1.0, np.zeros((3, 5))),
        (np.ones((3, 5)), 0.0),
        (np.ones((1, 3, 5)), np.zeros((3, 5))),
        (np.ones((3, 5)), np.zeros((5, 3))),
    ],
)
def test_invalid_detector_response_is_rejected(response):
    with pytest.raises(ValueError, match="detector_response_map"):
        make_task(response)


@pytest.mark.parametrize("component_index", [0, 1], ids=["gain", "offset"])
@pytest.mark.parametrize("as_tensor", [False, True], ids=["numpy", "tensor"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_response_is_rejected_even_at_masked_pixels(component_index, as_tensor, invalid):
    response = [np.ones((3, 5)), np.zeros((3, 5))]
    response[component_index][0, 0] = invalid
    if as_tensor:
        response = [torch.from_numpy(component) for component in response]
    mask = np.ones((3, 5), dtype=bool)
    mask[0, 0] = False
    name = ("gain", "offset")[component_index]
    with pytest.raises(ValueError, match=f"detector_response_map.*{name}.*finite"):
        make_task(tuple(response), valid_pixel_mask=mask)


@pytest.mark.parametrize("component_index", [0, 1], ids=["gain", "offset"])
@pytest.mark.parametrize("as_tensor", [False, True], ids=["numpy", "tensor"])
@pytest.mark.parametrize("imaginary", [0.0, 1.0])
def test_complex_response_is_rejected(component_index, as_tensor, imaginary):
    response = [np.ones((3, 5)), np.zeros((3, 5))]
    response[component_index] = response[component_index].astype(np.complex128) + imaginary * 1j
    if as_tensor:
        response = [torch.from_numpy(component) for component in response]
    name = ("gain", "offset")[component_index]
    with pytest.raises(ValueError, match=f"detector_response_map.*{name}.*real"):
        make_task(tuple(response))


@pytest.mark.parametrize("component_index", [0, 1], ids=["gain", "offset"])
def test_response_overflow_in_reconstruction_dtype_is_rejected(component_index):
    response = [np.ones((3, 5)), np.zeros((3, 5))]
    response[component_index][0, 0] = 1e100
    assert np.isfinite(response[component_index]).all()
    with pytest.raises(ValueError, match="detector_response_map.*finite"):
        make_task(tuple(response))


@pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64])
def test_unsigned_gain_is_accepted(dtype):
    gain = torch.ones((3, 5), dtype=dtype)
    task = make_task((gain, np.zeros((3, 5))))
    assert task.dataset.valid_pixel_mask.all()
    torch.testing.assert_close(task.detector_response.data[0], torch.ones((3, 5)))


@pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64])
def test_unsigned_zero_gain_is_rejected(dtype):
    gain = torch.ones((3, 5), dtype=dtype)
    gain[0, 0] = 0
    mask = np.ones((3, 5), dtype=bool)
    mask[0, 0] = False
    with pytest.raises(ValueError, match="detector_response_map.*gain.*strictly positive"):
        make_task((gain, np.zeros((3, 5))), valid_pixel_mask=mask)


@pytest.mark.parametrize("as_tensor", [False, True], ids=["numpy", "tensor"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("invalid", [-1.0, -1e-100, 0.0, 1e-100])
def test_gain_must_remain_positive_in_reconstruction_dtype(as_tensor, masked, invalid):
    gain = np.ones((3, 5), dtype=np.float64)
    gain[0, 0] = invalid
    if as_tensor:
        gain = torch.from_numpy(gain)
    mask = np.ones((3, 5), dtype=bool)
    mask[0, 0] = not masked
    # Values that round to zero in the reconstruction dtype are also invalid.
    with pytest.raises(ValueError, match="detector_response_map.*gain.*strictly positive"):
        make_task((gain, np.zeros((3, 5))), valid_pixel_mask=mask)


@pytest.mark.parametrize("fft_shift", [False, True])
@pytest.mark.parametrize("as_tensor", [False, True], ids=["numpy", "tensor"])
def test_response_preserves_user_mask(fft_shift, as_tensor):
    gain = np.ones((3, 5), dtype=np.float32)
    gain[0, :3] = [0.5, 2, 1e-8]
    mask = np.ones((3, 5), dtype=bool)
    mask[1, 3] = False
    expected = torch.from_numpy(mask)
    if fft_shift:
        expected = torch.fft.fftshift(expected)
    response = (gain, np.zeros((3, 5), dtype=np.float32))
    if as_tensor:
        response = tuple(torch.from_numpy(component) for component in response)
        mask = torch.from_numpy(mask)
    original_mask = np.asarray(mask).copy()
    task = make_task(response, fft_shift=fft_shift, valid_pixel_mask=mask)
    torch.testing.assert_close(task.dataset.valid_pixel_mask.cpu(), expected)
    torch.testing.assert_close(task.reconstructor.noise_model.valid_pixel_mask.cpu(), expected)
    np.testing.assert_array_equal(np.asarray(mask), original_mask)


@pytest.mark.parametrize(
    "options_class,noise",
    [(api.LSQMLOptions, "gaussian"), (api.LSQMLOptions, "poisson"),
     (api.AutodiffPtychographyOptions, "gaussian"), (api.EPIEOptions, "gaussian"),
     (api.BHOptions, "gaussian")],
)
def test_masked_measurements_do_not_affect_reconstruction_with_response(options_class, noise):
    gain = np.ones((3, 5), dtype=np.float32)
    offset = np.zeros((3, 5), dtype=np.float32)
    mask = torch.ones((3, 5), dtype=torch.bool, device="cpu")
    mask[0, :3] = False
    offset[0, 2] = 4  # An excluded pixel may have a nonzero background.
    reference = make_task((gain, offset), options_class, noise=noise, valid_pixel_mask=mask)
    changed = make_task((gain, offset), options_class, noise=noise, valid_pixel_mask=mask)
    with torch.no_grad():
        prediction = reference.reconstructor.forward_model(torch.arange(2))
        scale = torch.linspace(0.7, 1.2, 15).reshape(3, 5)
        data = scale * (prediction - torch.from_numpy(offset)) + torch.from_numpy(offset)
        reference.dataset.patterns.copy_(torch.where(mask, data, 0))
        changed.dataset.patterns.copy_(torch.where(mask, data, 1e6))
        initial_object = reference.get_data("object").clone()
    for task in (reference, changed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(192)
            task.run(n_epochs=2)  # Includes LSQML's epoch-0 probe rescaling.
        assert np.isfinite(task.reconstructor.loss_tracker.table["loss"].to_numpy()).all()
        for name in ("object", "probe"):
            assert torch.isfinite(task.get_data(name)).all()
        if options_class is api.LSQMLOptions:
            assert np.isfinite(task.reconstructor.reconstructor_buffers.fourier_errors).all()
    assert not torch.allclose(reference.get_data("object"), initial_object)
    for name in ("object", "probe"):
        torch.testing.assert_close(changed.get_data(name), reference.get_data(name), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        changed.reconstructor.loss_tracker.table["loss"].to_numpy(),
        reference.reconstructor.loss_tracker.table["loss"].to_numpy(), rtol=1e-6, atol=1e-6,
    )


@pytest.mark.parametrize("all_excluded", [False, True])
def test_reporting_masks_pixels_without_response_map(all_excluded):
    mask = np.full((3, 5), not all_excluded, dtype=bool)
    mask[0, 0] = False
    task = make_task(valid_pixel_mask=mask)
    reconstructor = task.reconstructor
    reconstructor.options.exclude_measured_pixels_below = 0.0
    reconstructor.loss_tracker.metric_function = MSELossOfSqrt()
    y_pred = torch.full((2, 3, 5), 4.0)
    y_true = torch.ones_like(y_pred)
    y_pred[:, 0, 0] = -4.0
    y_true[:, 0, 0] = -1.0  # Excluded by the shared detector mask.
    y_true[:, 0, 1] = 0.0  # Excluded by the measurement threshold.

    reconstructor.update_batch_loss(y_pred, y_true)
    reconstructor.update_fourier_error(y_pred, y_true)

    assert reconstructor.loss_tracker.epoch_loss == pytest.approx(0.0 if all_excluded else 1.0)
    assert reconstructor.reconstructor_buffers.fourier_errors[-1] == pytest.approx(
        0.0 if all_excluded else 13 / 15
    )


def test_preconditioner_masks_pixels_without_response_map():
    mask = np.ones((3, 5), dtype=bool)
    mask[0, 0] = False
    reference = make_task(valid_pixel_mask=mask)
    changed = make_task(valid_pixel_mask=mask)
    reference.dataset.patterns[:, 0, 0] = 0.0
    changed.dataset.patterns[:, 0, 0] = 1e6
    with torch.no_grad():
        reference.reconstructor.update_preconditioners()
        changed.reconstructor.update_preconditioners()
    torch.testing.assert_close(changed.object.preconditioner, reference.object.preconditioner)
