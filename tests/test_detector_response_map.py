import numpy as np
import pytest
import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask
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


def make_task(response=None, options_class=api.LSQMLOptions, fft_shift=False, blur=None):
    options = options_class()
    options.reconstructor_options.default_device = api.Devices.CPU
    options.reconstructor_options.batch_size = 2
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
    )


@pytest.mark.parametrize(
    "options_class", [api.LSQMLOptions, api.EPIEOptions, api.BHOptions, api.AutodiffPtychographyOptions]
)
@pytest.mark.parametrize("fft_shift", [False, True])
def test_response_after_detector_crop(options_class, fft_shift):
    multiplier = np.linspace(0.5, 2.0, 15, dtype=np.float32).reshape(3, 5)
    offset = torch.arange(15, dtype=torch.float32, device="cpu").reshape(3, 5)
    reference = make_task(options_class=options_class, fft_shift=fft_shift)
    corrected = make_task((multiplier, offset), options_class, fft_shift)
    expected_multiplier = torch.from_numpy(multiplier)
    expected_offset = offset
    if fft_shift:
        expected_multiplier = torch.fft.fftshift(expected_multiplier)
        expected_offset = torch.fft.fftshift(expected_offset)
    for actual, expected in zip(
        corrected.dataset.detector_response_map, (expected_multiplier, expected_offset)
    ):
        torch.testing.assert_close(actual, expected)
    indices = torch.tensor([0, 1])
    expected = reference.reconstructor.forward_model(indices) * expected_multiplier + expected_offset
    model = corrected.reconstructor.forward_model
    actual, patches = model(indices, return_object_patches=True)
    torch.testing.assert_close(actual, expected)
    if options_class in (api.LSQMLOptions, api.EPIEOptions):
        low_memory, low_memory_patches = model.forward_low_memory(indices, return_object_patches=True)
        torch.testing.assert_close(low_memory, expected)
        torch.testing.assert_close(low_memory_patches, patches)


def test_response_is_applied_after_blur():
    multiplier = np.linspace(0.5, 2.0, 15, dtype=np.float32).reshape(3, 5)
    offset = np.arange(15, dtype=np.float32).reshape(3, 5)
    reference = make_task(blur=0.2)
    corrected = make_task((multiplier, offset), blur=0.2)
    indices = torch.tensor([0, 1])
    expected = (
        reference.reconstructor.forward_model(indices) * torch.from_numpy(multiplier)
        + torch.from_numpy(offset)
    )
    torch.testing.assert_close(corrected.reconstructor.forward_model(indices), expected)
    torch.testing.assert_close(corrected.reconstructor.forward_model.forward_low_memory(indices), expected)


def test_identity_response_preserves_predictions():
    reference = make_task()
    identity = make_task((np.ones((3, 5)), np.zeros((3, 5))))
    assert reference.dataset.detector_response_map is None
    indices = torch.tensor([0, 1])
    torch.testing.assert_close(
        identity.reconstructor.forward_model(indices),
        reference.reconstructor.forward_model(indices),
        rtol=0,
        atol=0,
    )


def test_response_preserves_autograd_chain_rule():
    multiplier = np.linspace(0.5, 2.0, 15, dtype=np.float32).reshape(3, 5)
    offset = np.ones((3, 5), dtype=np.float32)
    reference = make_task(options_class=api.AutodiffPtychographyOptions)
    corrected = make_task((multiplier, offset), api.AutodiffPtychographyOptions)
    indices = torch.tensor([0, 1])
    expected = reference.reconstructor.forward_model(indices) * torch.from_numpy(multiplier)
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
