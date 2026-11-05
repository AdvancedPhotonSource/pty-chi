import math

import pytest
import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_default_complex_dtype, set_default_complex_dtype

POSITIONS = torch.tensor(
    [[-1.3, 0.4], [0.27, -2.6], [1.5, 1.5], [-2.2, -1.7], [2.6, -0.8]], dtype=torch.float64
)


@pytest.fixture(autouse=True)
def restore_defaults():
    dtype, device = torch.get_default_dtype(), torch.get_default_device()
    complex_dtype = get_default_complex_dtype()
    torch.set_default_device("cpu")
    yield
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    set_default_complex_dtype(complex_dtype)


def make_task(probe, options=None, method="bilinear", data=None, obj=None):
    options = options or api.EPIEOptions()
    options.reconstructor_options.default_device = api.Devices.CPU
    options.reconstructor_options.default_dtype = api.Dtypes.FLOAT64
    options.reconstructor_options.forward_model_options.subpixel_shift_all_probe_modes = True
    options.data_options.wavelength_m = 1e-10
    options.object_options.pixel_size_m = 1e-8
    options.object_options.patch_interpolation_method = method
    options.object_options.remove_object_probe_ambiguity.enabled = False
    options.probe_options.orthogonalize_incoherent_modes.enabled = False
    if obj is None:
        generator = torch.Generator().manual_seed(915)
        obj = torch.polar(
            1 + torch.rand((1, 24, 24), dtype=torch.float64, generator=generator),
            torch.rand((1, 24, 24), dtype=torch.float64, generator=generator),
        )
    if data is None:
        data = torch.ones((len(POSITIONS), *probe.shape[-2:]), dtype=torch.float64)
    return PtychographyTask(
        options, diffraction_data=data, object_data=obj, probe_data=probe,
        probe_position_y_px=POSITIONS[:, 0], probe_position_x_px=POSITIONS[:, 1],
    )


def random_probe():
    generator = torch.Generator().manual_seed(916)
    return torch.randn((1, 1, 8, 8), dtype=torch.complex128, generator=generator)


def split_into_two_modes(probe):
    """Two identical modes of half power describe the same illumination as `probe`."""
    return torch.cat([probe, probe], dim=1) / math.sqrt(2)


@pytest.mark.parametrize("method", ["bilinear", "fourier", "cubic"])
def test_split_probe_modes_predict_single_mode_intensities(method):
    indices = torch.arange(len(POSITIONS))
    probe = random_probe()
    expected = make_task(probe, method=method).reconstructor.forward_model.forward(indices)
    actual = make_task(split_into_two_modes(probe), method=method).reconstructor.forward_model.forward(indices)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "options_class", [api.EPIEOptions, api.DMOptions, api.RAAROptions, api.LSQMLOptions]
)
def test_split_probe_modes_reconstruct_like_single_mode(options_class):
    """Probe update directions of every mode must get the adjoint of the forward shift."""
    probe = random_probe()
    data = make_task(probe).reconstructor.forward_model.forward(torch.arange(len(POSITIONS)))
    results = []
    for initial_probe in (0.8 * probe, split_into_two_modes(0.8 * probe)):
        options = options_class()
        options.reconstructor_options.num_epochs = 3
        options.reconstructor_options.batch_size = 2
        options.object_options.optimizable = True
        options.probe_options.optimizable = True
        task = make_task(
            initial_probe, options, data=data.detach(),
            obj=torch.ones((1, 24, 24), dtype=torch.complex128),
        )
        task.run()
        results.append(torch.cat([
            task.get_data_to_cpu("object").flatten(),
            task.get_data_to_cpu("probe")[0].sum(0).flatten() / math.sqrt(task.probe.n_modes),
        ]))
    torch.testing.assert_close(results[1], results[0], rtol=1e-5, atol=1e-5)
