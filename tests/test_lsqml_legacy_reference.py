"""Small frozen LSQML references from 88bbdc7 (PyTorch 2.12.1, float64)."""

import numpy as np
import pytest
import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_default_complex_dtype, set_default_complex_dtype


# Generated with the original Gaussian/Poisson gradient and reciprocal-step
# routines, using the fixed 2x2 inputs below and the default numerical epsilons.
REFERENCE = {
    "gaussian": {
        "gradient": [[7.0000000004444445, 0], [3, 0.7888543803998316 + 1.5777087607996632j]],
        "psi_opt": [
            [-0.3486067976055345 + 0.3027864049000421j, 0.8486067974944234 - 0.3027864049000421j],
            [0.09860679749442339 + 1.1972135950999578j, 0.4013932023944655 - 1.1972135950999578j],
        ],
        "alpha": 0.5,
        "cached_alpha": 0.6,
    },
    "poisson": {
        "gradient": [[4.277777788751714, 0], [1.5, 1.0999987200010237 + 2.1999974400020474j]],
        "psi_opt": [
            [-0.23003724205223824 + 0.10654743867028604j, 1.1634153192774757 - 0.10654743867028604j],
            [0.1999421635941871 + 1.393452561329714j, 0.30648960226447314 - 1.393452561329714j],
        ],
        "alpha": 0.7153691257556152,
        "cached_alpha": 0.7153691257556152,
    },
}


@pytest.fixture(autouse=True)
def restore_defaults():
    original = torch.get_default_dtype(), torch.get_default_device(), get_default_complex_dtype()
    yield
    torch.set_default_dtype(original[0])
    torch.set_default_device(original[1])
    set_default_complex_dtype(original[2])


@pytest.mark.parametrize("noise", ["gaussian", "poisson"])
def test_reciprocal_step_matches_reference(noise):
    options = api.LSQMLOptions()
    options.reconstructor_options.default_device = api.Devices.CPU
    options.reconstructor_options.default_dtype = api.Dtypes.FLOAT64
    options.reconstructor_options.batch_size = 1
    options.reconstructor_options.noise_model = noise
    options.reconstructor_options.rescale_probe_intensity_in_first_epoch = False
    options.data_options.fft_shift = False
    options.object_options.pixel_size_m = 1.0
    options.probe_position_options.optimizable = False
    task = PtychographyTask(
        options,
        diffraction_data=np.array([[[1.0, 2.0], [0.0, 4.0]]]),
        object_data=np.ones((1, 4, 4), dtype=np.complex128),
        probe_data=np.array([[[[1 + 0.5j, 2 - 0.5j], [0.5 + 1j, 1 - 1j]]]]),
        probe_position_x_px=np.array([0.0]),
        probe_position_y_px=np.array([0.0]),
        valid_pixel_mask=np.array([[True, False], [True, True]]),
    )
    rec, model = task.reconstructor, task.reconstructor.forward_model
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(612)
        indices = torch.tensor([0])
        rec.alpha_psi_far_all_pos.fill_(0.6)
        predicted = model(indices)
        measured = task.dataset.patterns
        wave = model.intermediate_variables.psi_far
        gradient = rec.noise_model.backward_to_psi_far(predicted, measured, wave, forward_model=model)
        psi_opt = rec.run_reciprocal_space_step(predicted, measured, indices)
    expected = REFERENCE[noise]
    for name, value in (("gradient", gradient), ("psi_opt", psi_opt),
                        ("alpha", rec.alpha_psi_far), ("cached_alpha", rec.alpha_psi_far_all_pos)):
        torch.testing.assert_close(value.squeeze(), torch.tensor(expected[name], dtype=value.dtype),
                                   rtol=1e-12, atol=1e-12)
