import torch

import ptychi.api as api
import ptychi.maps as maps
from ptychi.api.options.raar import RAARReconstructorOptions
from ptychi.reconstructors import RAARReconstructor


def test_raar_options_select_raar_reconstructor():
    options = api.RAAROptions()

    assert options.reconstructor_options.get_reconstructor_type() is api.Reconstructors.RAAR
    assert maps.get_reconstructor_by_enum(api.Reconstructors.RAAR) is RAARReconstructor
    assert options.reconstructor_options.beta == 0.75


def test_raar_exit_wave_update_matches_paper_equation():
    reconstructor = object.__new__(RAARReconstructor)
    reconstructor.options = RAARReconstructorOptions(beta=0.75)
    reconstructor.psi = torch.tensor([1 + 1j, 2 - 1j], dtype=torch.complex64).reshape(2, 1, 1, 1)
    previous_exit_wave = reconstructor.psi.clone()
    pa_exit_wave = previous_exit_wave + (0.5 - 0.25j)
    q_exit_wave = previous_exit_wave * (0.8 + 0.1j)
    q_pa_exit_wave = pa_exit_wave * (0.9 - 0.2j)
    reconstructor.apply_data_projection = lambda exit_wave, y_true: pa_exit_wave
    reconstructor.synthesize_exit_wave_chunk = (
        lambda projected_object, start_pt, end_pt: projected_object[start_pt:end_pt]
    )

    error_squared, exit_wave_update = reconstructor.apply_raar_exit_wave_update(
        y_true=torch.zeros((2, 1, 1)),
        q_object=q_exit_wave,
        pa_object=q_pa_exit_wave,
        start_pts=[0],
        end_pts=[2],
        return_exit_wave_update=True,
    )

    beta = reconstructor.options.beta
    expected = (
        2 * beta * q_pa_exit_wave
        + (1 - 2 * beta) * pa_exit_wave
        + beta * (q_exit_wave - previous_exit_wave)
    )
    assert torch.allclose(reconstructor.psi, expected)
    assert torch.allclose(error_squared, ((expected - previous_exit_wave).abs() ** 2).sum())
    assert torch.allclose(exit_wave_update, expected - previous_exit_wave)
