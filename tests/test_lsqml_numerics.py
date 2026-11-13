"""LSQML update equations on inline complex arrays, without reconstruction epochs."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import ptychi.api as api
from ptychi import forward_models as fm
from ptychi.data_structures.object import PlanarObject
from ptychi.data_structures.probe import Probe
from ptychi.reconstructors.lsqml import LSQMLReconstructor, MultiprocessLSQMLReconstructor
from test_utils import make_task

pytestmark = pytest.mark.usefixtures("numerical_cpu")


@pytest.fixture
def reconstructor():
    recon = LSQMLReconstructor.__new__(LSQMLReconstructor)
    options = api.LSQMLOptions()
    options.object_options.multimodal_update = False
    options.object_options.optimal_step_size_scaler = 1.0
    options.probe_options.optimal_step_size_scaler = 1.0
    recon.options = options.reconstructor_options
    recon.dataset = SimpleNamespace(valid_pixel_mask=None)
    recon.parameter_group = SimpleNamespace(
        object=PlanarObject(
            data=torch.ones((1, 2, 2), dtype=torch.complex128), options=options.object_options
        ),
        probe=Probe(
            data=torch.ones((1, 2, 2, 2), dtype=torch.complex128), options=options.probe_options
        ),
    )
    recon.current_epoch = 1
    recon.current_minibatch = 0
    return recon


@pytest.mark.parametrize(
    "mode,expected",
    [
        (None, [[[5 - 1j, 6 + 2j]]]),
        (0, [[[2 + 0j, 2 + 2j]]]),
        (1, [[[3 - 1j, 4 + 0j]]]),
    ],
)
def test_object_direction_conjugates_probe_and_sums_modes(reconstructor, mode, expected):
    chi = torch.tensor([[[[1 + 1j, 2j]], [[1 + 3j, 4]]]])
    probe = torch.tensor([[[[1 + 1j, 1 + 1j]], [[1j, 1]]]])
    actual = reconstructor._calculate_object_patch_update_direction(chi, probe, mode)
    torch.testing.assert_close(actual, torch.tensor(expected)[:, None])


def test_probe_direction_uses_selected_slice_and_averages_positions(reconstructor):
    chi = torch.tensor([[[[2 + 2j, 4]]], [[[6j, 8]]]])
    patches = torch.tensor([[[[99, 99]], [[1j, 2]]], [[[99, 99]], [[2j, 4]]]])
    updates = reconstructor._calculate_probe_update_direction(chi, patches, slice_index=1)
    torch.testing.assert_close(updates, torch.tensor([[[[2 - 2j, 8]]], [[[12, 32]]]]))
    torch.testing.assert_close(
        reconstructor._precondition_probe_update_direction(updates), torch.tensor([[[7 - 1j, 20]]])
    )


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("mode", [None, 0])
def test_separate_least_squares_steps(reconstructor, sign, mode):
    # Real inner products: <[1,i], [4,4i]> = 8, norm squared = 2.
    direction = torch.tensor([[[[1, 1j]]]])
    probe = torch.ones((1, 2, 1, 2), dtype=torch.complex128)
    chi = sign * torch.tensor([[[[4, 4j]], [[8, 8j]]]])
    alpha_o = reconstructor.calculate_object_update_step_sizes(chi, direction, probe, mode)
    alpha_p = reconstructor.calculate_probe_update_step_sizes(
        chi, torch.ones_like(direction), direction[0].repeat(2, 1, 1), mode
    )
    expected = 0.0 if sign < 0 else (3.0 if mode is None else 2.0)
    torch.testing.assert_close(alpha_o, torch.tensor([expected]))
    torch.testing.assert_close(alpha_p, torch.tensor([expected]))


def test_object_step_scales_for_multislice_and_multimode(reconstructor):
    opts = api.LSQMLOptions().object_options
    opts.slice_spacings_m = [1e-5]
    opts.multimodal_update = True
    opts.optimal_step_size_scaler = 3.0
    reconstructor.parameter_group.object = PlanarObject(
        data=torch.ones((2, 2, 2), dtype=torch.complex128), options=opts
    )
    alpha = reconstructor.calculate_object_update_step_sizes(
        torch.full((1, 2, 1, 2), 8 + 0j), torch.ones((1, 1, 1, 2)), torch.ones((1, 2, 1, 2))
    )
    torch.testing.assert_close(alpha, torch.tensor([3.0]))  # (8/2) * 3 / 2 / 2


@pytest.mark.parametrize("sign", [1, -1])
def test_joint_least_squares_solves_coupled_system(reconstructor, sign):
    # Directions [1,0] and [1,1] give A=[[1.1,1],[1,2.2]], b=[3,5].
    # lambda_0 adds 1.32e-7 to each diagonal; solve by hand using the determinant.
    ao, ap = reconstructor.calculate_object_and_probe_update_step_sizes(
        sign * torch.tensor([[[[3 + 0j, 2]]]]),
        torch.ones((1, 1, 1, 2), dtype=torch.complex128),
        torch.tensor([[[[1 + 0j, 0]]]]),
        torch.tensor([[[1 + 0j, 1]]]),
        probe=torch.ones((1, 1, 1, 2), dtype=torch.complex128),
    )
    determinant = (1.1 + 1.32e-7) * (2.2 + 1.32e-7) - 1
    expected_o = (3 * (2.2 + 1.32e-7) - 5) / determinant if sign > 0 else 0.0
    expected_p = (5 * (1.1 + 1.32e-7) - 3) / determinant if sign > 0 else 0.0
    torch.testing.assert_close(ao, torch.tensor([expected_o]))
    torch.testing.assert_close(ap, torch.tensor([expected_p]))


@pytest.mark.parametrize("batching", list(api.BatchingModes))
@pytest.mark.parametrize("resampling", [False, True])
def test_gradient_reset_respects_compact_minibatches(reconstructor, batching, resampling):
    reconstructor.options.batching_mode = batching
    reconstructor.forward_model = SimpleNamespace(
        resampling_enabled=resampling,
        probe_grid_object_shape=(2, 2),
        object_update_to_native_grid=lambda x: x,
    )
    reconstructor._initialize_object_gradient()
    reconstructor._record_object_slice_gradient(0, torch.full((1, 2, 2), 3 + 2j))
    reconstructor.current_minibatch = 1
    reconstructor._initialize_object_gradient()
    actual = (
        reconstructor._object_gradient_probe_grid
        if resampling
        else -reconstructor.parameter_group.object.get_grad()
    )
    expected = 3 + 2j if batching == api.BatchingModes.COMPACT else 0j
    torch.testing.assert_close(actual, torch.full((1, 2, 2), expected))
    reconstructor.current_minibatch = 0
    reconstructor._initialize_object_gradient()
    actual = (
        reconstructor._object_gradient_probe_grid
        if resampling
        else reconstructor.parameter_group.object.get_grad()
    )
    torch.testing.assert_close(actual, torch.zeros_like(actual))


@pytest.mark.parametrize("n_slices", [1, 2])
def test_compact_accumulation_overlapping_patches_and_epoch_preconditioning(n_slices):
    options = api.LSQMLOptions()
    options.reconstructor_options.batching_mode = api.BatchingModes.COMPACT
    recon = make_task(options, n_slices=n_slices).reconstructor
    recon.current_minibatch = 0
    recon._initialize_object_gradient()
    fm = recon.forward_model
    # Centers in (y,x); 2x2 patches overlap in their middle column.
    positions = torch.tensor([[0.0, -1.0], [0.0, 0.0]])
    expected = torch.zeros((n_slices, 6, 6), dtype=torch.complex128)
    origin_y, origin_x = recon.parameter_group.object.pos_origin_coords.int().tolist()
    for batch in range(2):
        recon.current_minibatch = batch
        recon._initialize_object_gradient()
        for sl in range(n_slices):
            value = (sl + 1) * (batch + 1) * (1 + 2j)
            update = recon._combine_object_patch_update_directions(
                torch.full((1, 1, 2, 2), value),
                positions[batch : batch + 1],
                onto_accumulated=True,
                slice_index=sl,
            )
            recon._record_object_slice_gradient(sl, update)
            x = origin_x - 1 + batch
            expected[sl, origin_y : origin_y + 2, x : x + 2] += value
    torch.testing.assert_close(recon.parameter_group.object.get_grad(), -expected)
    # A uniform preconditioner makes the damping denominator exactly 5.
    fm.get_probe_grid_preconditioner = Mock(return_value=torch.full((6, 6), 4.0))
    recon.options.preconditioning_damping_factor = 0.75
    torch.testing.assert_close(
        recon._precondition_accumulated_object_update_direction(), expected / 5
    )


@pytest.mark.parametrize("batching", [api.BatchingModes.RANDOM, api.BatchingModes.COMPACT])
def test_multiprocess_object_accumulation_reduction_policy(reconstructor, batching):
    reconstructor.__class__ = MultiprocessLSQMLReconstructor
    reconstructor.options.batching_mode = batching
    reconstructor.parameter_group.object.initialize_grad()
    reconstructor.forward_model = SimpleNamespace(
        resampling_enabled=False,
        place_object_patches_on_probe_grid=Mock(return_value=torch.tensor([[1 + 2j, 3]])),
    )
    reconstructor.sync_buffer = Mock(side_effect=lambda value, **kw: value * 2)
    actual = reconstructor._combine_object_patch_update_directions(
        torch.ones((1, 1, 1, 2)), torch.zeros((1, 2))
    )
    multiplier = 1 if batching == api.BatchingModes.COMPACT else 2
    torch.testing.assert_close(actual, multiplier * torch.tensor([[[1 + 2j, 3]]]))
    assert reconstructor.sync_buffer.call_count == (batching != api.BatchingModes.COMPACT)


def test_gaussian_reciprocal_step_matches_amplitude_projection(reconstructor, monkeypatch):
    psi = torch.tensor([[[[3 + 4j, 2j]]]])
    reconstructor.noise_model = fm.PtychographyGaussianNoiseModel(eps=0)
    reconstructor.forward_model = make_task().reconstructor.forward_model
    reconstructor.forward_model.intermediate_variables["psi_far"] = psi
    monkeypatch.setattr(
        reconstructor.forward_model.free_space_propagator, "propagate_backward", lambda x: x
    )
    actual = reconstructor.run_reciprocal_space_step(
        torch.tensor([[[25.0, 4.0]]]), torch.tensor([[[100.0, 1.0]]]), torch.tensor([0])
    )
    torch.testing.assert_close(actual, torch.tensor([[[[6 + 8j, 1j]]]]))


def test_poisson_step_zero_residual_halves_previous_step_twice(reconstructor, monkeypatch):
    reconstructor.noise_model = fm.PtychographyPoissonNoiseModel()
    reconstructor.alpha_psi_far_all_pos = torch.tensor([0.8, 0.4, 0.6])
    reconstructor.forward_model = make_task().reconstructor.forward_model
    monkeypatch.setattr(torch, "randn", lambda shape, **kw: torch.zeros(shape, **kw))
    actual = reconstructor.get_psi_far_step_size(
        torch.ones((1, 1, 2)), torch.ones((1, 1, 2)), torch.tensor([1]), eps=0
    )
    torch.testing.assert_close(actual, torch.tensor([0.1]))
    torch.testing.assert_close(reconstructor.alpha_psi_far_all_pos, torch.tensor([0.8, 0.1, 0.6]))


@pytest.mark.parametrize("rescale", [False, True])
def test_epoch_zero_routes_to_intensity_scaling_or_updates(reconstructor, rescale):
    reconstructor.current_epoch = 0
    reconstructor.options.rescale_probe_intensity_in_first_epoch = rescale
    reconstructor.indices = []
    reconstructor.prepare_optimizer_closures = Mock()
    reconstructor.forward_model = Mock(return_value=torch.tensor([[[1.0, 4.0]]]))
    reconstructor.forward_model.detector_response.is_identity = True
    reconstructor.update_fourier_error = Mock()
    reconstructor.compute_reconstruction_parameter_updates = Mock()
    reconstructor.apply_reconstruction_parameter_updates = Mock()
    reconstructor.loss_tracker = Mock()
    reconstructor.reconstructor_buffers = SimpleNamespace(
        accumulated_true_intensity=0.0, accumulated_pred_intensity=0.0
    )
    reconstructor.run_minibatch([torch.tensor([0])], torch.tensor([[[4.0, 16.0]]]))
    assert reconstructor.compute_reconstruction_parameter_updates.call_count == (not rescale)
    assert reconstructor.apply_reconstruction_parameter_updates.call_count == (not rescale)
    if rescale:
        with torch.no_grad():
            reconstructor._apply_probe_intensity_scaling_correction()
        torch.testing.assert_close(
            reconstructor.parameter_group.probe.data, torch.full((1, 2, 2, 2), 2 + 0j)
        )


@pytest.mark.parametrize(
    "n_slices,slice_index,joint,expected_joint",
    [
        (1, 0, True, True),
        (1, 0, False, False),
        (2, 0, True, True),
        (2, 1, True, False),
        (2, 0, False, False),
    ],
)
@pytest.mark.parametrize("first_mode", [False, True])
def test_optimal_step_size_routing_and_indexed_buffers(
    reconstructor,
    n_slices,
    slice_index,
    joint,
    expected_joint,
    first_mode,
):
    reconstructor.parameter_group.object = SimpleNamespace(is_multislice=n_slices > 1)
    reconstructor.options.single_slice_solve_obj_prb_step_size_jointly = joint
    reconstructor.options.multislice_solve_obj_prb_step_size_jointly = joint
    reconstructor.options.solve_step_sizes_only_using_first_probe_mode = first_mode
    reconstructor.reconstructor_buffers = SimpleNamespace(
        alpha_object_all_pos_all_slices=torch.full((3, n_slices), -1.0),
        alpha_probe_all_pos=torch.full((3,), -1.0),
    )
    reconstructor.calculate_object_and_probe_update_step_sizes = Mock(
        return_value=(torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0]))
    )
    reconstructor.calculate_object_update_step_sizes = Mock(return_value=torch.tensor([6.0, 7.0]))
    reconstructor.calculate_probe_update_step_sizes = Mock(return_value=torch.tensor([8.0, 9.0]))
    all_modes, mode_zero = torch.tensor([11.0]), torch.tensor([22.0])
    reconstructor.calculate_optimal_step_sizes(
        torch.tensor([2, 0]), None, None, all_modes, None, mode_zero, slice_index=slice_index
    )
    expected_o = torch.full((3, n_slices), -1.0)
    expected_o[0, slice_index] = 3 if expected_joint else 7
    expected_o[2, slice_index] = 2 if expected_joint else 6
    torch.testing.assert_close(
        reconstructor.reconstructor_buffers.alpha_object_all_pos_all_slices, expected_o
    )
    torch.testing.assert_close(
        reconstructor.reconstructor_buffers.alpha_probe_all_pos,
        torch.tensor([5.0, -1.0, 4.0] if expected_joint else [9.0, -1.0, 8.0]),
    )
    solver = (
        reconstructor.calculate_object_and_probe_update_step_sizes
        if expected_joint
        else reconstructor.calculate_object_update_step_sizes
    )
    direction = solver.call_args.args[2 if expected_joint else 1]
    assert direction is (mode_zero if first_mode else all_modes)
    assert solver.call_args.kwargs["probe_mode_index"] == (0 if first_mode else None)


def test_compact_final_step_uses_minimum_of_batch_trimmed_means(reconstructor):
    # Twenty values discard one sample at each tail. The short batch is untrimmed.
    first = torch.tensor([[-100.0, -200.0]] + [[2.0, 8.0]] * 18 + [[100.0, 200.0]])
    second = torch.tensor([[4.0, 2.0], [6.0, 4.0]])
    reconstructor.indices = [torch.arange(20), torch.tensor([20, 21])]
    reconstructor.reconstructor_buffers = SimpleNamespace(
        alpha_object_all_pos_all_slices=torch.cat([first, second])
    )
    torch.testing.assert_close(
        reconstructor._calculate_final_object_update_step_size(), torch.tensor([2.0, 3.0])
    )


@pytest.mark.parametrize(
    "multimodal,first_mode,expected",
    [
        (False, False, 2 - 1j),
        (False, True, 2 - 1j),
        (True, False, 5 - 2j),
        (True, True, 5 - 2j),
    ],
)
def test_compact_object_directions_do_not_double_count_first_mode(
    reconstructor,
    multimodal,
    first_mode,
    expected,
):
    reconstructor.options.batching_mode = api.BatchingModes.COMPACT
    reconstructor.options.solve_step_sizes_only_using_first_probe_mode = first_mode
    reconstructor.options.preconditioning_damping_factor = 0.0
    reconstructor.parameter_group.object.options.multimodal_update = multimodal
    reconstructor.parameter_group.object.initialize_grad()
    reconstructor.forward_model = SimpleNamespace(
        resampling_enabled=False,
        place_object_patches_on_probe_grid=lambda pos, patches, **kw: patches.sum(0),
        get_probe_grid_preconditioner=lambda: torch.ones((2, 2)),
        extract_probe_grid_patches=lambda value, pos, shape, **kw: value[None],
    )
    chi = torch.tensor([2 - 1j, 1 + 3j]).reshape(1, 2, 1, 1).expand(1, 2, 2, 2)
    probe = torch.tensor([1 + 0j, 1j]).reshape(1, 2, 1, 1).expand_as(chi)
    for batch in range(2):
        combined, preconditioned, patches, mode_zero = (
            reconstructor.calculate_object_update_directions(chi, probe, torch.zeros((1, 2)))
        )
        torch.testing.assert_close(combined, torch.full((1, 2, 2), (batch + 1) * expected))
        torch.testing.assert_close(preconditioned, combined)
        torch.testing.assert_close(patches, combined[None])
        if first_mode or not multimodal:
            # The first-mode solve sees the previous batch's complete accumulator.
            torch.testing.assert_close(
                mode_zero, torch.full((1, 1, 2, 2), batch * expected + 2 - 1j)
            )
        else:
            assert mode_zero is None
        reconstructor._record_object_slice_gradient(0, combined)


@pytest.mark.parametrize(
    "n_slices,object_coefficients,probe_coefficient",
    [
        (2, [-3j, -2j], -6j),
        (3, [-12.0, -8.0, -6.0], -24.0),
    ],
)
@torch.no_grad()
def test_multislice_reverse_updates_use_incident_fields_and_adjoint_propagation(
    n_slices,
    object_coefficients,
    probe_coefficient,
):
    options = api.LSQMLOptions()
    options.reconstructor_options.preconditioning_damping_factor = 0.0
    options.reconstructor_options.solve_step_sizes_only_using_first_probe_mode = False
    task = make_task(options, n_slices=n_slices)
    recon = task.reconstructor
    model = recon.forward_model
    values = torch.tensor([2.0, 3.0, 4.0])[:n_slices, None, None]
    task.object.set_data(values.expand(n_slices, 6, 6).to(torch.complex128))
    # A toy unitary propagator multiplies by i; its adjoint multiplies by -i.
    # Slice transmissions 2,3,4 give incident fields P,2iP,-6P.
    model.propagate_to_next_slice = lambda psi, slice_index: 1j * psi
    model.propagate_to_previous_slice = Mock(side_effect=lambda psi, slice_index: -1j * psi)
    model.get_probe_grid_preconditioner = lambda: torch.ones((6, 6))
    recon.calculate_optimal_step_sizes = Mock()
    indices = torch.tensor([0])
    model(indices)
    psi = model.intermediate_variables["psi"]
    recon.run_real_space_step(psi + (1 + 2j), indices)

    # (1+2i)*conj(P) for P=[[1,2i],[3,4i]].
    conjugate_product = torch.tensor([[1 + 2j, 4 - 2j], [3 + 6j, 8 - 4j]])
    expected = torch.zeros((n_slices, 6, 6), dtype=torch.complex128)
    for i, coefficient in enumerate(object_coefficients):
        expected[i, 2:4, 1:3] = -coefficient * conjugate_product
    torch.testing.assert_close(task.object.get_grad(), expected)
    torch.testing.assert_close(
        task.probe.get_grad(), torch.full((1, 1, 2, 2), -probe_coefficient * (1 + 2j))
    )
    assert [
        call.kwargs["slice_index"] for call in model.propagate_to_previous_slice.call_args_list
    ] == list(range(n_slices - 1, 0, -1))
