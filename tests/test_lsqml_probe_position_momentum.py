from types import SimpleNamespace

import torch

import ptychi.api as api
from ptychi.reconstructors.lsqml import LSQMLReconstructor


def make_valid_options():
    options = api.LSQMLOptions()
    options.object_options.initial_guess = torch.ones((1, 8, 8), dtype=torch.complex64)
    options.probe_options.initial_guess = torch.ones((1, 1, 4, 4), dtype=torch.complex64)
    options.probe_position_options.position_y_px = torch.tensor([-1.0, 1.0])
    options.probe_position_options.position_x_px = torch.tensor([-1.0, 1.0])
    return options


def test_lsqml_probe_position_momentum_disabled_by_default():
    options = make_valid_options()

    options.check()

    assert options.probe_position_options.momentum_acceleration_gain == 0.0
    assert options.probe_position_options.momentum_acceleration_gradient_mixing_factor == 1
    assert options.probe_position_options.momentum_acceleration_memory == 3


def test_lsqml_probe_position_momentum_requires_positive_memory():
    options = make_valid_options()
    options.probe_position_options.momentum_acceleration_gain = 0.5
    options.probe_position_options.momentum_acceleration_memory = 0

    try:
        options.check()
    except ValueError as exc:
        assert "momentum_acceleration_memory" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive momentum memory.")


class DummyProbePositions:
    def __init__(
        self,
        tensor,
        momentum_acceleration_gain,
        momentum_acceleration_gradient_mixing_factor=1,
        step_size=0.25,
    ):
        self.tensor = tensor.clone()
        self.data = tensor.clone()
        self.step_size = step_size
        self.options = SimpleNamespace(
            momentum_acceleration_gain=momentum_acceleration_gain,
            momentum_acceleration_gradient_mixing_factor=momentum_acceleration_gradient_mixing_factor,
            momentum_acceleration_memory=2,
            correction_options=SimpleNamespace(
                update_magnitude_limit=None,
                clip_update_magnitude_by_mad=False,
            ),
        )
        self.position_correction = None
        self.last_grad = None
        self.step_calls = 0
        self.step_clip_update_args = []
        self.set_data_calls = []

    def set_grad(self, grad):
        self.last_grad = grad.clone()

    def get_grad(self):
        return self.last_grad

    def set_data(self, data, slicer=None, op="set"):
        self.set_data_calls.append((data.clone(), slicer, op))
        self.data = data.clone()

    def step_optimizer(self, clip_update=True):
        self.step_calls += 1
        self.step_clip_update_args.append(clip_update)
        self.data = self.data - self.step_size * self.last_grad


def test_update_probe_positions_uses_position_momentum_math_when_enabled():
    reconstructor = object.__new__(LSQMLReconstructor)

    probe_positions = DummyProbePositions(
        tensor=torch.zeros((4, 2), dtype=torch.float32),
        momentum_acceleration_gain=0.5,
        momentum_acceleration_gradient_mixing_factor=1.0,
    )
    expected_delta_pos = torch.tensor([[0.05, -0.04], [0.02, -0.01]], dtype=torch.float32)
    get_update_calls = []

    def get_update(chi, obj_patches, delta_o_patches, unique_probes, object_step_size):
        get_update_calls.append(
            (chi.clone(), obj_patches.clone(), delta_o_patches.clone(), unique_probes.clone(), object_step_size)
        )
        return expected_delta_pos.clone()

    probe_positions.position_correction = SimpleNamespace(get_update=get_update)
    reconstructor.parameter_group = SimpleNamespace(
        object=SimpleNamespace(step_size=0.25),
        probe_positions=probe_positions,
    )
    reconstructor.options = SimpleNamespace(momentum_acceleration_gradient_mixing_factor=0.0)
    reconstructor.forward_model = SimpleNamespace(free_space_propagation_distance_m=torch.inf)
    reconstructor.current_epoch = 2
    reconstructor.probe_position_momentum_params = {
        "position_update_history": [
            torch.tensor(
                [[0.0, 0.0], [0.03, -0.024], [0.0, 0.0], [0.012, -0.006]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[0.0, 0.0], [0.02, -0.016], [0.0, 0.0], [0.008, -0.004]],
                dtype=torch.float32,
            ),
        ],
        "position_update_history_epoch": 1,
        "velocity_map": torch.zeros((4, 2), dtype=torch.float32),
    }

    indices = torch.tensor([1, 3])
    chi = torch.ones((2, 1, 2, 2), dtype=torch.complex64)
    obj_patches = torch.ones((2, 1, 2, 2), dtype=torch.complex64)
    delta_o_patches = torch.ones((2, 1, 2, 2), dtype=torch.complex64)
    unique_probes = torch.ones((2, 1, 2, 2), dtype=torch.complex64)

    reconstructor.update_probe_positions(
        chi,
        indices,
        obj_patches,
        delta_o_patches,
        unique_probes,
        apply_updates=True,
    )

    assert len(get_update_calls) == 1
    assert get_update_calls[0][-1] == 0.25

    expected_update = torch.zeros((4, 2), dtype=torch.float32)
    expected_update[indices] = expected_delta_pos * 1.5
    assert torch.equal(probe_positions.last_grad, -expected_update)
    assert probe_positions.step_calls == 1
    assert probe_positions.step_clip_update_args == [False]
    assert len(probe_positions.set_data_calls) == 0

    expected_grad = torch.zeros((4, 2), dtype=torch.float32)
    expected_grad[indices] = expected_delta_pos * 0.25 * 1.5
    assert torch.equal(probe_positions.data, expected_grad)


def test_probe_position_momentum_history_is_stored_once_per_epoch():
    reconstructor = object.__new__(LSQMLReconstructor)

    probe_positions = DummyProbePositions(
        tensor=torch.zeros((4, 2), dtype=torch.float32),
        momentum_acceleration_gain=0.5,
        momentum_acceleration_gradient_mixing_factor=1.0,
    )
    reconstructor.parameter_group = SimpleNamespace(probe_positions=probe_positions)
    reconstructor.forward_model = SimpleNamespace(free_space_propagation_distance_m=torch.inf)
    reconstructor.current_epoch = 2
    reconstructor.probe_position_momentum_params = {
        "position_update_history": [
            torch.tensor(
                [[0.09, -0.03], [0.06, -0.02], [0.03, -0.01], [0.015, -0.005]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[0.06, -0.02], [0.04, -0.013333334], [0.02, -0.006666667], [0.01, -0.0033333334]],
                dtype=torch.float32,
            ),
        ],
        "position_update_history_epoch": 1,
        "velocity_map": torch.zeros((4, 2), dtype=torch.float32),
    }

    delta_pos_a = torch.tensor([[0.03, -0.01], [0.02, -0.006666667]], dtype=torch.float32)
    delta_pos_b = torch.tensor([[0.01, -0.0033333334], [0.005, -0.0016666667]], dtype=torch.float32)

    delta_pos_a_out = reconstructor._apply_probe_position_momentum(torch.tensor([0, 1]), delta_pos_a)
    assert len(reconstructor.probe_position_momentum_params["position_update_history"]) == 3

    delta_pos_b_out = reconstructor._apply_probe_position_momentum(torch.tensor([2, 3]), delta_pos_b)

    history = reconstructor.probe_position_momentum_params["position_update_history"]
    assert len(history) == 3
    assert reconstructor.probe_position_momentum_params["position_update_history_epoch"] == 2
    assert torch.equal(history[-1][0:2], delta_pos_a)
    assert torch.equal(history[-1][2:4], delta_pos_b)
    assert torch.equal(delta_pos_a_out, 1.5 * delta_pos_a)
    assert torch.equal(delta_pos_b_out, 1.5 * delta_pos_b)
    assert torch.equal(
        reconstructor.probe_position_momentum_params["velocity_map"][0:2], delta_pos_a
    )
    assert torch.equal(
        reconstructor.probe_position_momentum_params["velocity_map"][2:4], delta_pos_b
    )


def test_probe_position_momentum_history_rolls_over_across_epochs():
    reconstructor = object.__new__(LSQMLReconstructor)

    probe_positions = DummyProbePositions(
        tensor=torch.zeros((4, 2), dtype=torch.float32),
        momentum_acceleration_gain=0.5,
        momentum_acceleration_gradient_mixing_factor=1.0,
    )
    reconstructor.parameter_group = SimpleNamespace(probe_positions=probe_positions)
    reconstructor.forward_model = SimpleNamespace(free_space_propagation_distance_m=torch.inf)
    history_epoch_0 = torch.tensor(
        [[0.09, -0.03], [0.06, -0.02], [0.03, -0.01], [0.015, -0.005]],
        dtype=torch.float32,
    )
    history_epoch_1 = torch.tensor(
        [[0.06, -0.02], [0.04, -0.013333334], [0.02, -0.006666667], [0.01, -0.0033333334]],
        dtype=torch.float32,
    )
    history_epoch_2 = torch.tensor(
        [[0.03, -0.01], [0.02, -0.006666667], [0.01, -0.0033333334], [0.005, -0.0016666667]],
        dtype=torch.float32,
    )
    reconstructor.probe_position_momentum_params = {
        "position_update_history": [history_epoch_0.clone(), history_epoch_1.clone(), history_epoch_2.clone()],
        "position_update_history_epoch": 2,
        "velocity_map": torch.zeros((4, 2), dtype=torch.float32),
    }

    reconstructor.current_epoch = 3
    delta_pos_c = torch.tensor([[0.015, -0.005], [0.005, -0.0016666667]], dtype=torch.float32)
    reconstructor._apply_probe_position_momentum(torch.tensor([0, 3]), delta_pos_c)

    history = reconstructor.probe_position_momentum_params["position_update_history"]
    assert len(history) == 3
    assert reconstructor.probe_position_momentum_params["position_update_history_epoch"] == 3
    assert torch.equal(history[0], history_epoch_1)
    assert torch.equal(history[1], history_epoch_2)

    expected_epoch_3 = torch.zeros((4, 2), dtype=torch.float32)
    expected_epoch_3[torch.tensor([0, 3])] = delta_pos_c
    assert torch.equal(history[2], expected_epoch_3)


def test_probe_position_momentum_non_far_field_keeps_history_but_skips_velocity_update():
    reconstructor = object.__new__(LSQMLReconstructor)

    probe_positions = DummyProbePositions(
        tensor=torch.zeros((4, 2), dtype=torch.float32),
        momentum_acceleration_gain=0.5,
        momentum_acceleration_gradient_mixing_factor=1.0,
    )
    reconstructor.parameter_group = SimpleNamespace(probe_positions=probe_positions)
    reconstructor.forward_model = SimpleNamespace(free_space_propagation_distance_m=1.0)
    reconstructor.current_epoch = 2
    reconstructor.probe_position_momentum_params = {
        "position_update_history": [
            torch.tensor(
                [[0.09, -0.03], [0.06, -0.02], [0.03, -0.01], [0.015, -0.005]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[0.06, -0.02], [0.04, -0.013333334], [0.02, -0.006666667], [0.01, -0.0033333334]],
                dtype=torch.float32,
            ),
        ],
        "position_update_history_epoch": 1,
        "velocity_map": torch.full((4, 2), 7.0, dtype=torch.float32),
    }

    delta_pos = torch.tensor([[0.03, -0.01], [0.02, -0.006666667]], dtype=torch.float32)
    delta_pos_out = reconstructor._apply_probe_position_momentum(torch.tensor([0, 1]), delta_pos)

    history = reconstructor.probe_position_momentum_params["position_update_history"]
    assert len(history) == 2
    assert reconstructor.probe_position_momentum_params["position_update_history_epoch"] == 1
    assert torch.equal(
        reconstructor.probe_position_momentum_params["velocity_map"],
        torch.full((4, 2), 7.0, dtype=torch.float32),
    )
    assert torch.equal(delta_pos_out, delta_pos)


def test_apply_reconstruction_parameter_updates_uses_optimizer_for_positions_with_momentum():
    reconstructor = object.__new__(LSQMLReconstructor)
    probe_positions = DummyProbePositions(
        tensor=torch.zeros((4, 2), dtype=torch.float32),
        momentum_acceleration_gain=0.5,
        momentum_acceleration_gradient_mixing_factor=1.0,
    )
    grad = torch.zeros((4, 2), dtype=torch.float32)
    grad[torch.tensor([1, 3])] = torch.tensor([[-0.03, 0.02], [-0.01, 0.005]], dtype=torch.float32)
    probe_positions.set_grad(grad)

    reconstructor.current_epoch = 2
    reconstructor.parameter_group = SimpleNamespace(
        object=SimpleNamespace(
            optimization_enabled=lambda epoch: False,
        ),
        probe=SimpleNamespace(
            optimization_enabled=lambda epoch: False,
        ),
        probe_positions=SimpleNamespace(
            optimization_enabled=lambda epoch: True,
            step_optimizer=probe_positions.step_optimizer,
            options=probe_positions.options,
            data=probe_positions.data,
            get_grad=probe_positions.get_grad,
        ),
        opr_mode_weights=SimpleNamespace(
            optimization_enabled=lambda epoch: False,
        ),
    )
    reconstructor.reconstructor_buffers = SimpleNamespace(
        alpha_object_all_pos_all_slices=torch.ones((4, 1), dtype=torch.float32),
        alpha_probe_all_pos=torch.ones((4,), dtype=torch.float32),
    )
    reconstructor.options = SimpleNamespace(batching_mode=api.enums.BatchingModes.RANDOM)

    reconstructor.apply_reconstruction_parameter_updates(torch.tensor([1, 3]))

    assert probe_positions.step_calls == 1
    assert probe_positions.step_clip_update_args == [False]
    assert torch.equal(probe_positions.data, -probe_positions.step_size * grad)
