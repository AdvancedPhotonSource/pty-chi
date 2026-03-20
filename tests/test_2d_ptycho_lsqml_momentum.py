import argparse

import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.reconstructors.lsqml import LSQMLReconstructor, MomentumState
from ptychi.utils import get_suggested_object_size, get_default_complex_dtype, generate_initial_opr_mode_weights

import test_utils as tutils


class DummyProbeForMomentum:
    def __init__(self):
        self.data = torch.zeros((1, 1, 2, 2), dtype=torch.complex64)
        self.set_data_calls = []

    def set_data(self, data, slicer=None, op="set"):
        self.set_data_calls.append((data.clone(), slicer, op))
        self.data[slicer] = data


class DummyObjectROI:
    def get_bbox_with_top_left_origin(self):
        return self

    def get_slicer(self):
        return (slice(None), slice(None))


class DummyObjectForMomentum:
    def __init__(self, n_slices=1):
        self.data = torch.zeros((n_slices, 2, 2), dtype=torch.complex64)
        self.preconditioner = torch.ones((2, 2), dtype=torch.float32)
        self.roi_bbox = DummyObjectROI()
        self.n_slices = n_slices
        self.set_data_calls = []

    def set_data(self, data, slicer=None, op="set"):
        self.set_data_calls.append((data.clone(), slicer, op))
        self.data[slicer] = data


def test_apply_probe_momentum_preserves_probe_side_effects():
    reconstructor = object.__new__(LSQMLReconstructor)
    probe = DummyProbeForMomentum()
    reconstructor.parameter_group = type("PG", (), {"probe": probe})()
    reconstructor.options = type(
        "Opt", (), {"momentum_acceleration_gain": 0.5, "momentum_acceleration_gradient_mixing_factor": 1.0}
    )()
    reconstructor._fourier_error_ok = lambda: True

    normalized_history = [
        torch.ones((1, 2, 2), dtype=torch.complex64) * scale for scale in (0.25, 0.5, 0.75, 1.0)
    ]
    reconstructor.probe_momentum_params = MomentumState(
        update_direction_history=[x.clone() for x in normalized_history],
        velocity_map=torch.zeros((1, 2, 2), dtype=torch.complex64),
    )

    delta_p_hat = torch.ones((1, 2, 2), dtype=torch.complex64)
    reconstructor._apply_probe_momentum(torch.tensor(1.0), delta_p_hat)

    assert len(reconstructor.probe_momentum_params.update_direction_history) == 4
    assert torch.equal(reconstructor.probe_momentum_params.velocity_map[0], delta_p_hat[0])
    assert len(probe.set_data_calls) == 1
    assert torch.equal(probe.data[0, 0], 0.5 * delta_p_hat[0])


def test_apply_object_momentum_preserves_object_side_effects():
    reconstructor = object.__new__(LSQMLReconstructor)
    object_ = DummyObjectForMomentum()
    reconstructor.parameter_group = type("PG", (), {"object": object_})()
    reconstructor.options = type(
        "Opt", (), {"momentum_acceleration_gain": 0.5, "momentum_acceleration_gradient_mixing_factor": 1.0}
    )()
    reconstructor._fourier_error_ok = lambda: True

    normalized_history = [
        torch.ones((1, 2, 2), dtype=torch.complex64) * scale for scale in (0.5, 0.75, 1.0)
    ]
    reconstructor.object_momentum_params = MomentumState(
        update_direction_history=[x.clone() for x in normalized_history],
        velocity_map=torch.zeros((1, 2, 2), dtype=torch.complex64),
    )

    delta_o_hat = torch.ones((1, 2, 2), dtype=torch.complex64)
    reconstructor._apply_object_momentum(torch.tensor([1.0]), delta_o_hat)

    assert len(reconstructor.object_momentum_params.update_direction_history) == 3
    assert torch.equal(reconstructor.object_momentum_params.velocity_map[0], delta_o_hat[0])
    assert len(object_.set_data_calls) == 1
    expected_weight = object_.preconditioner / (0.1 * object_.preconditioner.max() + object_.preconditioner)
    assert torch.allclose(object_.data[0], expected_weight * 0.5 * delta_o_hat[0])


def test_apply_probe_momentum_keeps_velocity_during_warmup_when_fourier_error_fails():
    reconstructor = object.__new__(LSQMLReconstructor)
    probe = DummyProbeForMomentum()
    reconstructor.parameter_group = type("PG", (), {"probe": probe})()
    reconstructor.options = type(
        "Opt", (), {"momentum_acceleration_gain": 0.5, "momentum_acceleration_gradient_mixing_factor": 1.0}
    )()
    reconstructor._fourier_error_ok = lambda: False

    normalized_history = [
        torch.ones((1, 2, 2), dtype=torch.complex64) * scale for scale in (0.25, 0.5, 0.75)
    ]
    initial_velocity = torch.full((1, 2, 2), 3.0 + 0.0j, dtype=torch.complex64)
    reconstructor.probe_momentum_params = MomentumState(
        update_direction_history=[x.clone() for x in normalized_history],
        velocity_map=initial_velocity.clone(),
    )

    delta_p_hat = torch.ones((1, 2, 2), dtype=torch.complex64)
    reconstructor._apply_probe_momentum(torch.tensor(1.0), delta_p_hat)

    assert len(reconstructor.probe_momentum_params.update_direction_history) == 4
    assert torch.equal(reconstructor.probe_momentum_params.velocity_map, initial_velocity)
    assert len(probe.set_data_calls) == 0


def test_apply_object_momentum_updates_history_once_for_multislice_object():
    reconstructor = object.__new__(LSQMLReconstructor)
    object_ = DummyObjectForMomentum(n_slices=2)
    reconstructor.parameter_group = type("PG", (), {"object": object_})()
    reconstructor.options = type(
        "Opt", (), {"momentum_acceleration_gain": 0.5, "momentum_acceleration_gradient_mixing_factor": 1.0}
    )()
    reconstructor._fourier_error_ok = lambda: True

    history_a = torch.ones((2, 2, 2), dtype=torch.complex64) * 0.5
    history_b = torch.ones((2, 2, 2), dtype=torch.complex64) * 0.75
    history_c = torch.ones((2, 2, 2), dtype=torch.complex64) * 1.0
    reconstructor.object_momentum_params = MomentumState(
        update_direction_history=[history_a.clone(), history_b.clone(), history_c.clone()],
        velocity_map=torch.zeros((2, 2, 2), dtype=torch.complex64),
    )

    delta_o_hat = torch.ones((2, 2, 2), dtype=torch.complex64)
    reconstructor._apply_object_momentum(torch.tensor([1.0, 1.0]), delta_o_hat)

    history = reconstructor.object_momentum_params.update_direction_history
    assert len(history) == 3
    assert torch.equal(history[0], history_b)
    assert torch.equal(history[1], history_c)
    assert torch.equal(history[2], torch.ones((2, 2, 2), dtype=torch.complex64))
    assert torch.equal(reconstructor.object_momentum_params.velocity_map, delta_o_hat)
    assert len(object_.set_data_calls) == 2


def test_apply_object_momentum_keeps_all_slices_in_warmup_until_history_overflows():
    reconstructor = object.__new__(LSQMLReconstructor)
    object_ = DummyObjectForMomentum(n_slices=2)
    reconstructor.parameter_group = type("PG", (), {"object": object_})()
    reconstructor.options = type(
        "Opt", (), {"momentum_acceleration_gain": 0.5, "momentum_acceleration_gradient_mixing_factor": 1.0}
    )()
    reconstructor._fourier_error_ok = lambda: True

    normalized_history = [
        torch.ones((2, 2, 2), dtype=torch.complex64) * scale for scale in (0.5, 0.75)
    ]
    reconstructor.object_momentum_params = MomentumState(
        update_direction_history=[x.clone() for x in normalized_history],
        velocity_map=torch.zeros((2, 2, 2), dtype=torch.complex64),
    )

    delta_o_hat = torch.ones((2, 2, 2), dtype=torch.complex64)
    reconstructor._apply_object_momentum(torch.tensor([1.0, 1.0]), delta_o_hat)

    history = reconstructor.object_momentum_params.update_direction_history
    assert len(history) == 3
    assert torch.equal(history[0], normalized_history[0])
    assert torch.equal(history[1], normalized_history[1])
    assert torch.equal(history[2], torch.ones((2, 2, 2), dtype=torch.complex64))
    assert torch.equal(
        reconstructor.object_momentum_params.velocity_map,
        torch.zeros((2, 2, 2), dtype=torch.complex64),
    )
    assert len(object_.set_data_calls) == 0


class Test2DPtychoLSQMLMomentum(tutils.TungstenDataTester):
    
    @tutils.TungstenDataTester.wrap_recon_tester(name='test_2d_ptycho_lsqml_momentum')
    def test_2d_ptycho_lsqml_momentum(self):
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(pos_type='true')
        
        options = api.LSQMLOptions()
        options.data_options.data = data
        
        options.object_options.initial_guess = torch.ones([1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.optimizable = True
        options.object_options.optimizer = api.Optimizers.SGD
        options.object_options.step_size = 1
        
        options.probe_options.initial_guess = probe
        options.probe_options.optimizable = True
        options.probe_options.optimizer = api.Optimizers.SGD
        options.probe_options.step_size = 1

        options.probe_position_options.position_x_px = positions_px[:, 1]
        options.probe_position_options.position_y_px = positions_px[:, 0]
        options.probe_position_options.optimizable = False
        
        options.reconstructor_options.batch_size = 96
        options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
        options.reconstructor_options.momentum_acceleration_gain = 0.5
        options.reconstructor_options.batching_mode = api.BatchingModes.COMPACT
        options.reconstructor_options.num_epochs = 12
        options.reconstructor_options.allow_nondeterministic_algorithms = False
        task = PtychographyTask(options)
        task.run()
        
        recon = task.get_data_to_cpu('object', as_numpy=True)[0]
        return recon
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = Test2DPtychoLSQMLMomentum()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_2d_ptycho_lsqml_momentum()
