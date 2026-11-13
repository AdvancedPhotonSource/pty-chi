import pytest
import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask


def _task_data():
    return {
        "diffraction_data": torch.ones((2, 4, 4)),
        "object_data": torch.ones((1, 8, 8), dtype=torch.complex64),
        "probe_data": torch.ones((1, 1, 4, 4), dtype=torch.complex64),
        "probe_position_x_px": torch.tensor([-1.0, 1.0]),
        "probe_position_y_px": torch.tensor([-1.0, 1.0]),
    }


def _options():
    options = api.LSQMLOptions()
    options.reconstructor_options.default_device = api.Devices.CPU
    options.reconstructor_options.num_epochs = 1
    options.reconstructor_options.batch_size = 2
    options.object_options.optimizable = False
    options.probe_options.optimizable = False
    options.probe_position_options.optimizable = False
    return options


def test_task_constructor_accepts_data_kwargs():
    task = PtychographyTask(_options(), **_task_data())

    assert task.dataset.patterns.shape == (2, 4, 4)
    assert task.object.data.shape == (1, 8, 8)
    assert task.probe.data.shape == (1, 1, 4, 4)
    assert task.probe_positions.data.shape == (2, 2)
    assert torch.allclose(task.opr_mode_weights.data.detach().cpu(), torch.ones((2, 1)))
    assert torch.equal(task.dataset.valid_pixel_mask.cpu(), torch.ones((4, 4), dtype=torch.bool))


def test_task_constructor_rejects_unknown_keyword():
    with pytest.raises(TypeError, match="unexpected keyword argument 'detector_response_mpa'"):
        PtychographyTask(_options(), **_task_data(), detector_response_mpa=None)


def test_deprecated_option_data_fallback_warns():
    options = _options()
    data = _task_data()
    options.data_options.data = data["diffraction_data"]
    options.object_options.initial_guess = data["object_data"]
    options.probe_options.initial_guess = data["probe_data"]
    options.probe_position_options.position_x_px = data["probe_position_x_px"]
    options.probe_position_options.position_y_px = data["probe_position_y_px"]

    with pytest.warns(DeprecationWarning) as warnings_record:
        task = PtychographyTask(options)

    warning_messages = [str(w.message) for w in warnings_record]
    assert any("options.data_options.data" in message for message in warning_messages)
    assert task.dataset.patterns.shape == (2, 4, 4)


def test_task_kwargs_take_precedence_over_deprecated_option_data():
    options = _options()
    data = _task_data()
    options.data_options.data = torch.zeros_like(data["diffraction_data"])
    options.object_options.initial_guess = torch.zeros_like(data["object_data"])
    options.probe_options.initial_guess = torch.zeros_like(data["probe_data"])
    options.probe_position_options.position_x_px = data["probe_position_x_px"] + 10
    options.probe_position_options.position_y_px = data["probe_position_y_px"] + 10

    with pytest.warns(DeprecationWarning) as warnings_record:
        task = PtychographyTask(options, **data)

    warning_messages = [str(w.message) for w in warnings_record]
    assert any("was ignored" in message for message in warning_messages)
    assert torch.allclose(task.dataset.patterns.cpu(), data["diffraction_data"])
    assert torch.allclose(task.object.data.detach().cpu(), data["object_data"])
    assert torch.allclose(task.probe_positions.data.detach().cpu()[:, 1], data["probe_position_x_px"])


def test_explicit_none_valid_pixel_mask_overrides_deprecated_mask():
    options = _options()
    data = _task_data()
    options.data_options.valid_pixel_mask = torch.zeros((4, 4), dtype=torch.bool)

    with pytest.warns(DeprecationWarning, match="valid_pixel_mask.*ignored"):
        task = PtychographyTask(options, **data, valid_pixel_mask=None)

    assert torch.equal(task.dataset.valid_pixel_mask.cpu(), torch.ones((4, 4), dtype=torch.bool))


def test_missing_required_task_data_errors():
    data = _task_data()
    data.pop("probe_data")

    with pytest.raises(ValueError, match="`probe_data` is required"):
        PtychographyTask(_options(), **data)


def test_settings_serialization_excludes_large_arrays():
    options = _options()
    data = _task_data()
    options.data_options.data = data["diffraction_data"]
    options.data_options.valid_pixel_mask = torch.ones((4, 4), dtype=torch.bool)
    options.object_options.initial_guess = data["object_data"]
    options.probe_options.initial_guess = data["probe_data"]
    options.probe_position_options.position_x_px = data["probe_position_x_px"]
    options.probe_position_options.position_y_px = data["probe_position_y_px"]
    options.opr_mode_weight_options.initial_weights = torch.ones((2, 1))

    serialized = options.get_dict()

    assert "data" not in serialized["data_options"]
    assert "valid_pixel_mask" not in serialized["data_options"]
    assert "initial_guess" not in serialized["object_options"]
    assert "initial_guess" not in serialized["probe_options"]
    assert "position_x_px" not in serialized["probe_position_options"]
    assert "position_y_px" not in serialized["probe_position_options"]
    assert "initial_weights" not in serialized["opr_mode_weight_options"]
    assert serialized["reconstructor_options"]["default_device"] == api.Devices.CPU


def test_lightweight_reconstruction_uses_task_kwargs():
    options = api.EPIEOptions()
    options.reconstructor_options.default_device = api.Devices.CPU
    options.reconstructor_options.num_epochs = 1
    options.reconstructor_options.batch_size = 2
    options.reconstructor_options.allow_nondeterministic_algorithms = False
    options.object_options.step_size = 0.1
    options.probe_options.step_size = 0.1
    options.probe_position_options.optimizable = False

    task = PtychographyTask(options, **_task_data())
    task.run()

    assert task.reconstructor.current_epoch == 1
