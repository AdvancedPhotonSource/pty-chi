import pytest
import torch
from pydantic import ValidationError

import ptychi.api as api
import ptychi.image_proc as image_proc
import ptychi.workflows.progressive_resolution as progressive_resolution_module
from ptychi.workflows import ProgressiveResolutionWorkflow


def _task_options():
    options = api.EPIEOptions()
    options.data_options.fft_shift = False
    options.reconstructor_options.default_device = api.Devices.CPU
    options.object_options.pixel_size_m = 2.5
    return options


def _workflow_data():
    return {
        "diffraction_data": torch.arange(
            2 * 9 * 11, dtype=torch.float32, device="cpu"
        ).reshape(2, 9, 11),
        "object_data": torch.ones((1, 17, 19), dtype=torch.complex64, device="cpu"),
        "probe_data": torch.ones((1, 1, 9, 11), dtype=torch.complex64, device="cpu"),
        "probe_position_x_px": torch.tensor([-4.0, 4.0], device="cpu"),
        "probe_position_y_px": torch.tensor([-2.0, 2.0], device="cpu"),
        "opr_mode_weights_data": torch.ones((2, 1), device="cpu"),
        "valid_pixel_mask": torch.ones((9, 11), dtype=torch.bool, device="cpu"),
    }


class _MovableParameter:
    def __init__(self):
        self.devices = []
        self.preconditioner = None
        self.update_buffer = None
        self.optimizer = None

    def to(self, device):
        self.devices.append(torch.device(device))
        return self


class _ParameterGroup:
    def __init__(self):
        self.parameters = [_MovableParameter() for _ in range(4)]

    def get_all_reconstruct_parameters(self):
        return self.parameters


class _Buffers:
    def get_all_names(self):
        return []


class _Reconstructor:
    def __init__(self):
        self.parameter_group = _ParameterGroup()
        self.reconstructor_buffers = _Buffers()


class _Dataset:
    def __init__(self, patterns, valid_pixel_mask):
        self.patterns = patterns
        self.valid_pixel_mask = valid_pixel_mask
        self.devices = []

    def move_attributes_to_device(self, device=None):
        device = torch.device(device)
        self.devices.append(device)
        if self.valid_pixel_mask is not None:
            self.valid_pixel_mask = self.valid_pixel_mask.to(device)


class _FakeTask:
    instances = []
    fail_level = None

    def __init__(self, options, *args, **kwargs):
        self.options = options
        self.input_data = kwargs
        self.level = len(self.__class__.instances)
        self.__class__.instances.append(self)
        self.dataset = _Dataset(
            kwargs["diffraction_data"], kwargs.get("valid_pixel_mask")
        )
        self.reconstructor = _Reconstructor()
        self.offload_devices = []
        self.results = {
            "object": kwargs["object_data"].clone(),
            "probe": kwargs["probe_data"].clone(),
            "probe_positions": torch.stack(
                [kwargs["probe_position_y_px"], kwargs["probe_position_x_px"]], dim=1
            ),
            "opr_mode_weights": kwargs["opr_mode_weights_data"].clone(),
        }

    def run(self):
        if self.level == self.__class__.fail_level:
            raise RuntimeError("level failed")
        increment = self.level + 1
        self.results = {name: value + increment for name, value in self.results.items()}

    def get_data_to_cpu(self, name):
        return self.results[name].detach().cpu()

    def set_large_tensor_device(self, device=None):
        device = torch.device(device)
        self.offload_devices.append(device)
        self.dataset.patterns = self.dataset.patterns.to(device)
        self.dataset.move_attributes_to_device(device)
        for parameter in self.reconstructor.parameter_group.get_all_reconstruct_parameters():
            parameter.to(device)


def _reset_fake_task():
    _FakeTask.instances = []
    _FakeTask.fail_level = None


def test_progressive_resolution_options_validate_levels_and_epochs():
    options = api.ProgressiveResolutionWorkflowOptions(
        num_resolution_levels=2, num_epochs_all_levels=[1, 2]
    )
    assert options.num_resolution_levels == 2

    with pytest.raises(ValidationError, match="greater than 0"):
        api.ProgressiveResolutionWorkflowOptions(
            num_resolution_levels=0, num_epochs_all_levels=[]
        )
    with pytest.raises(ValidationError, match="one value for each"):
        api.ProgressiveResolutionWorkflowOptions(
            num_resolution_levels=2, num_epochs_all_levels=[1]
        )
    with pytest.raises(ValidationError, match="greater than 0"):
        api.ProgressiveResolutionWorkflowOptions(
            num_resolution_levels=2, num_epochs_all_levels=[1, 0]
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_workflow_stores_gpu_inputs_on_cpu():
    data = {name: value.cuda() for name, value in _workflow_data().items()}
    with pytest.warns(UserWarning, match="original tensors remain"):
        workflow = ProgressiveResolutionWorkflow(
            _task_options(),
            workflow_options=api.ProgressiveResolutionWorkflowOptions(
                num_resolution_levels=1, num_epochs_all_levels=[1]
            ),
            **data,
        )

    for name in data:
        assert getattr(workflow, name).device.type == "cpu"
        assert data[name].device.type == "cuda"


def test_workflow_runs_rounded_levels_and_transfers_results(monkeypatch):
    _reset_fake_task()
    monkeypatch.setattr(progressive_resolution_module, "PtychographyTask", _FakeTask)
    data = _workflow_data()
    original_options = _task_options()
    workflow = ProgressiveResolutionWorkflow(
        original_options,
        workflow_options=api.ProgressiveResolutionWorkflowOptions(
            num_resolution_levels=3, num_epochs_all_levels=[1, 2, 3]
        ),
        **data,
    )

    assert workflow.diffraction_data.device.type == "cpu"
    assert workflow.diffraction_data.data_ptr() != data["diffraction_data"].data_ptr()
    assert workflow.task_options is not original_options
    with pytest.raises(RuntimeError, match="not available"):
        workflow.get_full_resolution_task()

    workflow.run()

    assert workflow.tasks == _FakeTask.instances
    assert workflow.get_full_resolution_task() is workflow.tasks[-1]
    assert [task.options.reconstructor_options.num_epochs for task in workflow.tasks] == [
        1,
        2,
        3,
    ]
    assert [task.options.object_options.pixel_size_m for task in workflow.tasks] == [
        10.0,
        5.0,
        2.5,
    ]
    assert original_options.reconstructor_options.num_epochs == 100
    assert original_options.object_options.pixel_size_m == 2.5

    assert [tuple(task.input_data["diffraction_data"].shape[-2:]) for task in workflow.tasks] == [
        (2, 3),
        (5, 6),
        (9, 11),
    ]
    assert [tuple(task.input_data["object_data"].shape[-2:]) for task in workflow.tasks] == [
        (4, 5),
        (9, 10),
        (17, 19),
    ]
    assert [tuple(task.input_data["probe_data"].shape[-2:]) for task in workflow.tasks] == [
        (2, 3),
        (5, 6),
        (9, 11),
    ]
    assert tuple(workflow.tasks[-1].input_data["valid_pixel_mask"].shape) == (9, 11)

    for i_level in (1, 2):
        previous_positions = workflow.tasks[i_level - 1].results["probe_positions"]
        assert torch.equal(
            workflow.tasks[i_level].input_data["probe_position_y_px"],
            previous_positions[:, 0] * 2,
        )
        assert torch.equal(
            workflow.tasks[i_level].input_data["probe_position_x_px"],
            previous_positions[:, 1] * 2,
        )
        assert torch.equal(
            workflow.tasks[i_level].input_data["opr_mode_weights_data"],
            workflow.tasks[i_level - 1].results["opr_mode_weights"],
        )

    for task in workflow.tasks:
        assert task.dataset.patterns.device.type == "cpu"
        assert task.dataset.devices == [torch.device("cpu")]
        assert task.offload_devices == [torch.device("cpu")]
        assert all(
            parameter.devices == [torch.device("cpu")]
            for parameter in task.reconstructor.parameter_group.parameters
        )

    with pytest.raises(RuntimeError, match="already been run"):
        workflow.run()


def test_reciprocal_crop_respects_fft_shift():
    options = _task_options()
    options.data_options.fft_shift = True
    workflow = ProgressiveResolutionWorkflow(
        options,
        workflow_options=api.ProgressiveResolutionWorkflowOptions(
            num_resolution_levels=2, num_epochs_all_levels=[1, 1]
        ),
        **_workflow_data(),
    )
    data = workflow.diffraction_data
    expected_size = (5, 6)

    cropped_centered_data = workflow._build_level_diffraction_data(factor=2)
    expected = image_proc.central_crop(data, expected_size)
    assert torch.equal(cropped_centered_data, expected)

    workflow.task_options.data_options.fft_shift = False
    cropped_raw_data = workflow._crop_reciprocal_data(data, factor=2)
    expected = torch.fft.fftshift(
        image_proc.central_crop(
            torch.fft.ifftshift(data, dim=(-2, -1)), expected_size
        ),
        dim=(-2, -1),
    )
    assert torch.equal(cropped_raw_data, expected)


def test_near_field_data_and_mask_are_resized_in_real_space():
    options = _task_options()
    options.data_options.free_space_propagation_distance_m = 0.1
    data = _workflow_data()
    data["valid_pixel_mask"] = (
        torch.arange(9 * 11, device="cpu").reshape(9, 11) % 3 != 0
    )
    workflow = ProgressiveResolutionWorkflow(
        options,
        workflow_options=api.ProgressiveResolutionWorkflowOptions(
            num_resolution_levels=2, num_epochs_all_levels=[1, 1]
        ),
        **data,
    )
    expected_size = (5, 6)

    resized_data = workflow._build_level_diffraction_data(factor=2)
    expected_data = torch.nn.functional.interpolate(
        data["diffraction_data"][:, None],
        size=expected_size,
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    assert torch.equal(resized_data, expected_data)

    resized_mask = workflow._build_level_valid_pixel_mask(factor=2)
    expected_mask = torch.nn.functional.interpolate(
        data["valid_pixel_mask"][None, None].to(torch.float32),
        size=expected_size,
        mode="nearest",
    )[0, 0].to(torch.bool)
    assert resized_mask is not None
    assert resized_mask.dtype == torch.bool
    assert torch.equal(resized_mask, expected_mask)

    assert torch.equal(
        workflow._build_level_diffraction_data(factor=1),
        data["diffraction_data"],
    )
    assert torch.equal(
        workflow._build_level_valid_pixel_mask(factor=1),
        data["valid_pixel_mask"],
    )


def test_failed_level_is_offloaded(monkeypatch):
    _reset_fake_task()
    _FakeTask.fail_level = 1
    monkeypatch.setattr(progressive_resolution_module, "PtychographyTask", _FakeTask)
    workflow = ProgressiveResolutionWorkflow(
        _task_options(),
        workflow_options=api.ProgressiveResolutionWorkflowOptions(
            num_resolution_levels=3, num_epochs_all_levels=[1, 1, 1]
        ),
        **_workflow_data(),
    )

    with pytest.raises(RuntimeError, match="level failed"):
        workflow.run()

    assert len(workflow.tasks) == 2
    assert all(task.offload_devices == [torch.device("cpu")] for task in workflow.tasks)
    with pytest.raises(RuntimeError, match="not available"):
        workflow.get_full_resolution_task()


def test_deprecated_option_data_fallback_warns():
    options = _task_options()
    data = _workflow_data()
    options.data_options.data = data["diffraction_data"]
    options.object_options.initial_guess = data["object_data"]
    options.probe_options.initial_guess = data["probe_data"]
    options.probe_position_options.position_x_px = data["probe_position_x_px"]
    options.probe_position_options.position_y_px = data["probe_position_y_px"]

    with pytest.warns(DeprecationWarning):
        workflow = ProgressiveResolutionWorkflow(
            options,
            workflow_options=api.ProgressiveResolutionWorkflowOptions(
                num_resolution_levels=1, num_epochs_all_levels=[1]
            ),
        )

    assert workflow.diffraction_data.device.type == "cpu"


def test_multilevel_workflow_runs_real_tasks():
    options = _task_options()
    options.reconstructor_options.batch_size = 2
    options.reconstructor_options.allow_nondeterministic_algorithms = False
    options.object_options.step_size = 0.1
    options.probe_options.step_size = 0.1
    data = {
        "diffraction_data": torch.ones((2, 5, 5), device="cpu"),
        "object_data": torch.ones((1, 11, 11), dtype=torch.complex64, device="cpu"),
        "probe_data": torch.ones((1, 1, 5, 5), dtype=torch.complex64, device="cpu"),
        "probe_position_x_px": torch.tensor([-1.0, 1.0], device="cpu"),
        "probe_position_y_px": torch.tensor([-1.0, 1.0], device="cpu"),
    }
    workflow = ProgressiveResolutionWorkflow(
        options,
        workflow_options=api.ProgressiveResolutionWorkflowOptions(
            num_resolution_levels=2, num_epochs_all_levels=[1, 1]
        ),
        **data,
    )

    workflow.run()

    task = workflow.get_full_resolution_task()
    assert len(workflow.tasks) == 2
    assert workflow.tasks[0].dataset.patterns.shape[-2:] == (3, 3)
    assert task.reconstructor.current_epoch == 1
    assert task.dataset.patterns.device.type == "cpu"
    assert task.get_data_to_cpu("object").shape == data["object_data"].shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_multilevel_workflow_offloads_real_gpu_tasks():
    options = _task_options()
    options.data_options.save_data_on_device = True
    options.reconstructor_options.default_device = api.Devices.GPU
    options.reconstructor_options.batch_size = 2
    options.object_options.step_size = 0.1
    options.probe_options.step_size = 0.1
    data = {
        "diffraction_data": torch.ones((2, 5, 5), device="cpu"),
        "object_data": torch.ones((1, 11, 11), dtype=torch.complex64, device="cpu"),
        "probe_data": torch.ones((1, 1, 5, 5), dtype=torch.complex64, device="cpu"),
        "probe_position_x_px": torch.tensor([-1.0, 1.0], device="cpu"),
        "probe_position_y_px": torch.tensor([-1.0, 1.0], device="cpu"),
    }
    workflow = ProgressiveResolutionWorkflow(
        options,
        workflow_options=api.ProgressiveResolutionWorkflowOptions(
            num_resolution_levels=2, num_epochs_all_levels=[1, 1]
        ),
        **data,
    )

    workflow.run()

    for task in workflow.tasks:
        assert task.dataset.patterns.device.type == "cpu"
        assert task.dataset.valid_pixel_mask.device.type == "cpu"
        for name in ("object", "probe", "probe_positions", "opr_mode_weights"):
            assert task.get_data(name).device.type == "cpu"
