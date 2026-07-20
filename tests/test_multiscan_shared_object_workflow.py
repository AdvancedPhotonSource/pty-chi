from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

import ptychi.api as api
import ptychi.workflows.multiscan_shared_object as multiscan_module
from ptychi.workflows import MultiscanSharedObjectWorkflow


def _task_options(n_tasks=3):
    options = []
    for _ in range(n_tasks):
        task_options = api.EPIEOptions()
        task_options.reconstructor_options.default_device = api.Devices.CPU
        task_options.object_options.pixel_size_m = 2.5
        options.append(task_options)
    return options


def _workflow_data(n_tasks=3):
    return {
        "diffraction_data": [
            torch.full((2, 5, 5), i_task + 1, dtype=torch.float32)
            for i_task in range(n_tasks)
        ],
        "object_data": [
            torch.full((1, 11, 11), i_task * 10 + 1, dtype=torch.complex64)
            for i_task in range(n_tasks)
        ],
        "probe_data": [
            torch.full((1, 1, 5, 5), i_task + 1, dtype=torch.complex64)
            for i_task in range(n_tasks)
        ],
        "probe_position_x_px": [torch.tensor([-1.0, 1.0]) for _ in range(n_tasks)],
        "probe_position_y_px": [torch.tensor([-1.0, 1.0]) for _ in range(n_tasks)],
    }


def _workflow_options(outer=2, inner=1):
    return api.MultiscanSharedObjectWorkflowOptions(
        num_outer_epochs=outer,
        num_inner_epochs=inner,
    )


class _FakeTask:
    instances = []
    events = []
    fail_task = None

    def __init__(self, options, *args, **kwargs):
        self.options = options
        self.input_data = kwargs
        self.index = len(self.__class__.instances)
        self.__class__.instances.append(self)
        self.reconstructor = SimpleNamespace(
            pbar=SimpleNamespace(disable=False),
            current_epoch=0,
        )
        self.results = {
            "object": kwargs["object_data"].clone(),
            "probe": kwargs["probe_data"].clone(),
            "probe_positions": torch.stack(
                [kwargs["probe_position_y_px"], kwargs["probe_position_x_px"]], dim=1
            ),
            "opr_mode_weights": kwargs["opr_mode_weights_data"],
        }
        self.devices = []

    def build_default_device(self):
        self.__class__.events.append(("default_device", self.index))

    def build_default_dtype(self):
        self.__class__.events.append(("default_dtype", self.index))

    def set_large_tensor_device(self, device=None):
        self.devices.append(device)

    def run(self, n_epochs):
        self.__class__.events.append(("run", self.index, n_epochs))
        if self.index == self.__class__.fail_task:
            raise RuntimeError("task failed")
        self.results["object"] = self.results["object"] + self.index + 1
        self.reconstructor.current_epoch += n_epochs

    def copy_data_from_task(self, task, params_to_copy):
        self.__class__.events.append(("copy", task.index, self.index, params_to_copy))
        assert params_to_copy == ("object",)
        self.results["object"] = task.results["object"].clone()

    def get_data_to_cpu(self, name):
        value = self.results[name]
        return None if value is None else value.detach().cpu()


def _reset_fake_task():
    _FakeTask.instances = []
    _FakeTask.events = []
    _FakeTask.fail_task = None


def test_multiscan_options_validate_epoch_counts():
    options = _workflow_options(outer=2, inner=3)
    assert options.num_outer_epochs == 2
    assert options.num_inner_epochs == 3

    with pytest.raises(ValidationError, match="num_outer_epochs"):
        _workflow_options(outer=0)
    with pytest.raises(ValidationError, match="num_inner_epochs"):
        _workflow_options(inner=0)


def test_multiscan_requires_nonempty_options_list_and_equal_data_lists():
    with pytest.raises(TypeError, match="must be a list"):
        MultiscanSharedObjectWorkflow(
            tuple(_task_options(1)),
            workflow_options=_workflow_options(),
            **_workflow_data(1),
        )
    with pytest.raises(ValueError, match="at least one"):
        MultiscanSharedObjectWorkflow(
            [],
            workflow_options=_workflow_options(),
            **_workflow_data(0),
        )

    single_scan = MultiscanSharedObjectWorkflow(
        _task_options(1),
        workflow_options=_workflow_options(),
        **_workflow_data(1),
    )
    assert len(single_scan.task_options) == 1

    data = _workflow_data(2)
    data["probe_data"] = data["probe_data"][:1]
    with pytest.raises(ValueError, match="one member for each task"):
        MultiscanSharedObjectWorkflow(
            _task_options(2),
            workflow_options=_workflow_options(),
            **data,
        )


def test_multiscan_accepts_optional_whole_none_and_mixed_none_lists():
    data = _workflow_data(2)
    data["opr_mode_weights_data"] = None
    data["valid_pixel_mask"] = [None, torch.ones((5, 5), dtype=torch.bool)]
    workflow = MultiscanSharedObjectWorkflow(
        _task_options(2),
        workflow_options=_workflow_options(),
        **data,
    )

    assert workflow.opr_mode_weights_data == [None, None]
    assert workflow.valid_pixel_mask[0] is None
    assert torch.equal(workflow.valid_pixel_mask[1], data["valid_pixel_mask"][1])


def test_multiscan_validates_shared_object_geometry_and_support_origin():
    options = _task_options(2)
    options[1].object_options.determine_position_origin_coords_by = (
        api.ObjectPosOriginCoordsMethods.POSITIONS
    )
    with pytest.raises(ValueError, match="SUPPORT"):
        MultiscanSharedObjectWorkflow(
            options,
            workflow_options=_workflow_options(),
            **_workflow_data(2),
        )

    data = _workflow_data(2)
    data["object_data"][1] = torch.ones((1, 13, 11), dtype=torch.complex64)
    with pytest.raises(ValueError, match="same shape"):
        MultiscanSharedObjectWorkflow(
            _task_options(2),
            workflow_options=_workflow_options(),
            **data,
        )

    options = _task_options(2)
    options[1].object_options.pixel_size_m = 3.0
    with pytest.raises(ValueError, match="pixel and slice geometry"):
        MultiscanSharedObjectWorkflow(
            options,
            workflow_options=_workflow_options(),
            **_workflow_data(2),
        )


def test_multiscan_runs_ring_then_synchronizes_final_object(monkeypatch):
    _reset_fake_task()
    monkeypatch.setattr(multiscan_module, "PtychographyTask", _FakeTask)
    original_options = _task_options(3)
    data = _workflow_data(3)
    original_probes = [probe.clone() for probe in data["probe_data"]]
    workflow = MultiscanSharedObjectWorkflow(
        original_options,
        workflow_options=_workflow_options(outer=2, inner=4),
        **data,
    )

    assert all(options is not original for options, original in zip(workflow.task_options, original_options))
    assert [options.reconstructor_options.num_epochs for options in workflow.task_options] == [
        8,
        8,
        8,
    ]
    assert [options.reconstructor_options.num_epochs for options in original_options] == [
        100,
        100,
        100,
    ]
    assert workflow.object_data[0].data_ptr() != data["object_data"][0].data_ptr()

    workflow.run()

    run_events = [event for event in _FakeTask.events if event[0] == "run"]
    assert run_events == [
        ("run", 0, 4),
        ("run", 1, 4),
        ("run", 2, 4),
        ("run", 0, 4),
        ("run", 1, 4),
        ("run", 2, 4),
    ]
    assert [task.reconstructor.current_epoch for task in workflow.tasks] == [8, 8, 8]
    assert not workflow.tasks[0].reconstructor.pbar.disable
    assert all(task.reconstructor.pbar.disable for task in workflow.tasks[1:])
    assert all(task.devices == ["cpu", None, "cpu", None, "cpu"] for task in workflow.tasks)

    final_object = workflow.tasks[-1].results["object"]
    assert torch.all(final_object == 13)
    assert all(torch.equal(task.results["object"], final_object) for task in workflow.tasks)
    assert all(
        torch.equal(task.results["probe"], original_probe)
        for task, original_probe in zip(workflow.tasks, original_probes)
    )

    with pytest.raises(RuntimeError, match="already been run"):
        workflow.run()


def test_multiscan_offloads_active_task_when_run_fails(monkeypatch):
    _reset_fake_task()
    _FakeTask.fail_task = 1
    monkeypatch.setattr(multiscan_module, "PtychographyTask", _FakeTask)
    workflow = MultiscanSharedObjectWorkflow(
        _task_options(3),
        workflow_options=_workflow_options(),
        **_workflow_data(3),
    )

    with pytest.raises(RuntimeError, match="task failed"):
        workflow.run()

    assert workflow.tasks[0].devices[-1] == "cpu"
    assert workflow.tasks[1].devices[-1] == "cpu"
    assert workflow.tasks[2].devices == ["cpu"]


def test_multiscan_deprecated_option_data_fallback_warns():
    options = _task_options(2)
    data = _workflow_data(2)
    for i_task, task_options in enumerate(options):
        task_options.data_options.data = data["diffraction_data"][i_task]
        task_options.object_options.initial_guess = data["object_data"][i_task]
        task_options.probe_options.initial_guess = data["probe_data"][i_task]
        task_options.probe_position_options.position_x_px = data[
            "probe_position_x_px"
        ][i_task]
        task_options.probe_position_options.position_y_px = data[
            "probe_position_y_px"
        ][i_task]

    with pytest.warns(DeprecationWarning):
        workflow = MultiscanSharedObjectWorkflow(
            options,
            workflow_options=_workflow_options(),
        )

    assert len(workflow.diffraction_data) == 2
    assert all(task_options.data_options.data is None for task_options in workflow.task_options)


def test_multiscan_runs_real_cpu_tasks():
    options = _task_options(2)
    for task_options in options:
        task_options.reconstructor_options.batch_size = 2
        task_options.reconstructor_options.allow_nondeterministic_algorithms = False
        task_options.object_options.step_size = 0.1
        task_options.probe_options.step_size = 0.1
    workflow = MultiscanSharedObjectWorkflow(
        options,
        workflow_options=_workflow_options(outer=1, inner=1),
        **_workflow_data(2),
    )

    workflow.run()

    final_object = workflow.tasks[-1].get_data_to_cpu("object")
    for task in workflow.tasks:
        assert task.reconstructor.current_epoch == 1
        assert task.dataset.patterns.device.type == "cpu"
        torch.testing.assert_close(
            task.get_data_to_cpu("object"), final_object, equal_nan=True
        )
        for name in ("object", "probe", "probe_positions", "opr_mode_weights"):
            assert task.get_data(name).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_multiscan_copies_gpu_inputs_without_modifying_callers():
    data = {
        name: [value.cuda() for value in values]
        for name, values in _workflow_data(2).items()
    }
    with pytest.warns(UserWarning, match="original tensors remain"):
        workflow = MultiscanSharedObjectWorkflow(
            _task_options(2),
            workflow_options=_workflow_options(),
            **data,
        )

    for name, values in data.items():
        assert all(value.device.type == "cuda" for value in values)
        assert all(value.device.type == "cpu" for value in getattr(workflow, name))
