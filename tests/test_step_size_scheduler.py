import dataclasses

import pytest
import torch
from torch.utils.data import Dataset

import ptychi.api.enums as enums
from ptychi.api.options.base import OptimizationPlan, ParameterOptions
from ptychi.api.options.base import ReconstructorOptions
from ptychi.data_structures.base import ReconstructParameter
from ptychi.data_structures.parameter_group import ParameterGroup
from ptychi.reconstructors.base import IterativeReconstructor


def make_parameter(
    *,
    optimizable: bool = True,
    start: int = 0,
    stop: int | None = None,
    scheduler_class: str | None = "ExponentialLR",
    scheduler_options: dict | None = None,
):
    if scheduler_options is None:
        scheduler_options = {"gamma": 0.5}
    options = ParameterOptions(
        optimizable=optimizable,
        optimizer=enums.Optimizers.SGD,
        step_size=1.0,
        optimization_plan=OptimizationPlan(
            start=start,
            stop=stop,
            step_size_scheduler_class=scheduler_class,
            step_size_scheduler_options=scheduler_options,
        ),
    )
    return ReconstructParameter(shape=(1,), options=options, is_complex=False)


class TinyDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return torch.tensor(index), torch.tensor([0.0])

    def move_attributes_to_device(self, device):
        return None


@dataclasses.dataclass
class DummyParameterGroup(ParameterGroup):
    object: ReconstructParameter


class DummyReconstructor(IterativeReconstructor):
    def __init__(self, parameter_group, dataset, options):
        super().__init__(parameter_group=parameter_group, dataset=dataset, options=options)
        self.used_step_sizes = []

    def run_minibatch(self, input_data, y_true, *args, **kwargs):
        self.used_step_sizes.append(self.parameter_group.object.step_size)
        self.loss_tracker.update_batch_loss(loss=0.0)


def test_step_size_scheduler_updates_optimizer_and_cached_step_size():
    param = make_parameter()

    param.step_step_size_scheduler(epoch=0)

    assert param.optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    assert param.optimizer_params["lr"] == pytest.approx(0.5)
    assert param.step_size == pytest.approx(0.5)
    assert param.options.step_size == pytest.approx(1.0)


def test_step_size_scheduler_respects_optimization_interval():
    param = make_parameter(start=2, stop=4)

    param.step_step_size_scheduler(epoch=0)
    param.step_step_size_scheduler(epoch=1)
    assert param.optimizer.param_groups[0]["lr"] == pytest.approx(1.0)

    param.step_step_size_scheduler(epoch=2)
    assert param.optimizer.param_groups[0]["lr"] == pytest.approx(0.5)

    param.step_step_size_scheduler(epoch=3)
    assert param.optimizer.param_groups[0]["lr"] == pytest.approx(0.25)

    param.step_step_size_scheduler(epoch=4)
    assert param.optimizer.param_groups[0]["lr"] == pytest.approx(0.25)


def test_step_size_scheduler_is_not_built_when_disabled_or_non_optimizable():
    param_without_scheduler = make_parameter(scheduler_class=None)
    assert param_without_scheduler.step_size_scheduler is None

    non_optimizable_param = make_parameter(optimizable=False)
    assert non_optimizable_param.step_size_scheduler is None


def test_step_size_scheduler_requires_no_step_arguments():
    options = ParameterOptions(
        optimizable=True,
        optimizer=enums.Optimizers.SGD,
        step_size=1.0,
        optimization_plan=OptimizationPlan(
            step_size_scheduler_class="ReduceLROnPlateau",
        ),
    )

    with pytest.raises(ValueError, match="requires arguments in step"):
        ReconstructParameter(shape=(1,), options=options, is_complex=False)


def test_step_size_scheduler_is_used_by_reconstructor_epoch_loop():
    param = make_parameter()
    reconstructor = DummyReconstructor(
        parameter_group=DummyParameterGroup(object=param),
        dataset=TinyDataset(),
        options=ReconstructorOptions(
            num_epochs=2,
            batch_size=1,
            displayed_loss_function=None,
        ),
    )

    reconstructor.build()
    reconstructor.run()

    assert reconstructor.used_step_sizes == pytest.approx([1.0, 0.5])
    assert param.step_size == pytest.approx(0.25)
    assert param.options.step_size == pytest.approx(1.0)
