import pytest
import torch

import ptychi.api as api
import ptychi.maps as maps
from ptychi.api.options.base import ParameterOptions
from ptychi.api.task import PtychographyTask
from ptychi.data_structures.base import ReconstructParameter
from ptychi.utils import get_default_complex_dtype, set_default_complex_dtype


class RequiredClosureOptimizer(torch.optim.SGD):
    def step(self, closure):
        self.closure_calls = getattr(self, "closure_calls", 0) + 1
        with torch.enable_grad():
            loss = closure()
        super().step()
        return loss


class OptionalClosureOptimizer(RequiredClosureOptimizer):
    def step(self, closure=None):
        return super().step(closure)


class SuppliedGradientOptimizer(torch.optim.SGD):
    requires_closure = False


class ExplicitClosureOptimizer(OptionalClosureOptimizer):
    requires_closure = True


class NoClosureOptimizer(torch.optim.SGD):
    def step(self):
        return super().step()


def make_parameter(data, optimizer_class):
    parameter = ReconstructParameter(
        data=data, is_complex=data.is_complex(), options=ParameterOptions(step_size=0.1),
    )
    parameter.optimizer_class = optimizer_class
    parameter.build_optimizer()
    return parameter


@pytest.mark.parametrize("optimizer_name", list(api.Optimizers))
def test_builtin_optimizer_closure_policy(optimizer_name):
    optimizer_class = maps.get_optimizer_by_enum(optimizer_name)
    optimizer = optimizer_class([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    assert maps.optimizer_uses_closure(optimizer) == (optimizer_name == api.Optimizers.LBFGS)


@pytest.mark.parametrize("optimizer_class", [
    torch.optim.SGD, torch.optim.Adam, SuppliedGradientOptimizer, NoClosureOptimizer,
])
def test_supplied_gradients_are_preserved_without_automatic_closure(optimizer_class):
    parameter = make_parameter(torch.tensor([2.0]), optimizer_class)
    reference = torch.nn.Parameter(torch.tensor([2.0]))
    reference_optimizer = optimizer_class([reference], lr=0.1)
    parameter.set_grad(torch.tensor([3.0]))
    reference.grad = torch.tensor([3.0])
    parameter._automatic_optimizer_closure = lambda: pytest.fail("Unexpected closure evaluation")
    parameter.step_optimizer()
    reference_optimizer.step()
    torch.testing.assert_close(parameter.data, reference.detach())


def test_custom_closure_overrides_class_opt_out():
    parameter = make_parameter(torch.tensor([2.0]), SuppliedGradientOptimizer)
    parameter.optimizer_closure = parameter.make_optimizer_closure(
        lambda: parameter.data.square().sum()
    )
    parameter.step_optimizer()
    torch.testing.assert_close(parameter.data, torch.tensor([1.6]))


@pytest.mark.parametrize("complex_parameter", [False, True])
def test_closure_recomputes_target_gradients_without_touching_other_parameters(complex_parameter):
    value = 2 + 3j if complex_parameter else 2.0
    parameter = make_parameter(torch.tensor([value]), RequiredClosureOptimizer)
    other = torch.nn.Parameter(torch.tensor([4.0]))
    other.grad = torch.tensor([123.0])
    parameter.make_optimizer_closure(
        lambda: (parameter.data.abs().square() * other).sum(), store=True,
    )
    with torch.no_grad():
        parameter.step_optimizer()
        first = parameter.data.clone()
        parameter.step_optimizer()
    torch.testing.assert_close(first, torch.tensor([value]) * 0.2)
    torch.testing.assert_close(parameter.data, first * 0.2)
    torch.testing.assert_close(other.grad, torch.tensor([123.0]))


def test_optional_closure_detection_and_custom_override():
    parameter = make_parameter(torch.tensor([2.0]), OptionalClosureOptimizer)
    automatic_closure = parameter.make_optimizer_closure(
        lambda: parameter.data.square().sum(), store=True,
    )
    parameter.step_optimizer()
    torch.testing.assert_close(parameter.data, torch.tensor([1.6]))
    parameter.optimizer_closure = parameter.make_optimizer_closure(
        lambda: (parameter.data - 1.6).square().sum()
    )
    assert parameter._automatic_optimizer_closure is automatic_closure
    parameter.step_optimizer()
    torch.testing.assert_close(parameter.data, torch.tensor([1.6]))
    parameter.step_optimizer(closure=parameter.make_optimizer_closure(
        lambda: (parameter.data - 3).square().sum()
    ))
    torch.testing.assert_close(parameter.data, torch.tensor([1.88]))


def test_sgd_keeps_supplied_direction_and_required_closure_reports_missing_context():
    parameter = ReconstructParameter(data=torch.tensor([2.0]), options=ParameterOptions(step_size=0.1))
    parameter.set_grad(torch.tensor([3.0]))
    parameter._automatic_optimizer_closure = lambda: pytest.fail("SGD should use the supplied gradient")
    parameter.step_optimizer()
    torch.testing.assert_close(parameter.data, torch.tensor([1.7]))
    parameter.optimizer = RequiredClosureOptimizer([parameter.tensor], lr=0.1)
    parameter._automatic_optimizer_closure = None
    with pytest.raises(ValueError, match="needs an optimizer closure"):
        parameter.step_optimizer()


@pytest.fixture(autouse=True)
def restore_torch_defaults():
    dtype, device, complex_dtype = (
        torch.get_default_dtype(), torch.get_default_device(), get_default_complex_dtype()
    )
    yield
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    set_default_complex_dtype(complex_dtype)


def make_task(
    options_class=api.LSQMLOptions, parameter_name="real_space_scaling",
    optimizer=api.Optimizers.LBFGS, noise_model=api.NoiseModels.GAUSSIAN,
):
    options = options_class()
    options.reconstructor_options.default_device = api.Devices.CPU
    options.reconstructor_options.default_dtype = api.Dtypes.FLOAT64
    options.reconstructor_options.batch_size = 2
    options.reconstructor_options.num_epochs = 1
    options.reconstructor_options.forward_model_options.pad_for_shift = 2
    if isinstance(options, api.LSQMLOptions):
        options.reconstructor_options.rescale_probe_intensity_in_first_epoch = False
        options.reconstructor_options.noise_model = noise_model
    options.data_options.fft_shift = False
    options.object_options.pixel_size_m = 1.0
    options.object_options.remove_object_probe_ambiguity.enabled = False
    for name in ["object", "probe", "probe_position", "real_space_scaling", "opr_mode_weight"]:
        getattr(options, name + "_options").optimizable = name == parameter_name
    if options_class in (api.RPIEOptions, api.BHOptions):
        options.object_options.optimizable = True
    if parameter_name == "probe_position":
        options.probe_position_options.correction_options.clip_update_magnitude_by_mad = False
        options.probe_position_options.correction_options.update_magnitude_limit = None
    n_opr_modes = 2 if parameter_name == "opr_mode_weight" else 1
    parameter_options = getattr(options, parameter_name + "_options")
    parameter_options.optimizer = api.Optimizers.SGD if isinstance(optimizer, type) else optimizer
    parameter_options.optimizer_params = (
        {"line_search_fn": "strong_wolfe", "max_iter": 4} if optimizer == api.Optimizers.LBFGS else {}
    )
    parameter_options.step_size = 0.1
    generator = torch.Generator(device="cpu").manual_seed(312)
    task = PtychographyTask(
        options,
        diffraction_data=torch.ones((2, 5, 5), device="cpu"),
        object_data=1 + 0.1 * torch.randn((1, 17, 19), generator=generator, dtype=torch.complex128, device="cpu"),
        probe_data=torch.randn((n_opr_modes, 1, 5, 5), generator=generator, dtype=torch.complex128, device="cpu"),
        opr_mode_weights_data=torch.ones((2, n_opr_modes), device="cpu"),
        probe_position_x_px=torch.tensor([-1.3, 2.4], device="cpu"),
        probe_position_y_px=torch.tensor([-2.2, 1.1], device="cpu"),
    )
    if isinstance(optimizer, type):
        parameter = getattr(task.reconstructor.parameter_group, {
            "probe_position": "probe_positions", "opr_mode_weight": "opr_mode_weights",
        }.get(parameter_name, parameter_name))
        parameter.optimizer_class = optimizer
        parameter.build_optimizer()
    return task


@pytest.mark.parametrize("parameter_name", ["object", "probe", "probe_position", "real_space_scaling", "opr_mode_weight"])
def test_lsqml_lbfgs_closure_improves_loss_for_each_parameter(parameter_name):
    task = make_task(parameter_name=parameter_name)
    reconstructor = task.reconstructor
    inputs, target = [torch.arange(2)], reconstructor.dataset.patterns
    reconstructor.prepare_optimizer_closures(inputs, target)
    parameter = getattr(reconstructor.parameter_group, {
        "probe_position": "probe_positions", "opr_mode_weight": "opr_mode_weights",
    }.get(parameter_name, parameter_name))
    with torch.no_grad():
        before = reconstructor.compute_optimizer_loss(inputs, target).item()
        intermediates = reconstructor.forward_model.intermediate_variables
        parameter.step_optimizer()
        assert reconstructor.forward_model.intermediate_variables is intermediates
        after = reconstructor.compute_optimizer_loss(inputs, target).item()
    assert after < before


@pytest.mark.parametrize("options_class", [api.LSQMLOptions, api.RPIEOptions, api.BHOptions, api.DMOptions, api.AutodiffPtychographyOptions])
@pytest.mark.parametrize("optimizer_class", [RequiredClosureOptimizer, OptionalClosureOptimizer, ExplicitClosureOptimizer])
def test_reconstructors_supply_closure_during_run(options_class, optimizer_class):
    task = make_task(options_class, optimizer=optimizer_class)
    task.reconstructor.run()
    assert task.real_space_scaling.optimizer.closure_calls > 0
    assert task.real_space_scaling._automatic_optimizer_closure is None


@pytest.mark.parametrize("noise_model", [api.NoiseModels.GAUSSIAN, api.NoiseModels.POISSON])
def test_automatic_objective_honors_mask_and_noise_model(noise_model):
    task = make_task(noise_model=noise_model)
    reconstructor = task.reconstructor
    inputs, target = [torch.arange(2)], reconstructor.dataset.patterns.clone()
    reconstructor.dataset.valid_pixel_mask[0, 0] = False
    original = reconstructor.compute_optimizer_loss(inputs, target)
    target[:, 0, 0] = 1e12
    torch.testing.assert_close(reconstructor.compute_optimizer_loss(inputs, target), original)
    reconstructor.displayed_loss_function = lambda *_: pytest.fail("Display metric is not the objective")
    torch.testing.assert_close(reconstructor.compute_optimizer_loss(inputs, target), original)


def test_custom_closure_survives_automatic_binding():
    task = make_task(optimizer=RequiredClosureOptimizer)
    parameter = task.real_space_scaling
    calls = []

    def custom_closure():
        calls.append(True)
        parameter.optimizer.zero_grad()
        loss = (parameter.data - 1).square().sum()
        loss.backward()
        return loss

    parameter.optimizer_closure = custom_closure
    task.reconstructor.run()
    assert calls
    assert parameter.optimizer_closure is custom_closure
    torch.testing.assert_close(parameter.data, torch.ones_like(parameter.data))


def test_autodiff_closure_includes_regularization_and_preserves_other_gradients():
    task = make_task(api.AutodiffPtychographyOptions, parameter_name="object")
    reconstructor = task.reconstructor
    task.object.options.l2_norm_constraint.enabled = True
    task.object.options.l2_norm_constraint.weight = 0.3
    inputs, target = [torch.arange(2)], reconstructor.dataset.patterns
    task.probe.tensor.data.grad = torch.full_like(task.probe.tensor.data, 123)
    reconstructor.prepare_optimizer_closures(inputs, target)
    actual = task.object._automatic_optimizer_closure()
    prediction = reconstructor.forward_model(*inputs)
    mask = reconstructor.get_constrained_pixel_mask(target)
    expected = reconstructor.loss_function(prediction[mask], target[mask])
    expected = expected + 0.3 * task.object.data.abs().square().mean()
    expected_grad, = torch.autograd.grad(expected, task.object.tensor.data)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(task.object.tensor.data.grad, expected_grad)
    torch.testing.assert_close(task.probe.tensor.data.grad, torch.full_like(task.probe.tensor.data, 123))


def test_compact_object_closure_uses_all_patterns():
    task = make_task(parameter_name="object", optimizer=RequiredClosureOptimizer)
    reconstructor = task.reconstructor
    reconstructor.options.batching_mode = api.BatchingModes.COMPACT
    calls = []
    original = reconstructor.compute_optimizer_loss

    def record(input_data, y_true):
        calls.append(input_data[0].clone())
        return original(input_data, y_true)

    reconstructor.compute_optimizer_loss = record
    reconstructor.run()
    assert calls
    for indices in calls:
        assert sorted(indices.tolist()) == [0, 1]


@pytest.mark.parametrize("options_class", [api.DMOptions, api.RAAROptions])
def test_projection_only_parameters_reject_unused_closure_optimizer(options_class):
    with pytest.raises(ValueError, match="does not step an optimizer for object"):
        make_task(options_class, parameter_name="object")


@pytest.mark.parametrize("options_class", [api.LSQMLOptions, api.RPIEOptions, api.BHOptions, api.AutodiffPtychographyOptions])
@pytest.mark.parametrize("parameter_name", ["object", "probe"])
def test_object_and_probe_steps_use_shared_closure_path(options_class, parameter_name):
    task = make_task(options_class, parameter_name, RequiredClosureOptimizer)
    task.reconstructor.run()
    parameter = getattr(task.reconstructor.parameter_group, parameter_name)
    assert parameter.optimizer.closure_calls > 0


def test_legacy_step_arguments_build_a_closure():
    parameter = make_parameter(torch.tensor([2.0]), RequiredClosureOptimizer)
    with torch.no_grad():
        parameter.step_optimizer(
            forward_model=lambda: parameter.data,
            forward_model_args=[], target_data=torch.tensor([0.0]),
            loss_function=torch.nn.MSELoss(),
        )
    torch.testing.assert_close(parameter.data, torch.tensor([1.6]))
