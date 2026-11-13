"""Autodiff gradients and scheduling with small, independently solved examples."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import ptychi.api as api
from ptychi.api.options.base import ParameterOptions
from ptychi.data_structures.base import ReconstructParameter
from ptychi.data_structures.object import PlanarObject
from ptychi.data_structures.opr_mode_weights import OPRModeWeights
from ptychi.reconstructors.ad_ptychography import AutodiffPtychographyReconstructor
from test_utils import make_task

pytestmark = pytest.mark.usefixtures("numerical_cpu")


@pytest.mark.parametrize("is_complex", [False, True])
@pytest.mark.parametrize("data_as_parameter", [False, True])
def test_set_requires_grad_controls_underlying_storage(is_complex, data_as_parameter):
    data = torch.tensor([1 + 2j]) if is_complex else torch.tensor([1.0])
    parameter = ReconstructParameter(
        data=data,
        is_complex=is_complex,
        data_as_parameter=data_as_parameter,
        options=ParameterOptions(optimizable=True),
        build_optimizer=False,
    )
    stored_tensor = parameter.tensor.data if is_complex else parameter.tensor
    for requires_grad in [False, True]:
        parameter.set_requires_grad(requires_grad)
        assert stored_tensor.requires_grad is requires_grad
        assert parameter.optimizable is True
        assert parameter.options.optimizable is True
        torch.testing.assert_close(parameter.data, data)


@pytest.mark.parametrize(
    "kind,loss,gradient",
    [
        ("l1_norm_constraint", 1.75, [[[0.15 + 0.2j, 0.25j]]]),
        ("l2_norm_constraint", 7.25, [[[1.5 + 2j, 1j]]]),
    ],
)
def test_regularizers_add_complex_gradients_to_existing_data_gradient(kind, loss, gradient):
    options = api.AutodiffPtychographyOptions().object_options
    getattr(options, kind).enabled = True
    getattr(options, kind).weight = 0.5
    obj = PlanarObject(data=torch.tensor([[[3 + 4j, 2j]]]), options=options)
    obj.set_grad(torch.tensor([[[2 - 1j, 3 + 1j]]]))
    recon = AutodiffPtychographyReconstructor.__new__(AutodiffPtychographyReconstructor)
    recon.parameter_group = SimpleNamespace(object=obj)
    recon.current_epoch = 0
    actual = recon.apply_regularizers()
    torch.testing.assert_close(actual, torch.tensor(loss))
    torch.testing.assert_close(
        obj.get_grad(), torch.tensor(gradient) + torch.tensor([[[2 - 1j, 3 + 1j]]])
    )


def test_total_variation_regularizer_value_and_gradient():
    options = api.AutodiffPtychographyOptions().object_options
    options.total_variation.enabled = True
    options.total_variation.weight = 0.5
    obj = PlanarObject(data=torch.tensor([[[0j, 1 + 0j], [0j, 1 + 0j]]]), options=options)
    recon = AutodiffPtychographyReconstructor.__new__(AutodiffPtychographyReconstructor)
    recon.parameter_group = SimpleNamespace(object=obj)
    recon.current_epoch = 0
    loss = recon.apply_regularizers()
    torch.testing.assert_close(loss, torch.tensor(2**0.5 / 2))
    torch.testing.assert_close(obj.get_grad(), torch.tensor([[[-1 + 0j, 1], [-1, 1]]]) / 2**1.5)


@pytest.mark.parametrize(
    "epoch,expected", [(0, False), (1, True), (2, False), (3, True), (4, False)]
)
def test_autodiff_schedule_controls_object_and_slice_spacing_gradients(epoch, expected):
    options = api.AutodiffPtychographyOptions()
    options.object_options.optimization_plan.start = 1
    options.object_options.optimization_plan.stop = 4
    options.object_options.optimization_plan.stride = 2
    options.object_options.slice_spacing_options.optimizable = True
    options.object_options.slice_spacing_options.optimization_plan.start = 1
    options.object_options.slice_spacing_options.optimization_plan.stop = 4
    options.object_options.slice_spacing_options.optimization_plan.stride = 2
    task = make_task(options, n_slices=2)
    task.reconstructor.current_epoch = epoch
    task.reconstructor.set_requires_grad()
    assert task.object.tensor.data.requires_grad is expected
    assert task.object.slice_spacings.tensor.requires_grad is expected
    assert task.probe_positions.tensor.requires_grad is False


@pytest.mark.parametrize(
    "intensity,eigen,expected",
    [
        (False, True, [[0.0, 2.0, 3.0], [0.0, 5.0, 6.0]]),
        (True, False, [[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        (True, True, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ],
)
def test_opr_gradient_hook_freezes_disabled_weight_components(intensity, eigen, expected):
    options = api.AutodiffPtychographyOptions().opr_mode_weight_options
    options.optimizable = True
    options.optimize_intensity_variation = intensity
    options.optimize_eigenmode_weights = eigen
    weights = OPRModeWeights(data=torch.ones((2, 3)), options=options)
    weights.set_grad(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    recon = AutodiffPtychographyReconstructor.__new__(AutodiffPtychographyReconstructor)
    recon.parameter_group = SimpleNamespace(opr_mode_weights=weights)
    recon.current_epoch = 0
    recon.forward_model = SimpleNamespace(post_differentiation_hook=Mock())
    recon.run_post_differentiation_hooks([torch.tensor([0, 1])], torch.ones((2, 2, 2)))
    torch.testing.assert_close(weights.get_grad(), torch.tensor(expected))
    recon.forward_model.post_differentiation_hook.assert_called_once()


@pytest.mark.parametrize(
    "norm,expected_object,expected_probe", [(None, 0.125, 1.0), ("ortho", 0.5, 4.0)]
)
def test_autodiff_fft_and_minibatch_gradient_scaling(norm, expected_object, expected_probe):
    task = make_task(api.AutodiffPtychographyOptions())
    model = task.reconstructor.forward_model
    model.free_space_propagator.norm = norm
    task.object.set_grad(torch.ones_like(task.object.data))
    task.probe.set_grad(torch.ones_like(task.probe.data))
    model.post_differentiation_hook(torch.tensor([0, 1]), torch.ones((2, 2, 2)))
    # Probe maximum intensity=16; two patterns of four pixels each.
    torch.testing.assert_close(task.object.get_grad(), torch.full((1, 6, 6), expected_object + 0j))
    torch.testing.assert_close(task.probe.get_grad(), torch.full((1, 1, 2, 2), expected_probe + 0j))


def test_autodiff_minibatch_applies_simultaneous_object_probe_sgd_updates():
    options = api.AutodiffPtychographyOptions()
    for param in (options.object_options, options.probe_options):
        param.optimizer = api.Optimizers.SGD
        param.step_size = 0.1
    task = make_task(options)
    recon = task.reconstructor
    with torch.no_grad():
        task.probe.set_data(torch.ones((1, 1, 2, 2), dtype=torch.complex128))
    # A constant 2x2 exit wave has intensity 16 at DC. The measured amplitude
    # is 2, so each illuminated object pixel and probe pixel has gradient 1.
    recon.run_minibatch([torch.tensor([0])], torch.tensor([[[4.0, 0.0], [0.0, 0.0]]]))
    expected_object = torch.ones((1, 6, 6), dtype=torch.complex128)
    # The task places the first patch at rows 2:4, columns 1:3.
    expected_object[0, 2:4, 1:3] = 0.9
    torch.testing.assert_close(task.object.data, expected_object)
    torch.testing.assert_close(task.probe.data, torch.full((1, 1, 2, 2), 0.9 + 0j))
    assert task.object.tensor.data.grad is None
    assert task.probe.tensor.data.grad is None


def test_slice_spacing_gradient_matches_central_difference():
    options = api.AutodiffPtychographyOptions()
    options.object_options.slice_spacing_options.optimizable = True
    task = make_task(options, n_slices=2)
    with torch.no_grad():
        obj = task.object.data
        obj[1, 2:4, 1:4] = torch.tensor([[1.0, 0.5, 1.5], [2.0, 0.7, 0.3]])
        task.object.set_data(obj)
    indices = torch.tensor([0])
    spacing = task.object.slice_spacings.tensor
    # A weighted intensity sum is sensitive to inter-slice propagation.
    weights = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    loss = (task.reconstructor.forward_model(indices) * weights).sum()
    analytic = torch.autograd.grad(loss, spacing)[0]
    initial = spacing.detach().clone()
    h = 1e-7
    values = []
    with torch.no_grad():
        for delta in [h, -h]:
            task.object.slice_spacings.set_data(initial + delta)
            values.append((task.reconstructor.forward_model(indices) * weights).sum())
    expected = (values[0] - values[1]) / (2 * h)
    assert expected.abs() > 1e-3
    torch.testing.assert_close(analytic, expected.reshape(1), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("frozen", ["object", "probe"])
def test_scheduled_parameter_stays_fixed_during_autodiff_minibatch(frozen):
    options = api.AutodiffPtychographyOptions()
    getattr(options, frozen + "_options").optimization_plan.start = 2
    task = make_task(options)
    before = getattr(task, frozen).data.clone()
    task.reconstructor.run_minibatch([torch.tensor([0])], torch.ones((1, 2, 2)))
    torch.testing.assert_close(getattr(task, frozen).data, before)
