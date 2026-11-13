"""Shared reconstruction features tested independently of optimization epochs."""

from types import SimpleNamespace

import pytest
import torch

import ptychi.api as api
from ptychi.api.options.base import OptimizationPlan
from ptychi.data_structures.object import PlanarObject, DIPPlanarObject
from ptychi.data_structures.probe_positions import ProbePositions
from ptychi.forward_models import PlanarPtychographyForwardModel
from test_utils import make_task

pytestmark = pytest.mark.usefixtures("numerical_cpu")


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((2, 2), [[[0.0, 3.0], [12.0, 15.0]]]),
        ((3, 3), [[[0.0, 1.0, 3.0], [4.0, 5.0, 7.0], [12.0, 13.0, 15.0]]]),
    ],
)
def test_detector_crop_keeps_unshifted_fourier_coordinates(shape, expected):
    model = SimpleNamespace(detector_size=shape)
    result = PlanarPtychographyForwardModel.conform_to_detector_size(
        model, torch.arange(16.0).reshape(1, 4, 4)
    )
    torch.testing.assert_close(result, torch.tensor(expected))


@pytest.mark.parametrize("shape", [None, (4, 4)])
def test_detector_same_size_preserves_tensor(shape):
    data = torch.arange(16.0).reshape(1, 4, 4)
    assert (
        PlanarPtychographyForwardModel.conform_to_detector_size(
            SimpleNamespace(detector_size=shape), data
        )
        is data
    )


def test_detector_larger_than_probe_is_rejected():
    with pytest.raises(ValueError, match="larger"):
        PlanarPtychographyForwardModel.conform_to_detector_size(
            SimpleNamespace(detector_size=(5, 4)), torch.zeros((1, 4, 4))
        )


def test_detector_blur_has_normalized_gaussian_impulse_response():
    # sigma=.2 gives a 3x3 kernel, whose axis weights are [exp(-12.5),1,exp(-12.5)].
    impulse = torch.zeros((2, 7, 7))
    impulse[0, 3, 3] = 1
    impulse[1, 3, 3] = 2
    actual = PlanarPtychographyForwardModel.blur_diffraction_patterns(
        SimpleNamespace(diffraction_pattern_blur_sigma=0.2), impulse
    )
    q = 3.726653172078671e-6
    kernel = torch.tensor([[q * q, q, q * q], [q, 1.0, q], [q * q, q, q * q]]) / (1 + 2 * q) ** 2
    expected = torch.zeros_like(impulse)
    expected[0, 2:5, 2:5] = kernel
    expected[1, 2:5, 2:5] = 2 * kernel
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("n_slices", [1, 3])
@pytest.mark.parametrize("n_modes", [1, 2])
@torch.no_grad()
def test_multislice_forward_and_low_memory_match_inline_diffraction(n_slices, n_modes):
    task = make_task(n_slices=n_slices, n_modes=n_modes)
    # Constant slices multiply the probe by 2**n_slices; spacing zero isolates
    # slice ordering/modulation from the propagator's separately tested kernel.
    task.object.set_data(torch.full((n_slices, 6, 6), 2 + 0j))
    if n_slices > 1:
        task.object.slice_spacings.set_data(torch.zeros(n_slices - 1))
    model = task.reconstructor.forward_model
    indices = torch.tensor([0, 1])
    # FFT([[1,2i],[3,4i]]) = [[4+6i,4-6i],[-2-2i,-2+2i]].
    expected = torch.tensor([[[52.0, 52.0], [8.0, 8.0]]]).repeat(2, 1, 1)
    expected *= n_modes * 4**n_slices
    torch.testing.assert_close(model(indices), expected)
    torch.testing.assert_close(model.forward_low_memory(indices), expected)


@pytest.mark.parametrize("weight,expected", [(0.0, [3 + 4j, 0j]), (0.5, [2.85 + 3.8j, 0j])])
@torch.no_grad()
def test_object_l1_constraint_complex_subgradient(weight, expected):
    options = api.LSQMLOptions().object_options
    options.l1_norm_constraint.weight = weight
    obj = PlanarObject(data=torch.tensor([[[3 + 4j, 0j]]]), options=options)
    obj.constrain_l1_norm()
    torch.testing.assert_close(obj.data, torch.tensor([[expected]]))


@torch.no_grad()
def test_object_l2_constraint_complex_gradient():
    options = api.LSQMLOptions().object_options
    options.l2_norm_constraint.weight = 0.25
    obj = PlanarObject(data=torch.tensor([[[3 + 4j, 2j]]]), options=options)
    obj.constrain_l2_norm()
    torch.testing.assert_close(obj.data, torch.tensor([[[2.25 + 3j, 1.5j]]]))


@pytest.mark.parametrize("unwrap", [False, True])
@torch.no_grad()
def test_multislice_regularization_constant_amplitude_and_phase(unwrap):
    options = api.LSQMLOptions().object_options
    options.slice_spacings_m = [1e-5]
    options.multislice_regularization.weight = 1.0
    options.multislice_regularization.unwrap_phase = unwrap
    obj = PlanarObject(data=torch.full((2, 4, 4), 2 + 0j), options=options)
    obj.preconditioner = torch.ones((4, 4))
    obj.regularize_multislice()
    # DC passes the Fourier filter; amplitude relaxation maps 2 -> 2*(1+.9-1)=1.8.
    torch.testing.assert_close(obj.data, torch.full((2, 4, 4), 1.8 + 0j))


@pytest.mark.parametrize("weight", [0.0, 1.0])
@torch.no_grad()
def test_multislice_regularization_damps_slice_difference(weight):
    options = api.LSQMLOptions().object_options
    options.slice_spacings_m = [1e-5]
    options.multislice_regularization.weight = weight
    options.multislice_regularization.unwrap_phase = False
    data = torch.tensor([1.0, 2.0])[:, None, None].expand(2, 2, 2).to(torch.complex128)
    obj = PlanarObject(data=data, options=options)
    obj.preconditioner = torch.ones((2, 2))
    obj.regularize_multislice()
    # Slice-frequency 1/2 has filter 1 - 2*atan(250)/pi = 0.002546465508189...
    expected = [1.0, 2.0] if weight == 0 else [1.448854090521314, 0.902291818957372]
    torch.testing.assert_close(
        obj.data, torch.tensor(expected)[:, None, None].expand(2, 2, 2).to(torch.complex128)
    )


@pytest.mark.parametrize(
    "start,stop,stride,expected",
    [
        (0, None, 1, [True] * 6),
        (2, None, 1, [False, False, True, True, True, True]),
        (1, 5, 2, [False, True, False, True, False, False]),
    ],
)
def test_optimization_plan_boundaries(start, stop, stride, expected):
    plan = OptimizationPlan(start=start, stop=stop, stride=stride)
    assert [plan.is_enabled(epoch) for epoch in range(6)] == expected


@pytest.mark.parametrize(
    "flexibility,expected",
    [
        (0.0, [[3.0, 0.0], [-1.0, 2.0], [0.0, -2.0]]),
        (0.5, [[2.5, 0.0], [-1.5, 1.0], [0.0, -2.0]]),
        (1.0, [[2.0, 0.0], [-2.0, 0.0], [0.0, -2.0]]),
    ],
)
@torch.no_grad()
def test_affine_position_constraint_interpolates_to_transformed_initial_positions(
    flexibility,
    expected,
):
    options = api.LSQMLOptions().probe_position_options
    options.affine_transform_constraint.override_update_flexibility = flexibility
    pos = ProbePositions(data=torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]), options=options)
    pos.set_data(torch.tensor([[3.0, 0.0], [-1.0, 2.0], [0.0, -2.0]]))
    pos.affine_transform_matrix = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    pos.apply_affine_transform_constraint()
    torch.testing.assert_close(pos.data, torch.tensor(expected))


@torch.no_grad()
def test_affine_fit_recovers_scale_in_yx_coordinates():
    options = api.LSQMLOptions().probe_position_options
    options.affine_transform_constraint.degrees_of_freedom = [api.AffineDegreesOfFreedom.SCALE]
    initial = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    pos = ProbePositions(data=initial.clone(), options=options)
    pos.set_data(2 * initial)
    pos.update_affine_transform_matrix()
    torch.testing.assert_close(
        pos.affine_transform_matrix, torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    )


@pytest.mark.parametrize("constrain", [False, True])
@pytest.mark.parametrize("as_tuple", [False, True])
def test_dip_output_conversion_and_gradient(constrain, as_tuple):
    options = api.AutodiffPtychographyOptions().object_options
    options.experimental.deep_image_prior_options.constrain_object_outside_network = constrain
    # Only the network output transform is under test; no trained network is needed.
    obj = SimpleNamespace(n_slices=1, lateral_shape=(1, 2), options=options)
    values = torch.tensor([[[[0.0, 1.0]], [[0.0, torch.pi / 2]]]], requires_grad=True)
    output = (values[:, :1], values[:, 1:]) if as_tuple else values
    actual, _, _ = DIPPlanarObject.process_net_output(obj, output)
    expected = [[[0.5 + 0j, 0.7310585786300049j]]] if constrain else [[[0j, 1j]]]
    torch.testing.assert_close(torch.view_as_complex(actual), torch.tensor(expected))
    actual.square().sum().backward()
    expected_grad = (
        [[[[0.25, 0.28746968091443026]], [[0.0, 0.0]]]]
        if constrain
        else [[[[0.0, 2.0]], [[0.0, 0.0]]]]
    )
    torch.testing.assert_close(values.grad, torch.tensor(expected_grad), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("parameter_name", ["object", "probe"])
@pytest.mark.parametrize("residual", [False, True])
def test_dip_generation_backpropagates_to_network_parameters(parameter_name, residual, monkeypatch):
    import ptychi.maps as maps
    from ptychi.data_structures.probe import DIPProbe

    class ConstantNetwork(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.output = torch.nn.Parameter(torch.tensor([2.0, 0.0]).reshape(1, 2, 1, 1))

        def forward(self, inputs):
            return self.output.expand(inputs.shape[0], 2, *inputs.shape[-2:])

    monkeypatch.setattr(maps, "get_nn_model_by_enum", lambda model: ConstantNetwork)
    options = getattr(api.AutodiffPtychographyOptions(), parameter_name + "_options")
    options.optimizer = api.Optimizers.SGD
    options.step_size = 0.1
    dip = options.experimental.deep_image_prior_options
    dip.enabled = True
    dip.residual_generation = residual
    if parameter_name == "object":
        dip.constrain_object_outside_network = False
    shape = (1, 2, 2) if parameter_name == "object" else (1, 1, 2, 2)
    cls = DIPPlanarObject if parameter_name == "object" else DIPProbe
    parameter = cls(data=torch.full(shape, 3 + 1j), options=options)
    generated = parameter.generate()
    torch.testing.assert_close(generated, torch.full(shape, 5 + 1j if residual else 2 + 0j))
    generated.real.sum().backward()
    torch.testing.assert_close(
        parameter.model.output.grad, torch.tensor([4.0, 0.0]).reshape(1, 2, 1, 1)
    )
    parameter.optimizer.step()
    torch.testing.assert_close(
        parameter.generate(), torch.full(shape, 4.6 + 1j if residual else 1.6 + 0j)
    )
