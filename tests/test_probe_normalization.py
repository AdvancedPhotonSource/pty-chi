import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ptychi.api as api
import ptychi.maths as pmath
import ptychi.propagate as propagation
import ptychi.utils as utils
from ptychi.api.task import PtychographyTask
from ptychi.api.options import base as obase
from ptychi.data_structures.object import PlanarObject
from ptychi.data_structures.probe import Probe
from ptychi.data_structures.probe_positions import ProbePositions
from ptychi.reconstructors.bh import BHReconstructor


@pytest.fixture(autouse=True)
def cpu_defaults():
    device = torch.get_default_device()
    dtype = torch.get_default_dtype()
    complex_dtype = utils.get_default_complex_dtype()
    fft_precision = pmath.get_use_double_precision_for_fft()
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    utils.set_default_complex_dtype(torch.complex64)
    yield
    pmath.set_use_double_precision_for_fft(fft_precision)
    utils.set_default_complex_dtype(complex_dtype)
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)


@pytest.fixture(params=["fourier", "angular", "fresnel", "fraunhofer"])
def propagator(request):
    if request.param == "fourier":
        return propagation.FourierPropagator()
    parameters = propagation.WavefieldPropagatorParameters.create_simple(
        wavelength_m=1e-10, width_px=12, height_px=8,
        pixel_width_m=2e-8, pixel_height_m=3e-8, propagation_distance_m=-0.0043,
    )
    cls = {
        "angular": propagation.AngularSpectrumPropagator,
        "fresnel": propagation.FresnelTransformPropagator,
        "fraunhofer": propagation.FraunhoferPropagator,
    }[request.param]
    return cls(parameters)


@pytest.mark.parametrize("as_numpy", [False, True])
@pytest.mark.parametrize("layout", ["modes", "single_opr", "weighted_opr"])
def test_rescale_probe_matches_detector_power(propagator, as_numpy, layout):
    probe = torch.randn(2, 8, 12, dtype=torch.complex64)
    weights = None
    if layout == "single_opr":
        probe = probe[None]
        composed = probe[0]
    elif layout == "weighted_opr":
        probe = probe[:, None]
        weights = torch.tensor([[0.8, 0.2], [0.8, 0.2]])
        composed = (probe * weights.mean(0)[:, None, None, None]).sum(0)
    else:
        composed = probe
    patterns = propagator(composed).abs().square().sum(0)[None]
    if as_numpy:
        probe, patterns = probe.numpy(), patterns.numpy()
    scaled = utils.rescale_probe(probe * 0.25, patterns, weights)
    np.testing.assert_allclose(utils.to_numpy(scaled), utils.to_numpy(probe), rtol=2e-6)
    assert type(scaled) is type(probe)


def test_rescale_probe_uses_evanescent_filter():
    parameters = propagation.WavefieldPropagatorParameters.create_simple(
        wavelength_m=1e-9, width_px=12, height_px=8,
        pixel_width_m=0.4e-9, pixel_height_m=0.4e-9, propagation_distance_m=5e-9,
    )
    propagator = propagation.AngularSpectrumPropagator(parameters)
    probe = torch.randn(1, 8, 12, dtype=torch.complex64)
    patterns = propagator(probe).abs().square()
    assert patterns.sum() < probe.abs().square().sum()
    torch.testing.assert_close(utils.rescale_probe(probe, patterns, propagator=propagator), probe)


def test_detector_power_constraint_preserves_transmission(propagator):
    probe_data = torch.randn(1, 2, 8, 12, dtype=torch.complex64)
    detector_power = propagator(probe_data[0]).abs().square().sum().item()
    options = obase.ProbeOptions()
    options.power_constraint.probe_power = detector_power
    probe = Probe(data=probe_data, options=options)
    obj = PlanarObject(data=torch.ones(1, 8, 12, dtype=torch.complex64), options=obase.ObjectOptions())
    with torch.no_grad():
        probe.constrain_probe_power(obj)
    torch.testing.assert_close(probe.data, probe_data)
    torch.testing.assert_close(obj.data, torch.ones_like(obj.data))


def test_bh_gradient_matches_autograd_for_every_geometry(propagator):
    wavefield = torch.randn(2, 3, 8, 12, dtype=torch.complex64, requires_grad=True)
    field = propagator(wavefield)
    data = torch.rand(2, 1, 8, 12) + 0.5
    loss = (field.abs().square().sum(1, keepdim=True).sqrt() - data).square().sum()
    expected = torch.autograd.grad(loss, wavefield)[0]
    reconstructor = object.__new__(BHReconstructor)
    reconstructor.eps = 0.0
    reconstructor.current_constrained_pixel_mask = torch.ones_like(data, dtype=torch.bool)
    reconstructor.forward_model = SimpleNamespace(free_space_propagator=propagator)
    actual = reconstructor.gradientF(field.detach(), data)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_autodiff_gradient_scaling_matches_amplitude_projection(propagator):
    options = api.AutodiffPtychographyOptions()
    options.reconstructor_options.default_device = api.Devices.CPU
    options.data_options.fft_shift = False
    options.object_options.pixel_size_m = 2e-8
    options.probe_options.optimizable = False
    options.probe_position_options.optimizable = False
    task = PtychographyTask(
        options, diffraction_data=torch.ones(1, 8, 12),
        object_data=torch.ones(1, 8, 12, dtype=torch.complex64),
        probe_data=torch.ones(1, 1, 8, 12, dtype=torch.complex64),
        probe_position_x_px=torch.zeros(1), probe_position_y_px=torch.zeros(1),
    )
    model = task.reconstructor.forward_model
    model.free_space_propagator = propagator
    indices = torch.tensor([0])
    prediction = model(indices)
    measured = prediction.detach() * 0.81
    loss = task.reconstructor.loss_function(prediction, measured)
    loss.backward()
    model.post_differentiation_hook(indices, measured)
    # Amplitude residual is 0.1 * propagated field. The unitary adjoint brings
    # it back to 0.1 * exit wave; differentiating squared error supplies 2.
    torch.testing.assert_close(
        task.object.get_grad(), torch.full_like(task.object.data, 0.2),
        rtol=2e-5, atol=2e-6,
    )


def test_preconditioner_is_independent_of_propagation_geometry(propagator):
    probe_data = torch.randn(1, 2, 8, 12, dtype=torch.complex64)
    probe = Probe(data=probe_data, options=obase.ProbeOptions())
    obj = PlanarObject(data=torch.ones(1, 24, 28, dtype=torch.complex64), options=obase.ObjectOptions())
    positions = ProbePositions(data=torch.zeros(1, 2), options=obase.ProbePositionOptions())
    patterns = propagator(probe_data[0]).abs().square().sum(0)[None]
    # Legacy far-field initialization, before converting its probe to photon
    # amplitudes. Its illumination map must acquire the same factor of H*W.
    old_probe = probe_data / math.sqrt(96)
    old_patterns = torch.fft.fft2(old_probe[0]).abs().square().sum(0)[None]
    old_renormalization = torch.sqrt(96 / old_patterns.sum())
    reference_probe = Probe(
        data=old_probe / (math.sqrt(96) * 2 * old_renormalization),
        options=obase.ProbeOptions(),
    )
    expected = 96 * obj.calculate_illumination_map(reference_probe, positions, use_all_modes=True)
    obj.initialize_preconditioner(probe, positions, patterns, use_all_modes=True)
    torch.testing.assert_close(obj.preconditioner, expected)
    obj.preconditioner = None
    obj.update_preconditioner(probe, positions, patterns, use_all_modes=True)
    illumination = obj.calculate_illumination_map(probe, positions, use_all_modes=True)
    torch.testing.assert_close(obj.preconditioner, (expected + illumination) / 2)


@pytest.mark.parametrize(
    "distance,pixel_size,propagator_type",
    [
        (float("inf"), 2e-7, propagation.FourierPropagator),
        (0.0043, 3e-7, propagation.AngularSpectrumPropagator),
        (0.0043, 1e-9, propagation.FresnelTransformPropagator),
    ],
)
def test_reconstructor_preconditioner_uses_photon_counts(distance, pixel_size, propagator_type):
    options = api.EPIEOptions()
    options.reconstructor_options.default_device = api.Devices.CPU
    options.data_options.fft_shift = False
    options.data_options.wavelength_m = 1e-10
    options.data_options.free_space_propagation_distance_m = distance
    options.object_options.pixel_size_m = pixel_size
    options.probe_position_options.optimizable = False
    task = PtychographyTask(
        options, diffraction_data=torch.ones(1, 8, 12),
        object_data=torch.ones(1, 24, 28, dtype=torch.complex64),
        probe_data=torch.randn(1, 1, 8, 12, dtype=torch.complex64),
        probe_position_x_px=torch.zeros(1), probe_position_y_px=torch.zeros(1),
    )
    model = task.reconstructor.forward_model
    assert isinstance(model.free_space_propagator, propagator_type)
    with torch.no_grad():
        patterns = model(torch.tensor([0]))
        task.reconstructor.dataset.patterns.copy_(patterns)
        sampling = model.get_object_sampling_parameters()
        task.object.update_preconditioner(
            task.probe, task.probe_positions, patterns, sampling=sampling,
        )
        expected = task.object.preconditioner.clone()
        task.object.preconditioner = None
        task.reconstructor.update_preconditioners()
    torch.testing.assert_close(task.object.preconditioner, expected)
