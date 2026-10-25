"""Small CPU tests for GM-rPIE/GM-MAGPIE; no external reconstruction data."""

import json
import math
import random
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest
import torch
from pydantic import ValidationError

import ptychi.image_proc as ip
import ptychi.maths as pmath
from ptychi import api, utils
from ptychi.api.options.magpie import (
    MAGPIECapabilityError,
    MAGPIEInitializationError,
    MAGPIEMultimodeError,
    MAGPIEMultisliceError,
    MAGPIEOPRError,
    MAGPIEReconstructorOptions,
)
from ptychi.api.task import PtychographyTask
from ptychi.data_structures.object import PlanarObject
from ptychi.data_structures.opr_mode_weights import OPRModeWeights
from ptychi.data_structures.parameter_group import PlanarPtychographyParameterGroup
from ptychi.data_structures.probe import Probe
from ptychi.data_structures.probe_positions import ProbePositions
from ptychi.io_handles import PtychographyDataset
from ptychi.reconstructors.magpie import (
    MAGPIEReconstructor,
    _prolong_2x,
    _restrict_2x,
    _safe_divide,
    aligned_geom_mean,
)
from ptychi.timing import timer_utils


@pytest.fixture(autouse=True)
def cpu_defaults():
    """Task construction sets process-wide defaults; restore them for other tests."""
    device, dtype = torch.get_default_device(), torch.get_default_dtype()
    complex_dtype = utils.get_default_complex_dtype()
    fft_precision = pmath.get_use_double_precision_for_fft()
    nondeterministic = pmath.get_allow_nondeterministic_algorithms()
    torch_rng, numpy_rng, python_rng = (
        torch.get_rng_state(),
        np.random.get_state(),
        random.getstate(),
    )
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    utils.set_default_complex_dtype(complex_dtype)
    pmath.set_use_double_precision_for_fft(fft_precision)
    pmath.set_allow_nondeterministic_algorithms(nondeterministic)
    torch.set_rng_state(torch_rng)
    np.random.set_state(numpy_rng)
    random.setstate(python_rng)


def _complex_random(shape, seed=17, dtype=torch.complex64):
    generator = torch.Generator().manual_seed(seed)
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    return torch.complex(
        torch.randn(shape, generator=generator, dtype=real_dtype),
        torch.randn(shape, generator=generator, dtype=real_dtype),
    )


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_geometric_mean_uses_short_phase_arc(dtype):
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    phases = torch.tensor([math.pi - 0.2, -math.pi + 0.2], dtype=real_dtype)
    a = torch.polar(torch.full_like(phases, 4), phases)
    b = torch.polar(torch.full_like(phases, 9), phases.flip(0))

    result = aligned_geom_mean(a, b)

    assert result.dtype == dtype
    torch.testing.assert_close(result, torch.full((2,), -6 + 0j, dtype=dtype))
    torch.testing.assert_close(result.square(), a * b)
    torch.testing.assert_close(aligned_geom_mean(a, a), a)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_geometric_mean_zero_broadcast_and_antipodal_branch(dtype):
    a = torch.tensor([[0], [1], [1j]], dtype=dtype)
    b = torch.tensor([0, -1, 4], dtype=dtype)
    result = aligned_geom_mean(a, b)
    root_half = math.sqrt(0.5)
    expected = torch.tensor(
        [[0, 0, 0], [0, 1j, 2], [0, -root_half + 1j * root_half, 2 * root_half * (1 + 1j)]],
        dtype=dtype,
    )
    torch.testing.assert_close(result, expected)
    assert torch.isfinite(result).all()
    # The negative signed-zero side of the antipodal cut has the same +i tie break.
    torch.testing.assert_close(
        aligned_geom_mean(
            torch.ones(1, dtype=dtype), torch.tensor([complex(-1, -0.0)], dtype=dtype)
        ),
        torch.tensor([1j], dtype=dtype),
    )


def test_geometric_mean_promotes_precision():
    result = aligned_geom_mean(
        torch.tensor([1j], dtype=torch.complex64),
        torch.tensor([9j], dtype=torch.complex128),
    )
    assert result.dtype == torch.complex128
    torch.testing.assert_close(result, torch.tensor([3j], dtype=torch.complex128))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_transfer_operators_and_safe_division(dtype):
    values = torch.arange(24).reshape(1, 1, 4, 6).to(dtype) * (1 + 2j)
    expected = torch.tensor([[[[3.5, 5.5, 7.5], [15.5, 17.5, 19.5]]]], dtype=dtype) * (1 + 2j)
    coarse = _restrict_2x(values)
    torch.testing.assert_close(coarse, expected)
    fine = _prolong_2x(coarse)
    for row in range(4):
        for column in range(6):
            torch.testing.assert_close(fine[..., row, column], expected[..., row // 2, column // 2])
    torch.testing.assert_close(_restrict_2x(fine), coarse)
    assert coarse.dtype == fine.dtype == dtype
    quotient = _safe_divide(torch.tensor([1 + 2j, 3 + 4j], dtype=dtype), torch.tensor([0.0, 2.0]))
    torch.testing.assert_close(quotient, torch.tensor([0, 1.5 + 2j], dtype=dtype))
    assert torch.isfinite(quotient).all()


def _endpoint(levels):
    reconstructor = object.__new__(MAGPIEReconstructor)
    reconstructor.num_levels = levels
    return reconstructor


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_one_level_object_proposal_is_minibatch_normalized_rpie(dtype):
    factor = _complex_random((3, 1, 4, 4), seed=5, dtype=dtype)
    counterpart = _complex_random(factor.shape, seed=6, dtype=dtype)
    counterpart *= torch.tensor([1, 2, 4], dtype=factor.real.dtype)[:, None, None, None]
    target = _complex_random(factor.shape, seed=7, dtype=dtype)
    alpha = 0.2

    actual = _endpoint(1)._object_proposal(factor, counterpart, target, alpha)

    power = counterpart.abs().square()
    denominator = (1 - alpha) * power + alpha * power.amax()
    expected = factor + counterpart.conj() * (target - counterpart * factor) / denominator
    torch.testing.assert_close(actual, expected)
    per_scan = (1 - alpha) * power + alpha * power.amax(dim=(-2, -1), keepdim=True)
    assert not torch.allclose(
        actual, factor + counterpart.conj() * (target - counterpart * factor) / per_scan
    )
    assert actual.dtype == dtype


@pytest.mark.parametrize("levels", [1, 3])
def test_zero_counterpart_does_not_change_unobserved_factor(levels):
    factor = _complex_random((2, 1, 4, 4))
    result = _endpoint(levels)._object_proposal(factor, torch.zeros_like(factor), factor * 2, 0.2)
    torch.testing.assert_close(result, factor)
    assert torch.isfinite(result).all()


class _CapturedProposal(MAGPIEReconstructor):
    def __init__(self):
        self.num_levels = 2
        self.steps = []

    def _proximal_step(self, z, q, target, q_power, regularizer):
        self.steps.append((z, q, target, q_power, regularizer))
        return super()._proximal_step(z, q, target, q_power, regularizer)


def test_weighted_restriction_preserves_galerkin_normal_equation():
    proposal = _CapturedProposal()
    z = _complex_random((2, 1, 4, 4), seed=11, dtype=torch.complex128)
    q = _complex_random(z.shape, seed=12, dtype=torch.complex128)
    target = _complex_random(z.shape, seed=13, dtype=torch.complex128)
    alpha = 0.35
    proposal._object_proposal(z, q, target, alpha)
    z_c, q_c, target_c, power_c, regularizer_c = proposal.steps[0]

    # Independent 2x2 block sums verify both the coarse gradient and curvature.
    for batch in range(2):
        for row in range(2):
            for column in range(2):
                block = (batch, 0, slice(2 * row, 2 * row + 2), slice(2 * column, 2 * column + 2))
                at = (batch, 0, row, column)
                power = q[block].abs().square()
                coherence = q[block].mean().abs().square() / power.mean()
                expected_gradient = (
                    coherence * (q[block].conj() * (target[block] - q[block] * z[block])).mean()
                )
                expected_curvature = (
                    coherence * ((1 - alpha) * power + alpha * q.abs().square().max()).mean()
                )
                torch.testing.assert_close(
                    q_c[at].conj() * (target_c[at] - q_c[at] * z_c[at]), expected_gradient
                )
                torch.testing.assert_close(power_c[at] + regularizer_c[at], expected_curvature)


def _options(
    levels=1,
    dtype=torch.complex64,
    refine_positions=False,
    method=api.PatchInterpolationMethods.FOURIER,
):
    options = api.MAGPIEOptions()
    options.data_options.fft_shift = False
    options.reconstructor_options.default_device = api.Devices.CPU
    options.reconstructor_options.default_dtype = (
        api.Dtypes.FLOAT64 if dtype == torch.complex128 else api.Dtypes.FLOAT32
    )
    options.reconstructor_options.random_seed = 42
    options.reconstructor_options.num_epochs = 1
    options.reconstructor_options.batch_size = 2
    options.reconstructor_options.multigrid_levels = levels
    options.reconstructor_options.allow_nondeterministic_algorithms = False
    options.object_options.alpha = 0.2
    options.object_options.patch_interpolation_method = method
    options.object_options.remove_object_probe_ambiguity.enabled = False
    options.probe_options.alpha = 0.4
    options.probe_position_options.optimizable = refine_positions
    options.probe_position_options.step_size = 0.05
    options.probe_position_options.correction_options.clip_update_magnitude_by_mad = False
    return options


def _tiny_data(dtype=torch.complex64):
    """Three overlapping 4x4 frames, including distinct fractional shifts."""
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    yy, xx = torch.meshgrid(
        torch.arange(17, dtype=real_dtype), torch.arange(17, dtype=real_dtype), indexing="ij"
    )
    obj = torch.polar(0.8 + 0.1 * torch.cos(xx / 3), 0.2 * torch.sin(yy / 2) + 0.1 * xx)
    py, px = torch.meshgrid(
        torch.arange(4, dtype=real_dtype) - 1.5,
        torch.arange(4, dtype=real_dtype) - 1.5,
        indexing="ij",
    )
    probe = torch.polar(torch.exp(-(px.square() + py.square()) / 6), 0.1 * px * py)
    positions = torch.tensor([[-1.23, -1.12], [0.11, 0.31], [1.12, 0.81]], dtype=real_dtype)
    return {
        "diffraction_data": torch.ones((3, 4, 4), dtype=real_dtype),
        "object_data": obj[None],
        "probe_data": probe[None, None],
        "probe_position_y_px": positions[:, 0],
        "probe_position_x_px": positions[:, 1],
    }


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_probe_proposal_is_per_scan_rpie(dtype):
    task = PtychographyTask(_options(dtype=dtype), **_tiny_data(dtype))
    obj = _complex_random((3, 1, 4, 4), seed=19, dtype=dtype)
    obj *= torch.tensor([1, 2, 4], dtype=obj.real.dtype)[:, None, None, None]
    probe = _complex_random(obj.shape, seed=20, dtype=dtype)
    residual = _complex_random(obj.shape, seed=21, dtype=dtype)
    alpha = task.options.probe_options.alpha
    power = obj.abs().square()
    expected = probe + obj.conj() * residual / (
        (1 - alpha) * power + alpha * power.amax(dim=(-2, -1), keepdim=True)
    )
    torch.testing.assert_close(task.reconstructor._probe_proposal(obj, probe, residual), expected)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_object_synthesis_is_counterpart_weighted_overlap_average(dtype):
    task = PtychographyTask(_options(dtype=dtype), **_tiny_data(dtype))
    indices = torch.tensor([2, 0, 1])
    old = _complex_random((3, 1, 4, 4), seed=30, dtype=dtype)
    local = old + _complex_random(old.shape, seed=31, dtype=dtype)
    counterpart = _complex_random(old.shape, seed=32, dtype=dtype)
    counterpart[0, :, :2] = 0
    old_object = task.object.data.clone()
    result = task.reconstructor._synthesize_object_update(indices, old, local, counterpart)

    numerator = torch.zeros_like(task.object.data)
    denominator = torch.zeros_like(task.object.data.real)
    centers = task.probe_positions.data[indices].round() + task.object.pos_origin_coords
    for batch, center in enumerate(centers):
        top, left = (center - 1.5).round().int().tolist()
        for row in range(4):
            for column in range(4):
                weight = counterpart[batch, 0, row, column].abs().square()
                numerator[0, top + row, left + column] += (
                    weight * (local - old)[batch, 0, row, column]
                )
                denominator[0, top + row, left + column] += weight
    expected = torch.zeros_like(numerator)
    covered = denominator > 0
    expected[covered] = numerator[covered] / denominator[covered]
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(task.object.data, old_object)
    assert torch.count_nonzero(result[~covered]) == 0


@pytest.mark.parametrize(
    "method", [api.PatchInterpolationMethods.FOURIER, api.PatchInterpolationMethods.BILINEAR]
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("confidence_case", ["normal", "zero", "underflow"])
def test_probe_consensus_singleton_is_bit_exact_native_adjoint(method, dtype, confidence_case):
    task = PtychographyTask(_options(dtype=dtype, method=method), **_tiny_data(dtype))
    rec = task.reconstructor
    indices = torch.tensor([2])
    old = _complex_random((1, 1, 4, 4), seed=140, dtype=dtype)
    local = old + _complex_random(old.shape, seed=141, dtype=dtype)
    counterpart = _complex_random(old.shape, seed=142, dtype=dtype)
    if confidence_case == "zero":
        counterpart.zero_()
    elif confidence_case == "underflow":
        counterpart *= torch.finfo(counterpart.real.dtype).tiny
    expected = rec.adjoint_shift_probe_update_direction(
        indices, local - old, first_mode_only=True
    ).mean(0)
    actual = rec._synthesize_probe_update(indices, old, local, counterpart)
    assert torch.equal(actual, expected)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize(
    "method", [api.PatchInterpolationMethods.FOURIER, api.PatchInterpolationMethods.BILINEAR]
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("confidence_case", ["zero", "underflow"])
def test_probe_consensus_zero_confidence_falls_back_to_native_adjoint_mean(
    method, dtype, confidence_case
):
    task = PtychographyTask(_options(dtype=dtype, method=method), **_tiny_data(dtype))
    rec = task.reconstructor
    indices = torch.tensor([2, 0, 1])
    old = _complex_random((3, 1, 4, 4), seed=140, dtype=dtype)
    local = old + _complex_random(old.shape, seed=141, dtype=dtype)
    counterpart = torch.zeros_like(old)
    if confidence_case == "underflow":
        counterpart.fill_(torch.finfo(counterpart.real.dtype).tiny)
    assert torch.count_nonzero(counterpart.abs().square()) == 0
    expected = rec.adjoint_shift_probe_update_direction(
        indices, local - old, first_mode_only=True
    ).mean(0)
    actual = rec._synthesize_probe_update(indices, old, local, counterpart)
    assert torch.equal(actual, expected)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize(
    "method",
    [api.PatchInterpolationMethods.FOURIER, api.PatchInterpolationMethods.BILINEAR],
)
def test_probe_consensus_is_dense_quadratic_minimizer(method):
    dtype = torch.complex128
    task = PtychographyTask(_options(dtype=dtype, method=method), **_tiny_data(dtype))
    rec = task.reconstructor
    indices = torch.tensor([2, 0, 1])
    old = _complex_random((3, 1, 4, 4), seed=150, dtype=dtype)
    local = old + _complex_random(old.shape, seed=151, dtype=dtype)
    counterpart = _complex_random(old.shape, seed=152, dtype=dtype)
    previous_probe = task.probe.data.clone()
    actual = rec._synthesize_probe_update(indices, old, local, counterpart).flatten()
    basis = torch.eye(16, dtype=dtype).reshape(16, 1, 4, 4)
    normal = torch.zeros((16, 16), dtype=dtype)
    rhs = torch.zeros(16, dtype=dtype)
    shifts = rec.forward_model.get_probe_grid_position_data()[2]
    for batch, index in enumerate(indices):
        shifted = rec.forward_model.shift_unique_probes(index.repeat(16), basis)
        operator = shifted.reshape(16, 16).T
        correction = operator.H @ (local[batch] - old[batch]).flatten()
        # Assemble the reverse coordinate transport from basis images. Applying
        # this matrix to complex patches retains interference before taking power.
        transported_basis = ip.shift_images(
            basis[:, 0],
            -shifts[index].expand(16, -1),
            method=method,
            adjoint=False,
            pad=rec.forward_model.pad_for_shift,
        )
        transport = transported_basis.reshape(16, 16).T
        confidence = (transport @ counterpart[batch].flatten()).abs().square()
        normal += torch.diag(confidence.to(dtype))
        rhs += confidence * correction
    expected = torch.linalg.solve(normal, rhs)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    assert torch.isfinite(actual).all()
    assert torch.equal(task.probe.data, previous_probe)


@pytest.mark.parametrize(
    "method",
    [api.PatchInterpolationMethods.FOURIER, api.PatchInterpolationMethods.BILINEAR],
)
def test_probe_confidence_preserves_constant_patches_at_padded_boundaries(method):
    dtype = torch.complex128
    task = PtychographyTask(_options(dtype=dtype, method=method), **_tiny_data(dtype))
    rec = task.reconstructor
    indices = torch.tensor([2, 0, 1])
    old = _complex_random((3, 1, 4, 4), seed=160, dtype=dtype)
    local = old + _complex_random(old.shape, seed=161, dtype=dtype)
    constants = torch.tensor([1, 2j, -3], dtype=dtype)[:, None, None, None]
    counterpart = constants.expand_as(old)
    corrections = rec.adjoint_shift_probe_update_direction(
        indices, local - old, first_mode_only=True
    )
    expected = (corrections * constants.abs().square()).sum(0) / 14

    actual = rec._synthesize_probe_update(indices, old, local, counterpart)

    # Reverse coordinate shifts preserve constants all the way to the boundary;
    # the adjoint of the padded shift does not preserve these confidence maps.
    assert rec.forward_model.pad_for_shift == 1
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    "method",
    [api.PatchInterpolationMethods.FOURIER, api.PatchInterpolationMethods.BILINEAR],
)
def test_probe_confidence_has_no_floor_for_high_contrast_patches(method):
    dtype = torch.complex128
    data = _tiny_data(dtype)
    data["probe_position_y_px"] = data["probe_position_y_px"].round()
    data["probe_position_x_px"] = data["probe_position_x_px"].round()
    task = PtychographyTask(_options(dtype=dtype, method=method), **data)
    rec = task.reconstructor
    indices = torch.tensor([0, 1])
    old = torch.zeros((2, 1, 4, 4), dtype=dtype)
    local = torch.empty_like(old)
    local[0].fill_(1 + 2j)
    local[1].fill_(-2 + 1j)
    counterpart = torch.ones_like(old)
    counterpart[0, 0, 0, 0] = 0
    counterpart[0, 0, 3, 3] = 100
    expected = local.mean(0)
    expected[0, 0, 0] = local[1, 0, 0, 0]
    expected[0, 3, 3] = (10000 * local[0, 0, 3, 3] + local[1, 0, 3, 3]) / 10001

    actual = rec._synthesize_probe_update(indices, old, local, counterpart)

    # A bright pixel must not give a zero-amplitude pixel artificial confidence.
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("levels", [1, 2])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@torch.no_grad()
def test_native_task_compute_then_apply_is_simultaneous_and_unscaled(levels, dtype):
    task = PtychographyTask(_options(levels, dtype, refine_positions=True), **_tiny_data(dtype))
    reconstructor = task.reconstructor
    assert isinstance(reconstructor, MAGPIEReconstructor)
    assert reconstructor.num_levels == levels
    indices = torch.tensor([2, 0])
    before_o, before_p, before_pos = (
        task.object.data.clone(),
        task.probe.data.clone(),
        task.probe_positions.data.clone(),
    )
    with patch.object(
        task.probe_positions.position_correction,
        "get_update",
        wraps=task.probe_positions.position_correction.get_update,
    ) as get_update:
        (delta_o, delta_p, delta_pos), prediction = reconstructor.compute_updates(
            indices, task.dataset.patterns[indices]
        )
    get_update.assert_called_once()
    torch.testing.assert_close(task.object.data, before_o)
    torch.testing.assert_close(task.probe.data, before_p)
    torch.testing.assert_close(task.probe_positions.data, before_pos)
    for delta in (delta_o, delta_p):
        assert delta.dtype == dtype
        assert torch.isfinite(delta).all()
        assert torch.count_nonzero(delta) > 0
    assert prediction.shape == (2, 4, 4)
    assert delta_pos.shape == before_pos.shape
    assert torch.count_nonzero(delta_pos[1]) == 0
    with (
        patch.object(task.object.optimizer, "step") as object_step,
        patch.object(task.probe.optimizer, "step") as probe_step,
        patch.object(
            task.probe_positions, "step_optimizer", wraps=task.probe_positions.step_optimizer
        ) as position_step,
    ):
        reconstructor.apply_updates(delta_o, delta_p, delta_pos)
    object_step.assert_not_called()
    probe_step.assert_not_called()
    position_step.assert_called_once()
    torch.testing.assert_close(task.object.data, before_o + delta_o)
    torch.testing.assert_close(task.probe.data[0], before_p[0] + delta_p)
    torch.testing.assert_close(
        task.probe_positions.data, before_pos + task.probe_positions.step_size * delta_pos
    )


def test_magpie_options_roundtrip_and_public_selection():
    options = _options(levels=2)
    encoded = json.loads(json.dumps(options.get_dict()))
    restored = api.MAGPIEOptions()
    restored.load_from_dict(encoded, strict=True)
    # JSON normalizes inherited tuple-valued options to lists.
    assert json.loads(json.dumps(restored.get_dict())) == encoded
    assert restored.reconstructor_options.get_reconstructor_type() == api.Reconstructors.MAGPIE
    assert restored.reconstructor_options.forward_model_options.pad_for_shift == 1
    restored.check()


@pytest.mark.parametrize("value", [0, -1, 1.5, True, None])
def test_multigrid_levels_require_a_positive_integer(value):
    with pytest.raises(ValidationError, match="multigrid_levels"):
        MAGPIEReconstructorOptions(multigrid_levels=value)
    options = api.MAGPIEOptions()
    with pytest.raises(ValidationError, match="multigrid_levels"):
        options.reconstructor_options.multigrid_levels = value


def _set_option(options, dotted_path, value):
    parts = dotted_path.split(".")
    for name in parts[:-1]:
        options = getattr(options, name)
    setattr(options, parts[-1], value)


@pytest.mark.parametrize(
    "path,value",
    [
        ("reconstructor_options.batching_mode", api.BatchingModes.COMPACT),
        ("reconstructor_options.batching_mode", api.BatchingModes.UNIFORM),
        ("object_options.step_size", 0.5),
        ("probe_options.step_size", 0.5),
        ("object_options.optimizable", False),
        ("probe_options.optimizable", False),
        ("object_options.optimization_plan.start", 1),
        ("probe_options.optimization_plan.stop", 2),
        ("object_options.optimization_plan.stride", 2),
        ("probe_options.optimization_plan.step_size_scheduler_class", "StepLR"),
        ("object_options.optimizer", api.Optimizers.ADAM),
        ("object_options.optimizer_params", {"momentum": 0.9}),
        ("probe_options.optimizer_params", {"lr": 0.25}),
        ("probe_options.optimizer_params", {"weight_decay": 0.1}),
        ("object_options.patch_interpolation_method", api.PatchInterpolationMethods.NEAREST),
        ("reconstructor_options.default_dtype", api.Dtypes.FLOAT16),
        ("reconstructor_options.forward_model_options.pad_for_shift", 0),
        ("probe_options.experimental.sdl_probe_options.enabled", True),
        ("reconstructor_options.forward_model_options.diffraction_pattern_blur_sigma", 1.0),
    ],
)
def test_unsupported_settings_fail_options_validation(path, value):
    options = _options()
    _set_option(options, path, value)
    with pytest.raises(MAGPIECapabilityError):
        options.check()


@pytest.mark.parametrize("levels", [1, 2])
def test_zero_detector_blur_matches_disabled_blur(levels):
    tasks = []
    for sigma in (None, 0.0):
        options = _options(levels=levels)
        options.reconstructor_options.forward_model_options.diffraction_pattern_blur_sigma = sigma
        options.check()
        assert (
            options.reconstructor_options.forward_model_options.diffraction_pattern_blur_sigma
            is None
        )
        task = PtychographyTask(options, **_tiny_data())
        assert task.reconstructor.forward_model.diffraction_pattern_blur_sigma is None
        indices = torch.tensor([0, 1])
        task.reconstructor.run_minibatch((indices,), task.dataset.patterns[indices])
        assert np.isfinite(task.reconstructor.loss_tracker.epoch_loss)
        tasks.append(task)

    torch.testing.assert_close(tasks[0].object.data, tasks[1].object.data, rtol=0, atol=0)
    torch.testing.assert_close(tasks[0].probe.data, tasks[1].probe.data, rtol=0, atol=0)
    assert (
        tasks[0].reconstructor.loss_tracker.epoch_loss
        == tasks[1].reconstructor.loss_tracker.epoch_loss
    )


@pytest.mark.parametrize(
    "path,value",
    [
        ("probe_options.pixel_size_m", 2.0),
        ("probe_options.pixel_size_aspect_ratio", 2.0),
    ],
)
def test_sampling_capabilities_are_checked_on_constructed_forward_model(path, value):
    options = _options()
    _set_option(options, path, value)
    options.check()
    with pytest.raises(MAGPIECapabilityError, match="resampling"):
        PtychographyTask(options, **_tiny_data())


@pytest.mark.parametrize("levels", [1, 2])
@pytest.mark.parametrize("operator", [api.ShiftOperators.AUTO, api.ShiftOperators.FULL_OBJECT])
def test_full_object_sampling_is_rejected_after_geometry_resolution(levels, operator):
    options, data = _options(levels=levels), _tiny_data()
    options.data_options.free_space_propagation_distance_m = 0.01
    options.data_options.wavelength_m = 1e-9
    options.object_options.pixel_size_m = 1e-6
    options.reconstructor_options.forward_model_options.shift_operator = operator
    data["object_data"] = data["object_data"][:, 6:10, 6:10]
    data["probe_position_x_px"] /= 4
    data["probe_position_y_px"] /= 4
    # AUTO depends on the actual object/probe geometry, not options alone.
    options.check()
    with pytest.raises(MAGPIECapabilityError, match=r"FULL_OBJECT.*ShiftOperators\.PATCH"):
        PtychographyTask(options, **data)


@pytest.mark.parametrize("levels", [1, 2])
@pytest.mark.parametrize(
    "operator,object_size,distance",
    [
        (api.ShiftOperators.PATCH, 4, 0.01),
        (api.ShiftOperators.AUTO, 17, 0.01),
        (api.ShiftOperators.AUTO, 4, float("inf")),
    ],
)
def test_supported_patch_sampling_runs(levels, operator, object_size, distance):
    options, data = _options(levels=levels), _tiny_data()
    options.data_options.free_space_propagation_distance_m = distance
    options.data_options.wavelength_m = 1e-9
    options.object_options.pixel_size_m = 1e-6
    options.reconstructor_options.forward_model_options.shift_operator = operator
    if object_size == 4:
        data["object_data"] = data["object_data"][:, 6:10, 6:10]
        data["probe_position_x_px"] /= 4
        data["probe_position_y_px"] /= 4
    task = PtychographyTask(options, **data)
    assert task.reconstructor.forward_model.resolved_shift_operator == api.ShiftOperators.PATCH
    task.run()
    assert torch.isfinite(task.object.data).all()
    assert torch.isfinite(task.probe.data).all()
    assert np.isfinite(task.reconstructor.loss_tracker.table["loss"].to_numpy()).all()


def test_detector_and_probe_shape_mismatch_is_rejected_during_setup():
    options, data = _options(), _tiny_data()
    options.check()
    data["diffraction_data"] = torch.ones((3, 2, 2))
    with pytest.raises(MAGPIECapabilityError, match="matching detector and probe shapes"):
        PtychographyTask(options, **data)


@pytest.mark.parametrize(
    "kind,error",
    [
        ("multislice", MAGPIEMultisliceError),
        ("multimode", MAGPIEMultimodeError),
        ("opr", MAGPIEOPRError),
        ("variable_probe", MAGPIEOPRError),
    ],
)
def test_unsupported_tensor_capabilities_have_explicit_exceptions(kind, error):
    options, data = _options(), _tiny_data()
    if kind == "multislice":
        data["object_data"] = data["object_data"].repeat(2, 1, 1)
        options.object_options.slice_spacings_m = [1e-6]
        options.data_options.wavelength_m = 1e-9
    elif kind == "multimode":
        data["probe_data"] = data["probe_data"].repeat(1, 2, 1, 1)
    elif kind == "opr":
        data["probe_data"] = data["probe_data"].repeat(2, 1, 1, 1)
        data["opr_mode_weights_data"] = torch.ones((3, 2))
    else:
        data["opr_mode_weights_data"] = torch.tensor([[1.0], [0.8], [1.1]])
    with pytest.raises(error):
        PtychographyTask(options, **data)


@pytest.mark.parametrize(
    "path,value,error",
    [
        ("object_options.slice_spacing_options.optimizable", True, MAGPIEMultisliceError),
        ("object_options.multislice_regularization.enabled", True, MAGPIEMultisliceError),
        ("opr_mode_weight_options.optimize_intensity_variation", True, MAGPIEOPRError),
        ("opr_mode_weight_options.smoothing.enabled", True, MAGPIEOPRError),
    ],
)
def test_unsupported_capability_options_have_explicit_exceptions(path, value, error):
    options = _options()
    _set_option(options, path, value)
    with pytest.raises(error):
        options.check()


@pytest.mark.parametrize("factor", ["object", "probe"])
@pytest.mark.parametrize("as_numpy", [False, True])
@pytest.mark.parametrize(
    "invalid", [0, float("nan"), float("inf"), -float("inf"), complex(1, float("nan"))]
)
def test_invalid_initial_factors_fail_constructed_data_validation(factor, as_numpy, invalid):
    data = _tiny_data()
    key = f"{factor}_data"
    if invalid == 0:
        data[key].zero_()
        reason = "all-zero"
    else:
        data[key].reshape(-1)[0] = invalid
        reason = "NaN or infinity"
    if as_numpy:
        data[key] = data[key].numpy()
    with pytest.raises(MAGPIEInitializationError, match=rf"{reason}.*{key}.*Supply"):
        PtychographyTask(_options(), **data)


@pytest.mark.parametrize("factor", ["object", "probe"])
def test_invalid_deprecated_initial_guess_is_checked(factor):
    options, data = _options(), _tiny_data()
    values = data.pop(f"{factor}_data")
    getattr(options, f"{factor}_options").initial_guess = torch.zeros_like(values)
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(MAGPIEInitializationError, match=rf"all-zero {factor}_data"),
    ):
        PtychographyTask(options, **data)


@pytest.mark.parametrize("factor", ["object", "probe"])
def test_explicit_valid_initial_guess_overrides_invalid_deprecated_guess(factor):
    options, data = _options(), _tiny_data()
    getattr(options, f"{factor}_options").initial_guess = torch.zeros_like(data[f"{factor}_data"])
    with pytest.warns(DeprecationWarning, match="ignored"):
        task = PtychographyTask(options, **data)
    torch.testing.assert_close(getattr(task, factor).data, data[f"{factor}_data"])


@pytest.mark.parametrize("factor", ["object", "probe"])
@pytest.mark.parametrize("invalid", [0, float("nan")])
@torch.no_grad()
def test_direct_reconstructor_rejects_invalid_initial_factors(factor, invalid):
    task = PtychographyTask(_options(), **_tiny_data())
    parameter = getattr(task, factor)
    parameter.set_data(torch.full_like(parameter.data, invalid))
    with pytest.raises(MAGPIEInitializationError, match=f"{factor}_data"):
        MAGPIEReconstructor(
            task.reconstructor.parameter_group, task.dataset, task.options.reconstructor_options
        ).check_inputs()


@pytest.mark.parametrize("factor", ["object", "probe"])
@pytest.mark.parametrize("value", [1e100, 1e-100])
def test_initial_factors_are_rechecked_after_dtype_conversion(factor, value):
    data = _tiny_data(torch.complex128)
    data[f"{factor}_data"].fill_(value)
    with pytest.raises(MAGPIEInitializationError, match=f"{factor}_data"):
        PtychographyTask(_options(dtype=torch.complex64), **data)


@pytest.mark.parametrize("factor", ["object", "probe"])
@pytest.mark.parametrize("value", [1e100, 1e-100])
@pytest.mark.parametrize("container", [list, tuple])
def test_python_initial_guesses_preserve_precision_before_task_setup(factor, value, container):
    assert torch.get_default_dtype() == torch.float32
    data = _tiny_data(torch.complex128)
    key = f"{factor}_data"
    data[key].fill_(value)
    expected = data[key].clone()
    data[key] = container(data[key].tolist())
    task = PtychographyTask(_options(dtype=torch.complex128), **data)
    torch.testing.assert_close(getattr(task, factor).data, expected, rtol=0, atol=0)


@pytest.mark.parametrize("levels", [1, 2])
@pytest.mark.parametrize("container", ["tensor", "numpy", "list", "tuple"])
def test_single_zero_object_pixel_is_rejected_during_setup(levels, container):
    data = _tiny_data()
    data["object_data"][0, 8, 8] = 0
    if container == "numpy":
        data["object_data"] = data["object_data"].numpy()
    elif container in ("list", "tuple"):
        data["object_data"] = data["object_data"].tolist()
        if container == "tuple":
            data["object_data"] = tuple(data["object_data"])
    with pytest.raises(
        MAGPIEInitializationError,
        match="zero entries in object_data.*Supply.*nonzero amplitude at every pixel",
    ):
        PtychographyTask(_options(levels=levels), **data)


@pytest.mark.parametrize("levels", [1, 2])
@torch.no_grad()
def test_direct_reconstructor_rejects_single_zero_object_pixel(levels):
    task = PtychographyTask(_options(levels=levels), **_tiny_data())
    values = task.object.data
    values[0, 8, 8] = 0
    task.object.set_data(values)
    with pytest.raises(MAGPIEInitializationError, match="zero entries in object_data"):
        MAGPIEReconstructor(
            task.reconstructor.parameter_group, task.dataset, task.options.reconstructor_options
        ).check_inputs()


def test_single_object_pixel_underflow_is_rejected_after_dtype_conversion():
    data = _tiny_data(torch.complex128)
    data["object_data"][0, 8, 8] = 1e-100
    with pytest.raises(MAGPIEInitializationError, match="zero entries in object_data"):
        PtychographyTask(_options(dtype=torch.complex64), **data)


@pytest.mark.parametrize("levels", [1, 2])
def test_partly_zero_initial_probe_remains_valid(levels):
    options, data = _options(levels=levels), _tiny_data()
    options.object_options.remove_object_probe_ambiguity.enabled = True
    options.reconstructor_options.batch_size = 1
    data["probe_data"][0, 0, 2, 2] = 0
    task = PtychographyTask(options, **data)
    torch.testing.assert_close(task.object.data, data["object_data"])
    torch.testing.assert_close(task.probe.data, data["probe_data"])
    task.run()
    assert torch.isfinite(task.object.data).all()
    assert torch.isfinite(task.probe.data).all()
    assert np.isfinite(task.reconstructor.loss_tracker.table["loss"]).all()


def test_excessive_multigrid_depth_is_rejected_not_truncated():
    options = _options(levels=4)
    options.check()
    with pytest.raises(MAGPIECapabilityError, match="(?i)(multigrid|level|divisib)"):
        PtychographyTask(options, **_tiny_data())


def test_position_correction_uses_supported_native_gradient_method():
    options = _options(refine_positions=True)
    options.probe_position_options.correction_options.correction_type = (
        api.PositionCorrectionTypes.CROSS_CORRELATION
    )
    with pytest.raises(MAGPIECapabilityError, match="(?i)gradient"):
        options.check()


def test_explicit_none_weights_override_deprecated_initial_weights():
    options = _options()
    options.opr_mode_weight_options.initial_weights = [[1.0], [0.5], [2.0]]
    with pytest.warns(DeprecationWarning, match="ignored"):
        task = PtychographyTask(options, opr_mode_weights_data=None, **_tiny_data())
    torch.testing.assert_close(
        task.opr_mode_weights.data, torch.ones_like(task.opr_mode_weights.data)
    )


@torch.no_grad()
def test_native_low_memory_forward_model_produces_same_updates():
    options = _options(levels=2, dtype=torch.complex128)
    task = PtychographyTask(options, **_tiny_data(torch.complex128))
    options_low_memory = _options(levels=2, dtype=torch.complex128)
    options_low_memory.reconstructor_options.forward_model_options.low_memory_mode = True
    low_memory_task = PtychographyTask(options_low_memory, **_tiny_data(torch.complex128))
    indices = torch.tensor([2, 0])
    expected, expected_prediction = task.reconstructor.compute_updates(
        indices, task.dataset.patterns[indices]
    )
    actual, actual_prediction = low_memory_task.reconstructor.compute_updates(
        indices, low_memory_task.dataset.patterns[indices]
    )
    for actual_factor, expected_factor in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(actual_factor, expected_factor)
    assert actual[2] is expected[2] is None
    torch.testing.assert_close(actual_prediction, expected_prediction)


@pytest.mark.parametrize("levels", [1, 2])
@torch.no_grad()
def test_inherited_minibatch_loop_handles_short_final_batch(levels):
    task = PtychographyTask(_options(levels=levels), **_tiny_data())
    reconstructor = task.reconstructor
    batches = list(reconstructor.dataloader)
    assert [len(batch[0]) for batch in batches] == [2, 1]
    torch.testing.assert_close(
        torch.cat([batch[0] for batch in batches]).sort().values, torch.arange(3)
    )
    for batch in batches:
        input_data, measured = reconstructor.prepare_batch_data(batch)
        reconstructor.run_minibatch(input_data, measured)
    assert reconstructor.loss_tracker.accumulated_num_batches == 2
    assert np.isfinite(reconstructor.loss_tracker.epoch_loss)
    assert torch.isfinite(task.object.data).all()
    assert torch.isfinite(task.probe.data).all()


def test_task_validates_each_options_component_once():
    class NoArgumentCheckOptions(api.MAGPIEOptions):
        def check(self):
            super().check()

    options = NoArgumentCheckOptions()
    options.load_from_dict(_options().get_dict())
    components = (
        options.data_options,
        options.reconstructor_options,
        options.object_options,
        options.probe_options,
        options.probe_position_options,
        options.opr_mode_weight_options,
    )
    with ExitStack() as stack:
        validators = [
            stack.enter_context(
                patch.object(
                    type(component), "check", autospec=True, side_effect=type(component).check
                )
            )
            for component in components
        ]
        task = PtychographyTask(options, **_tiny_data())
        # Constructed-data validation must not dispatch options validation again.
        task.reconstructor.check_inputs()
        for validator, component in zip(validators, components):
            validator.assert_called_once_with(component, options)


def test_direct_construction_uses_explicitly_validated_complete_options():
    options, data = _options(), _tiny_data()
    options.check()
    group = PlanarPtychographyParameterGroup(
        object=PlanarObject(data=data["object_data"], options=options.object_options),
        probe=Probe(data=data["probe_data"], options=options.probe_options),
        probe_positions=ProbePositions(
            data=torch.stack((data["probe_position_y_px"], data["probe_position_x_px"]), dim=1),
            options=options.probe_position_options,
        ),
        opr_mode_weights=OPRModeWeights(
            data=torch.ones((3, 1)), options=options.opr_mode_weight_options
        ),
    )
    dataset = PtychographyDataset(
        data["diffraction_data"],
        wavelength_m=options.data_options.wavelength_m,
        free_space_propagation_distance_m=options.data_options.free_space_propagation_distance_m,
        fft_shift=options.data_options.fft_shift,
    )
    reconstructor = MAGPIEReconstructor(group, dataset, options.reconstructor_options)
    reconstructor.build()
    indices = torch.tensor([0, 1])
    reconstructor.run_minibatch((indices,), dataset.patterns[indices])
    assert torch.isfinite(group.object.data).all()
    assert torch.isfinite(group.probe.data).all()
    assert np.isfinite(reconstructor.loss_tracker.epoch_loss)


def test_update_stages_appear_in_standard_timing_reports(monkeypatch):
    task = PtychographyTask(_options(), **_tiny_data())
    names = (
        "ENABLE_TIMING",
        "ELAPSED_TIME_DICT",
        "ADVANCED_TIME_DICT",
        "CURRENT_DICT_REFERENCE",
        "TIMING_OVERHEAD_ARRAY",
    )
    saved = {name: getattr(timer_utils, name) for name in names}
    # The native timer synchronizes the configured accelerator even on CPU.
    monkeypatch.setattr(
        timer_utils.AcceleratorModuleWrapper.get_module(), "synchronize", lambda: None
    )
    try:
        timer_utils.clear_timer_globals()
        timer_utils.toggle_timer(True)
        indices = torch.tensor([0, 1])
        task.reconstructor.run_minibatch((indices,), task.dataset.patterns[indices])
        detailed = timer_utils.ADVANCED_TIME_DICT["PIEReconstructor.run_minibatch"]
        for name in ("MAGPIEReconstructor.compute_updates", "MAGPIEReconstructor.apply_updates"):
            elapsed = timer_utils.ELAPSED_TIME_DICT[name]
            assert elapsed.shape == (1,)
            assert np.isfinite(elapsed).all() and (elapsed >= 0).all()
            np.testing.assert_array_equal(detailed[name]["time"], elapsed)
    finally:
        for name, value in saved.items():
            setattr(timer_utils, name, value)
