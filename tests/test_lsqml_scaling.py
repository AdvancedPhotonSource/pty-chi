import pytest
import torch

import ptychi.api as api
import ptychi.maths as pmath
import ptychi.propagate as propagation
import ptychi.utils as utils
from ptychi.api.task import PtychographyTask


@pytest.fixture(autouse=True)
def cpu_double_precision():
    device, dtype = torch.get_default_device(), torch.get_default_dtype()
    complex_dtype = utils.get_default_complex_dtype()
    fft_precision = pmath.get_use_double_precision_for_fft()
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    utils.set_default_complex_dtype(torch.complex128)
    yield
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    utils.set_default_complex_dtype(complex_dtype)
    pmath.set_use_double_precision_for_fft(fft_precision)


class FieldUnitsPropagator(torch.nn.Module):
    """Represent the same detector field with source amplitudes multiplied by s."""

    def __init__(self, propagator, scale):
        super().__init__()
        self.propagator = propagator
        self.scale = scale

    def propagate_forward(self, field):
        return self.propagator.propagate_forward(field) / self.scale

    def propagate_backward(self, field):
        return self.propagator.propagate_backward(field) * self.scale


def make_task(scale=1.0, jointly=True, propagator_type=propagation.FourierPropagator):
    generator = torch.Generator().manual_seed(71)
    probe = 1 + 0.2 * torch.randn(1, 2, 8, 12, dtype=torch.complex128, generator=generator)
    options = api.LSQMLOptions()
    options.reconstructor_options.default_device = api.Devices.CPU
    options.reconstructor_options.default_dtype = api.Dtypes.FLOAT64
    options.reconstructor_options.use_double_precision_for_fft = True
    options.reconstructor_options.rescale_probe_intensity_in_first_epoch = False
    options.reconstructor_options.single_slice_solve_obj_prb_step_size_jointly = jointly
    options.reconstructor_options.batch_size = 4
    options.data_options.fft_shift = False
    options.object_options.pixel_size_m = 2e-8
    options.object_options.build_preconditioner_with_all_modes = True
    options.probe_options.optimizable = True
    options.probe_position_options.optimizable = False
    task = PtychographyTask(
        options, diffraction_data=torch.ones(4, 8, 12),
        object_data=torch.ones(1, 16, 20, dtype=torch.complex128),
        probe_data=probe * scale,
        probe_position_x_px=torch.tensor([-2., 2., -2., 2.]),
        probe_position_y_px=torch.tensor([-2., -2., 2., 2.]),
    )
    if propagator_type is propagation.FourierPropagator:
        propagator = propagator_type()
    else:
        parameters = propagation.WavefieldPropagatorParameters.create_simple(
            wavelength_m=1e-10, width_px=12, height_px=8,
            pixel_width_m=2e-8, pixel_height_m=2e-8, propagation_distance_m=0.0043,
        )
        propagator = propagator_type(parameters)
    model = task.reconstructor.forward_model
    model.clear_resampling_caches()
    model.free_space_propagator = FieldUnitsPropagator(propagator, scale)
    return task


@pytest.mark.parametrize("scale", [1e-4, 256.0])
@pytest.mark.parametrize("jointly", [False, True])
@pytest.mark.parametrize("propagator_type", [
    propagation.FourierPropagator, propagation.AngularSpectrumPropagator,
    propagation.FresnelTransformPropagator, propagation.FraunhoferPropagator,
])
@torch.no_grad()
def test_lsqml_updates_preserve_field_units(scale, jointly, propagator_type):
    tasks = [make_task(s, jointly, propagator_type) for s in (1.0, scale)]
    indices = torch.arange(4)
    model = tasks[0].reconstructor.forward_model
    # Spatially varying synthetic data exercise changes in the probe profile.
    measured = model(indices) * torch.linspace(0.6, 0.95, 96).reshape(8, 12)
    for task in tasks:
        task.reconstructor.dataset.patterns.copy_(measured)
    for epoch in range(3):
        states = []
        for task in tasks:
            recon = task.reconstructor
            recon.current_epoch = epoch
            recon.run_pre_epoch_hooks()
            predicted = recon.forward_model(indices)
            psi_opt = recon.run_reciprocal_space_step(predicted, measured, indices)
            chi = psi_opt - recon.forward_model.intermediate_variables["psi"]
            recon.run_real_space_step(psi_opt, indices)
            direction = -task.object.get_grad().clone()
            recon.apply_reconstruction_parameter_updates(indices)
            states.append({
                "prediction": predicted, "chi": chi,
                "preconditioner": task.object.preconditioner.clone(), "direction": direction,
                "alpha_o": recon.reconstructor_buffers.alpha_object_all_pos_all_slices.clone(),
                "alpha_p": recon.reconstructor_buffers.alpha_probe_all_pos.clone(),
                "object_update": -task.object.get_grad().clone(),
                "probe_update": -task.probe.get_grad().clone(),
                "object": task.object.data.clone(), "probe": task.probe.data.clone(),
            })
        factors = {"chi": scale, "preconditioner": scale**2,
                   "probe_update": scale, "probe": scale}
        for name, reference in states[0].items():
            torch.testing.assert_close(
                states[1][name] / factors.get(name, 1), reference, rtol=1e-9, atol=1e-11,
                msg=lambda msg: f"{name}, epoch {epoch}: {msg}",
            )


@pytest.mark.parametrize("jointly", [False, True])
@pytest.mark.parametrize("case", ["zero_object", "zero_probe", "zero_both", "collinear"])
@torch.no_grad()
def test_step_sizes_handle_degenerate_directions(jointly, case):
    recon = make_task(jointly=jointly).reconstructor
    patches = torch.ones(4, 1, 8, 12, dtype=torch.complex128)
    probe = torch.ones(1, 2, 8, 12, dtype=torch.complex128)
    delta_o, delta_p = patches.clone(), probe[0].clone()
    if case in ("zero_object", "zero_both"):
        delta_o.zero_()
    if case in ("zero_probe", "zero_both"):
        delta_p.zero_()
    chi = delta_o * probe + delta_p
    if jointly:
        alpha_o, alpha_p = recon.calculate_object_and_probe_update_step_sizes(
            chi, patches, delta_o, delta_p, probe=probe,
        )
    else:
        alpha_o = recon.calculate_object_update_step_sizes(chi, delta_o, probe=probe)
        alpha_p = recon.calculate_probe_update_step_sizes(chi, patches, delta_p)
    assert torch.isfinite(alpha_o).all() and torch.isfinite(alpha_p).all()
    for direction, alpha in ((delta_o, alpha_o), (delta_p, alpha_p)):
        if torch.count_nonzero(direction) == 0:
            assert torch.count_nonzero(alpha) == 0
        else:
            assert (alpha > 0).all()


@pytest.mark.parametrize("scale", [1e-3, 256.0])
@torch.no_grad()
def test_constraints_and_momentum_preserve_field_units(scale):
    tasks = [make_task(s) for s in (1, scale)]
    for task, s in zip(tasks, (1, scale)):
        task.reconstructor.run_pre_epoch_hooks()
        task.object.set_data(task.object.data * 1.2)
        task.probe.probe_power = 100 * s**2
        task.probe.constrain_probe_power(task.object)
    torch.testing.assert_close(tasks[1].object.data, tasks[0].object.data)
    torch.testing.assert_close(tasks[1].probe.data / scale, tasks[0].probe.data)

    for task in tasks:
        task.object.remove_object_probe_ambiguity(task.probe)
    torch.testing.assert_close(tasks[1].object.data, tasks[0].object.data)
    torch.testing.assert_close(tasks[1].probe.data / scale, tasks[0].probe.data)

    object_direction = torch.linspace(0.01, 0.02, 320).reshape(1, 16, 20).to(torch.complex128)
    probe_direction = torch.full((2, 8, 12), 0.02 + 0.01j, dtype=torch.complex128)
    for _ in range(6):
        for task, s in zip(tasks, (1, scale)):
            recon = task.reconstructor
            recon.options.momentum_acceleration_gain = 0.5
            recon._apply_object_momentum(torch.ones(1), object_direction)
            recon._apply_probe_momentum(torch.tensor(1.0), probe_direction * s)
        torch.testing.assert_close(tasks[1].object.data, tasks[0].object.data)
        torch.testing.assert_close(tasks[1].probe.data / scale, tasks[0].probe.data)
        torch.testing.assert_close(
            tasks[1].reconstructor.object_momentum_params.velocity_map,
            tasks[0].reconstructor.object_momentum_params.velocity_map,
        )
        torch.testing.assert_close(
            tasks[1].reconstructor.probe_momentum_params.velocity_map / scale,
            tasks[0].reconstructor.probe_momentum_params.velocity_map,
        )
