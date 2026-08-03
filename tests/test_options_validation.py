import json

import numpy as np
import warnings

import pytest
import torch
from pydantic import ValidationError

import ptychi.api as api
from ptychi.api.options import base
from ptychi.data_structures.object import PlanarObject
from ptychi.data_structures.probe import SynthesisDictLearnProbe


def test_unknown_option_fields_are_forbidden():
    with pytest.raises(ValidationError):
        api.LSQMLOptions(not_a_real_field=True)

    options = api.LSQMLOptions()
    with pytest.raises(ValidationError):
        options.not_a_real_field = True


def test_assignment_validation_rejects_bad_types_and_enum_values():
    options = api.LSQMLOptions()

    with pytest.raises(ValidationError):
        options.reconstructor_options.batch_size = "not an int"

    with pytest.raises(ValidationError):
        options.reconstructor_options.default_device = "not-a-device"


def test_documented_ranges_are_validated():
    options = api.LSQMLOptions()

    with pytest.raises(ValidationError):
        options.object_options.smoothness_constraint.alpha = 0.2

    with pytest.raises(ValidationError):
        options.probe_position_options.correction_options.update_magnitude_limit = 0

    with pytest.raises(ValidationError):
        options.reconstructor_options.num_epochs = 0

    with pytest.raises(ValidationError):
        options.probe_options.optimization_plan.stride = 0

    dm_options = api.DMOptions()
    with pytest.raises(ValidationError):
        dm_options.object_options.inertia = 1.5

    raar_options = api.RAAROptions()
    with pytest.raises(ValidationError):
        raar_options.reconstructor_options.beta = 0.5


def test_probe_center_deprecated_alias_is_promoted_by_validation():
    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        options = base.ProbeCenterConstraintOptions(use_intensity_for_com=True)

    assert options.use_total_intensity_for_com is True
    assert any(issubclass(w.category, DeprecationWarning) for w in warnings_record)


def test_probe_center_incompatible_local_settings_are_rejected():
    options = base.ProbeCenterConstraintOptions()
    options.center_modes_individually = True

    with pytest.raises(ValidationError, match="use_total_intensity_for_com"):
        options.use_total_intensity_for_com = True

    assert options.use_total_intensity_for_com is False


def test_opr_mode_weight_optimization_requires_at_least_one_enabled_component():
    with pytest.raises(ValidationError, match="OPRModeWeights"):
        base.OPRModeWeightsOptions(
            optimizable=True,
            optimize_eigenmode_weights=False,
            optimize_intensity_variation=False,
        )

    options = base.OPRModeWeightsOptions()
    options.optimizable = True

    with pytest.raises(ValidationError, match="OPRModeWeights"):
        options.optimize_eigenmode_weights = False

    assert options.optimize_eigenmode_weights is True


def test_lsqml_position_momentum_memory_is_positive():
    options = api.LSQMLOptions()

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        options.probe_position_options.momentum_acceleration_memory = 0


def test_task_data_option_fields_preserve_array_objects():
    options = api.LSQMLOptions()
    diffraction_data = torch.ones((2, 2, 2))
    valid_pixel_mask = np.ones((2, 2), dtype=bool)
    object_initial_guess = torch.ones((1, 4, 4), dtype=torch.complex64)
    probe_initial_guess = torch.ones((1, 1, 2, 2), dtype=torch.complex64)
    probe_position_x_px = torch.tensor([0.0, 1.0])
    probe_position_y_px = np.array([0.0, 1.0])
    opr_mode_weights = torch.ones((2, 1))

    options.data_options.data = diffraction_data
    options.data_options.valid_pixel_mask = valid_pixel_mask
    options.object_options.initial_guess = object_initial_guess
    options.probe_options.initial_guess = probe_initial_guess
    options.probe_position_options.position_x_px = probe_position_x_px
    options.probe_position_options.position_y_px = probe_position_y_px
    options.opr_mode_weight_options.initial_weights = opr_mode_weights

    assert options.data_options.data is diffraction_data
    assert options.data_options.valid_pixel_mask is valid_pixel_mask
    assert options.object_options.initial_guess is object_initial_guess
    assert options.probe_options.initial_guess is probe_initial_guess
    assert options.probe_position_options.position_x_px is probe_position_x_px
    assert options.probe_position_options.position_y_px is probe_position_y_px
    assert options.opr_mode_weight_options.initial_weights is opr_mode_weights

    json.dumps(options.get_dict())


def test_settings_array_like_option_fields_store_json_native_containers():
    options = api.LSQMLOptions()
    options.object_options.slice_spacings_m = np.array([1e-6])
    options.object_options.position_origin_coords = torch.tensor([2.0, 2.0])
    options.object_options.hard_limits_magnitude_phase.abs_lim = torch.tensor([0.5, 1.0])
    options.probe_options.support_constraint.fixed_probe_support_params = torch.tensor(
        [1.0, 1.0, 1.0, 1.0]
    )

    assert isinstance(options.object_options.slice_spacings_m, list)
    assert isinstance(options.object_options.position_origin_coords, list)
    assert isinstance(options.object_options.hard_limits_magnitude_phase.abs_lim, list)
    assert isinstance(options.probe_options.support_constraint.fixed_probe_support_params, list)

    json.dumps(options.get_dict())


def test_position_origin_coords_list_is_converted_for_backend_use():
    options = base.ObjectOptions(
        optimizable=False,
        determine_position_origin_coords_by=api.ObjectPosOriginCoordsMethods.SPECIFIED,
        position_origin_coords=(2.0, 3.0),
    )
    obj = PlanarObject(data=torch.ones((1, 4, 4), dtype=torch.complex64), options=options)

    obj.update_pos_origin_coordinates()

    assert isinstance(options.position_origin_coords, (list, tuple))
    assert torch.allclose(obj.pos_origin_coords.cpu(), torch.tensor([2.0, 3.0], device="cpu"))


def test_slice_spacings_list_is_converted_for_backend_use():
    options = base.ObjectOptions(optimizable=False, slice_spacings_m=(1e-6,))
    obj = PlanarObject(data=torch.ones((2, 4, 4), dtype=torch.complex64), options=options)

    assert isinstance(options.slice_spacings_m, (list, tuple))
    assert torch.allclose(obj.slice_spacings.data.cpu(), torch.tensor([1e-6], device="cpu"))


def test_synthesis_dictionary_lists_are_converted_for_backend_use():
    options = api.RPIEOptions().probe_options
    options.optimizable = False
    d_mat = np.eye(4, dtype=np.complex64)
    options.experimental.sdl_probe_options.d_mat = d_mat
    options.experimental.sdl_probe_options.d_mat_conj_transpose = d_mat.conj().T
    options.experimental.sdl_probe_options.d_mat_pinv = np.linalg.pinv(d_mat)
    options.experimental.sdl_probe_options.probe_sparse_code_nnz = 2

    probe = SynthesisDictLearnProbe(
        data=torch.ones((1, 1, 2, 2), dtype=torch.complex64),
        options=options,
    )

    assert isinstance(options.experimental.sdl_probe_options.d_mat, list)
    assert torch.is_tensor(probe.dictionary_matrix)
    assert probe.dictionary_matrix.dtype == torch.complex64
