from ptychi.api import enums
from ptychi.api.options import base
from ptychi.api.registry import RECONSTRUCTOR_OPTIONS_MAP


def test_reconstructor_options_map_is_complete():
    assert set(RECONSTRUCTOR_OPTIONS_MAP) == set(enums.Reconstructors)

    expected_option_types = {
        "object_options": base.ObjectOptions,
        "probe_options": base.ProbeOptions,
        "probe_position_options": base.ProbePositionOptions,
        "opr_mode_weight_options": base.OPRModeWeightsOptions,
        "reconstructor_options": base.ReconstructorOptions,
    }
    for option_classes in RECONSTRUCTOR_OPTIONS_MAP.values():
        assert set(option_classes) == set(expected_option_types)
        for field_name, base_class in expected_option_types.items():
            assert issubclass(option_classes[field_name], base_class)
