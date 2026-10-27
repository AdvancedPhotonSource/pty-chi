import pytest
import torch

import ptychi.api as api
from ptychi.api.options.base import ObjectOptions
from ptychi.data_structures.object import PlanarObject


def _make_object(shape, method):
    options = ObjectOptions(
        optimizable=False,
        determine_position_origin_coords_by=method,
    )
    data = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(1, *shape)
    return PlanarObject(data=data.to(torch.complex64), options=options)


@pytest.mark.parametrize(
    "shape, origin, start",
    [
        ((8, 10), (3.5, 4.5), (2, 3)),
        ((9, 11), (4.5, 5.5), (3, 4)),
    ],
)
def test_support_origin_extracts_even_patch_at_object_center(shape, origin, start):
    object_ = _make_object(shape, api.ObjectPosOriginCoordsMethods.SUPPORT)

    expected_origin = object_.pos_origin_coords.new_tensor(origin)
    assert torch.equal(object_.pos_origin_coords, expected_origin)

    object_.update_pos_origin_coordinates()
    patch = object_.extract_patches(
        torch.zeros((1, 2)),
        patch_shape=(4, 4),
        integer_mode=True,
    )

    assert torch.equal(object_.pos_origin_coords, expected_origin)
    y, x = start
    assert torch.equal(patch[0, 0], object_.data[0, y:y + 4, x:x + 4])


def test_positions_origin_centers_scan_midpoint_in_object():
    object_ = _make_object((8, 10), api.ObjectPosOriginCoordsMethods.POSITIONS)
    positions = torch.tensor([[-2.0, -3.0], [2.0, 3.0]])

    object_.update_pos_origin_coordinates(positions)

    expected_origin = object_.pos_origin_coords.new_tensor([3.5, 4.5])
    assert torch.equal(object_.pos_origin_coords, expected_origin)
