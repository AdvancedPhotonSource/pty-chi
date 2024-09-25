import torch
from ptychointerim.image_proc import find_cross_corr_peak


def compute_positions_cross_correlation_update(obj_patches: torch.Tensor, 
                                                updated_obj_patches: torch.Tensor, 
                                                indices: torch.Tensor, 
                                                positions: torch.Tensor,
                                                probe: torch.Tensor):
    """
    Use cross-correlation position correction to compute an update to the probe positions.

    Based on the paper:
    - Translation position determination in ptychographic coherent diffraction imaging (2013) - Fucai Zhang

    :param obj_patches: A (batch_size, h, w) patches of the object.
    :param updated_obj_patches: A (batch_size, h, w) patches of the object with the new updates applied.
    :param indices: A (batch_size) tensor specifying the position index that each object patch corresponds to.
    :param positions: A (n_positions, 2) tensor of all measurement positions.
    :param probe: A (h, w) tensor of the probe.
    """

    delta_pos = torch.zeros_like(positions)

    probe_thresh = probe.abs().max() * 0.1
    probe_mask = probe.abs() > probe_thresh

    for i in range(len(positions[indices])):
        delta_pos[indices[i]] = -find_cross_corr_peak(
            updated_obj_patches[i] * probe_mask,
            obj_patches[i] * probe_mask,
            scale=20000, real_space_width=.01
        )

    return delta_pos