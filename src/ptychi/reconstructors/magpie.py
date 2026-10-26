"""Geometric-mean rPIE and multigrid MAGPIE for blind ptychography.

Based on Zhang et al., "Stochastic Multigrid Method for Blind Ptychography",
arXiv:2511.01793, and the authors' implementation at
https://github.com/borongzhang/blind_magpie_ptychography.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

import ptychi.image_proc as ip
from ptychi.api import enums
from ptychi.api.options.magpie import (
    MAGPIECapabilityError,
    MAGPIEInitializationError,
    MAGPIEMultimodeError,
    MAGPIEMultisliceError,
    MAGPIEOPRError,
    MAGPIEReconstructorOptions,
)
from ptychi.reconstructors.pie import PIEReconstructor
from ptychi.timing.timer_utils import timer


def _check_data_capabilities(
    object_data: Tensor, probe_data: Tensor, opr_mode_weights_data: Tensor
):
    """Check the supported slice/mode counts and constructed OPR weight values."""
    if object_data.shape[0] != 1:
        raise MAGPIEMultisliceError(
            f"GM-rPIE/GM-MAGPIE received {object_data.shape[0]} object slices but support "
            "a single slice; supply object_data with shape (1, height, width)."
        )
    if probe_data.shape[0] != 1:
        raise MAGPIEOPRError(
            f"GM-rPIE/GM-MAGPIE received {probe_data.shape[0]} OPR modes but do not support "
            "OPR; supply probe_data with one shared mode, shape (1, 1, height, width)."
        )
    if probe_data.shape[1] != 1:
        raise MAGPIEMultimodeError(
            f"GM-rPIE/GM-MAGPIE received {probe_data.shape[1]} incoherent probe modes but support "
            "a single coherent mode; supply probe_data with shape (1, 1, height, width)."
        )
    if opr_mode_weights_data.shape[-1] != 1 or not torch.all(opr_mode_weights_data == 1).item():
        raise MAGPIEOPRError(
            f"GM-rPIE/GM-MAGPIE received OPR weights with shape {tuple(opr_mode_weights_data.shape)} "
            "and multiple modes or non-unity values. OPR and scan-dependent intensity "
            "variation are unsupported; omit the weights or supply one mode fixed at unity."
        )


def _check_initial_factors(object_data: Tensor, probe_data: Tensor):
    """Validate the initial factors after native parameter dtype conversion."""
    for name, values in (("object", object_data), ("probe", probe_data)):
        if not torch.isfinite(values).all().item():
            raise MAGPIEInitializationError(
                f"GM-rPIE/GM-MAGPIE received NaN or infinity in {name}_data. "
                f"Supply an initial {name} guess containing only finite values."
            )
        if not torch.any(values != 0).item():
            raise MAGPIEInitializationError(
                f"GM-rPIE/GM-MAGPIE received all-zero {name}_data. "
                f"Supply a finite initial {name} guess with nonzero amplitude; "
                "geometric-mean updates cannot recover an entirely zero factor."
            )
        if name == "object" and torch.any(values == 0).item():
            raise MAGPIEInitializationError(
                "GM-rPIE/GM-MAGPIE received zero entries in object_data. "
                "Supply a finite initial object guess with nonzero amplitude at every pixel "
                "(for example, an array of complex ones); exact-zero object pixels cannot "
                "be updated by the geometric mean."
            )


def _power(x: Tensor) -> Tensor:
    return x.real.square() + x.imag.square()


def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    """Divide where the denominator is nonzero, returning zero elsewhere."""
    nonzero = denominator != 0
    safe_denominator = torch.where(nonzero, denominator, 1)
    return torch.where(nonzero, numerator / safe_denominator, 0)


def _restrict_2x(x: Tensor) -> Tensor:
    """Average adjacent 2 x 2 pixels, preserving real or complex dtype."""
    if torch.is_complex(x):
        return torch.complex(F.avg_pool2d(x.real, 2, 2), F.avg_pool2d(x.imag, 2, 2))
    return F.avg_pool2d(x, 2, 2)


def _prolong_2x(x: Tensor) -> Tensor:
    """Prolong by assigning the same value to all four child pixels."""
    return x.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)


def aligned_geom_mean(a: Tensor, b: Tensor) -> Tensor:
    """Return the elementwise geometric mean on the shorter complex phase arc.

    Inputs broadcast and retain their promoted complex precision. Antipodal
    phases choose the positive relative-imaginary square root deterministically;
    finite zero inputs give zero. Nonfinite inputs propagate.
    """
    dtype = torch.promote_types(a.dtype, b.dtype)
    if not dtype.is_complex:
        dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    a, b = torch.broadcast_tensors(a.to(dtype), b.to(dtype))
    phase_a, phase_b = torch.sgn(a), torch.sgn(b)
    relative_phase = phase_b * phase_a.conj()
    relative_root = torch.sqrt(relative_phase)
    tolerance = 2 * torch.finfo(relative_phase.real.dtype).eps
    flip_tie = (
        (relative_phase.real < 0)
        & (relative_phase.imag.abs() <= tolerance)
        & (relative_root.imag < 0)
    )
    relative_root = torch.where(flip_tie, -relative_root, relative_root)
    phase_midpoint = phase_a * relative_root
    magnitude = a.abs().sqrt() * b.abs().sqrt()
    return magnitude * phase_midpoint


class MAGPIEReconstructor(PIEReconstructor):
    """GM-rPIE (one level) and GM-MAGPIE (multiple object-grid levels).

    Supports one object slice and one shared coherent probe mode, without OPR
    or independent object/probe sampling. Both factors are updated in every
    random minibatch with unit step sizes. Native position refinement and
    shared constraints remain available. Unsupported capabilities fail at setup.

    Object and probe proposals use the same pre-update forward state. Their
    phase-aligned geometric means are synthesized with counterpart-intensity
    weights. Probe corrections are first transported by the native shift
    adjoint, then averaged using the intensity of each reverse-shifted object
    patch. This approximates the reference probe synthesis; a singleton gives
    exactly the direct adjoint correction when the intermediates are finite.
    """

    def __init__(self, parameter_group, dataset, options=None, *args, **kwargs):
        if options is None:
            options = MAGPIEReconstructorOptions()
        super().__init__(parameter_group, dataset, options, *args, **kwargs)

    def check_inputs(self, *args, **kwargs):
        # Data structures import the reconstructor registry during module setup.
        from ptychi.data_structures.object import DIPObject
        from ptychi.data_structures.probe import ProbeRepresentation

        super().check_inputs(*args, **kwargs)
        group = self.parameter_group
        object_, probe = group.object, group.probe
        if probe.representation != ProbeRepresentation.NORMAL or isinstance(object_, DIPObject):
            raise MAGPIECapabilityError(
                "GM-rPIE/GM-MAGPIE requires directly represented object and probe arrays. "
                "Use ordinary PlanarObject and Probe parameters."
            )
        object_data, probe_data = object_.data, probe.data
        _check_data_capabilities(object_data, probe_data, group.opr_mode_weights.data)
        _check_initial_factors(object_data, probe_data)
        if self.forward_model.resolved_shift_operator != enums.ShiftOperators.PATCH:
            raise MAGPIECapabilityError(
                "GM-rPIE/GM-MAGPIE require patch-based object sampling. "
                "FULL_OBJECT sampling, including automatic near-field selection, is unsupported; "
                "set reconstructor_options.forward_model_options.shift_operator="
                "ShiftOperators.PATCH."
            )
        if self.forward_model.resampling_enabled:
            raise MAGPIECapabilityError(
                "GM-rPIE/GM-MAGPIE does not support object/probe resampling. "
                "Use matching object and probe pixel sizes and aspect ratios, or select PIE."
            )
        if tuple(self.dataset.patterns.shape[-2:]) != tuple(probe_data.shape[-2:]):
            raise MAGPIECapabilityError(
                "GM-rPIE/GM-MAGPIE requires matching detector and probe shapes; received "
                f"{tuple(self.dataset.patterns.shape[-2:])} and {tuple(probe_data.shape[-2:])}. "
                "Supply matching shapes or select a reconstructor supporting detector cropping."
            )
        height, width = probe_data.shape[-2:]
        max_levels = 1
        while min(height, width) >= 2 and height % 2 == 0 and width % 2 == 0:
            height, width = height // 2, width // 2
            max_levels += 1
        if self.options.multigrid_levels > max_levels:
            raise MAGPIECapabilityError(
                f"multigrid_levels={self.options.multigrid_levels} exceeds the {max_levels} "
                f"dyadic levels supported by probe shape {tuple(probe_data.shape[-2:])}. "
                f"Choose a level count from 1 to {max_levels}."
            )
        self.num_levels = self.options.multigrid_levels

    def _proximal_step(
        self,
        z: torch.Tensor,
        q: torch.Tensor,
        target: torch.Tensor,
        q_power: torch.Tensor,
        regularizer: torch.Tensor,
    ) -> torch.Tensor:
        numerator = q.conj() * (target - q * z)
        return z + _safe_divide(numerator, q_power + regularizer)

    def _object_proposal(
        self,
        factor_old: torch.Tensor,
        counterpart_old: torch.Tensor,
        psi_prime: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Return a multigrid MAGPIE endpoint for one bilinear factor."""
        q_levels = [counterpart_old]
        for _ in range(1, self.num_levels):
            q_levels.append(_restrict_2x(q_levels[-1]))

        q_power = [_power(q) for q in q_levels]
        # Use the same minibatch-wide fine-grid normalization as rPIE. Besides
        # keeping every scan on a common scale, this makes a one-grid MAGPIE
        # cycle reduce directly to the corresponding rPIE endpoint.
        q_max = q_power[0].amax()
        regularizer = [alpha * (q_max - q_power[0]).clamp_min(0)]
        downsampled_power = []
        for level in range(1, self.num_levels):
            power_down = _restrict_2x(q_power[level - 1])
            downsampled_power.append(power_down)
            transfer = _safe_divide(q_power[level], power_down)
            transfer = torch.where(power_down > 0, transfer, torch.ones_like(transfer))
            regularizer.append(transfer * _restrict_2x(regularizer[level - 1]))

        z_levels = [factor_old]
        target_levels = [psi_prime]
        for level in range(1, self.num_levels):
            power_down = _prolong_2x(downsampled_power[level - 1])
            weight_z = _safe_divide(q_power[level - 1], power_down)
            weight_target = _safe_divide(
                _prolong_2x(q_levels[level]) * q_levels[level - 1].conj(),
                power_down,
            )
            restricted = _restrict_2x(
                torch.cat(
                    (
                        weight_z * z_levels[level - 1],
                        weight_target * target_levels[level - 1],
                    ),
                    dim=1,
                )
            )
            z_levels.append(restricted[:, :1])
            target_levels.append(restricted[:, 1:])

        level = self.num_levels - 1
        z_new = self._proximal_step(
            z_levels[level],
            q_levels[level],
            target_levels[level],
            q_power[level],
            regularizer[level],
        )
        for level in range(self.num_levels - 2, -1, -1):
            coarse_correction = _prolong_2x(z_new - z_levels[level + 1])
            z_new = z_levels[level] + coarse_correction
            z_new = self._proximal_step(
                z_new,
                q_levels[level],
                target_levels[level],
                q_power[level],
                regularizer[level],
            )
        return z_new

    def _probe_proposal(self, object_old: Tensor, probe_old: Tensor, delta_psi: Tensor) -> Tensor:
        """Return local rPIE probe endpoints using per-patch object normalization."""
        alpha = self.parameter_group.probe.options.alpha
        object_power = _power(object_old)
        object_max = object_power.amax(dim=(-2, -1), keepdim=True)
        denominator = (1 - alpha) * object_power + alpha * object_max
        return probe_old + _safe_divide(object_old.conj() * delta_psi, denominator)

    def _synthesize_object_update(self, indices, object_old, object_local, probe_counterpart):
        """Return the full object correction from normalized weighted patch sums."""
        positions = self.parameter_group.probe_positions.tensor[indices]
        weight = _power(probe_counterpart)
        numerator = self.forward_model.place_object_patches_on_probe_grid(
            positions, (weight * (object_local - object_old))[:, 0], integer_mode=True
        )
        denominator = self.forward_model.place_object_patches_on_probe_grid(
            positions, weight[:, 0], integer_mode=True
        )
        # Equal sampling is validated before construction of any updates. The
        # normalized ratio has no claimed extension to independently sampled grids.
        update = self.forward_model.object_update_to_native_grid(
            _safe_divide(numerator, denominator)
        )
        return update.unsqueeze(0)

    def _synthesize_probe_update(self, indices, probe_old, probe_local, object_counterpart):
        """Combine adjoint corrections using reverse-shifted object intensities.

        Minimize sum_k c_k * |delta - a_k|**2, with a_k = S_k* d_k and
        c_k = |T_k object_counterpart_k|**2. S_k* is the native probe-shift
        adjoint; T_k uses native forward interpolation with reversed shifts,
        replicate padding, and cropping. No confidence floor or clipping is needed.

        This is the original weighted residual in common probe coordinates
        for ideal translations. Discrete interpolation and padded boundaries
        make it approximate; descent of the original objective is not assured.
        """
        corrections = self.adjoint_shift_probe_update_direction(
            indices, probe_local - probe_old, first_mode_only=True
        )
        # A singleton consensus is exactly its adjoint correction, without reduction roundoff.
        if corrections.shape[0] == 1:
            return corrections[0]
        fractional_shifts = self.forward_model.get_probe_grid_position_data()[2][indices]
        object_on_probe_grid = ip.shift_images(
            object_counterpart[:, 0],
            -fractional_shifts,
            method=self.parameter_group.object.options.patch_interpolation_method,
            adjoint=False,
            pad=self.forward_model.pad_for_shift,
        )[:, None]
        confidence = object_on_probe_grid.abs().square()
        mean_correction = corrections.mean(dim=0)
        # Centering selects the ordinary mean wherever the total confidence is zero.
        return mean_correction + _safe_divide(
            (confidence * (corrections - mean_correction)).sum(dim=0),
            confidence.sum(dim=0),
        )

    @timer()
    @torch.no_grad()
    def compute_updates(self, indices: Tensor, y_true: Tensor):
        """Return ``((delta_object, delta_probe, delta_positions), intensities)``.

        The first two updates are already synthesized, unlike PIE's per-position
        probe updates. All directions are computed before mutating any parameter.
        """
        indices = indices.cpu()
        y_pred = self.forward_model.forward(indices)
        variables = self.forward_model.intermediate_variables
        object_old = variables["obj_patches"][:, :1]
        probe_old = variables.shifted_unique_probes[0]
        psi_prime = self.replace_propagated_exit_wave_magnitude(
            variables["psi_far"],
            y_true,
            constrained_pixel_mask=self.get_constrained_pixel_mask(y_true),
        )
        psi_prime = self.forward_model.free_space_propagator.propagate_backward(psi_prime)
        delta_psi = psi_prime - variables["psi"]
        object_plus = self._object_proposal(
            object_old, probe_old, psi_prime, self.parameter_group.object.options.alpha
        )
        probe_plus = self._probe_proposal(object_old, probe_old, delta_psi)
        object_local = aligned_geom_mean(object_old, object_plus)
        probe_local = aligned_geom_mean(probe_old, probe_plus)
        delta_o = self._synthesize_object_update(indices, object_old, object_local, probe_local)
        delta_p = self._synthesize_probe_update(indices, probe_old, probe_local, object_local)

        delta_pos = None
        positions = self.parameter_group.probe_positions
        if positions.optimization_enabled(self.current_epoch):
            delta_pos = torch.zeros_like(positions.data)
            direction = positions.position_correction.get_update(
                delta_psi,
                object_old,
                object_plus - object_old,
                probe_old,
                self.parameter_group.object.step_size,
            )
            delta_pos[indices] = self.forward_model.probe_displacements_to_object_pixels(direction)
        return (delta_o, delta_p, delta_pos), y_pred

    @timer()
    @torch.no_grad()
    def apply_updates(self, delta_o, delta_p_i, delta_pos, *args, **kwargs):
        """Apply synthesized factor corrections once; use the native position optimizer."""
        if delta_o is not None:
            self.parameter_group.object.set_data(delta_o, op="add")
        if delta_p_i is not None:
            self.parameter_group.probe.set_data(delta_p_i, slicer=0, op="add")
        if delta_pos is not None:
            positions = self.parameter_group.probe_positions
            positions.set_grad(-delta_pos)
            positions.step_optimizer()
