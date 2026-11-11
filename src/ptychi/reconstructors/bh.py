# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Optional, TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from ptychi.reconstructors.base import (
    AnalyticalIterativePtychographyReconstructor,
)
from ptychi.metrics import MSELossOfSqrt
from ptychi.maths import reprod, redot
from ptychi.api.enums import ShiftOperators
import ptychi.forward_models as fm
import ptychi.image_proc as ip

if TYPE_CHECKING:
    import ptychi.api as api
    import ptychi.data_structures.parameter_group as pg

FIRST_ORDERS = ((1, 0), (0, 1))
SECOND_ORDERS = ((2, 0), (1, 1), (0, 2))


class BHReconstructor(AnalyticalIterativePtychographyReconstructor):
    """
    Bilinear Hessian Reconstruction Method:
    Implements Gradient Descent and Conjugate Gradient algorithms (Carlsson & Nikitin, 2025, in prep).

    So far this implementation supports the reconstruction of the object, probe, and position parameters,
    including multiple mutually incoherent probe modes. Multiple OPR modes are not yet supported.

    The algorithm accepts a parameter `rho`, which controls the optimization of the probe and position updates.
    The value of `rho` is adjusted based on data size and the initial guess.
    Higher values of `rho` give more weight to the updates, while lower values reduce their influence on the reconstruction.

    It is important to adjust the `rho` values carefully to avoid converging to local minima.
    """

    parameter_group: "pg.PlanarPtychographyParameterGroup"

    def __init__(
        self,
        parameter_group: "pg.PlanarPtychographyParameterGroup",
        dataset: Dataset,
        options: Optional["api.options.bh.BHReconstructorOptions"] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            parameter_group=parameter_group,
            dataset=dataset,
            options=options,
            *args,
            **kwargs,
        )

        # constant to avoid divisions by 0 in calculations
        self.eps = 1e-8
        # cg varibales
        if self.options.method == "CG":
            self.eta_pos = torch.empty([dataset.patterns.shape[0],2])
            
    def build_forward_model(self):
        # BHReconstructor requires subpixel shifts on the object, not the probe.
        self.forward_model = fm.PlanarPtychographyForwardModel(
            parameter_group=self.parameter_group, 
            retain_intermediates=True,
            detector_size=tuple(self.dataset.patterns.shape[-2:]),
            wavelength_m=self.dataset.wavelength_m,
            free_space_propagation_distance_m=self.dataset.free_space_propagation_distance_m,
            pad_for_shift=self.options.forward_model_options.pad_for_shift,
            low_memory_mode=self.options.forward_model_options.low_memory_mode,
            shift_operator=self.options.forward_model_options.shift_operator,
            apply_subpixel_shifts_on_probe=False
        )

    def build_loss_tracker(self):
        if self.displayed_loss_function is None:
            self.displayed_loss_function = MSELossOfSqrt()
        return super().build_loss_tracker()

    def check_inputs(self, *args, **kwargs):
        if self.parameter_group.object.is_multislice:
            raise NotImplementedError("The BH only supports 2D objects.")
        if self.parameter_group.probe.has_multiple_opr_modes:
            raise NotImplementedError("The BH does not support multiple OPR modes yet.")

    def run_minibatch(self, input_data, y_true, *args, **kwargs):
        self.prepare_optimizer_closures(input_data, y_true)
        (delta_o, delta_p, delta_pos), y_pred = self.compute_updates(*input_data, y_true)
        self.apply_updates(delta_o, delta_p, delta_pos)
        self.loss_tracker.update_batch_loss_with_metric_function(y_pred, y_true)

    def orthogonalize_incoherent_probe_modes(self) -> None:
        probe = self.parameter_group.probe
        if self.options.method == "CG" and hasattr(self, "eta_p"):
            transform = probe.constrain_incoherent_modes_orthogonality(return_transform=True)
            if transform is not None:
                # Keep the previous search direction in the probe's current mode basis.
                self.eta_p = (transform @ self.eta_p.flatten(1)).reshape_as(self.eta_p)
        else:
            super().orthogonalize_incoherent_probe_modes()

    def apply_updates(self, delta_o, delta_p, delta_pos, *args, **kwargs):
        """
        Apply updates to optimizable parameters given the updates calculated by self.compute_updates.

        Parameters
        ----------
        delta_o : Tensor
            A (n_replica, h, w, 2) tensor of object update vector.
        delta_p : Tensor
            A (n_replicate, n_opr_modes, n_modes, h, w, 2) tensor of probe update vector.
        delta_pos : Tensor
            A (n_positions, 2) tensor of probe position vectors.
        """
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        probe_positions = self.parameter_group.probe_positions

        if delta_o is not None:
            object_.set_grad(-delta_o)
            object_.step_optimizer()

        if delta_p is not None:
            probe.set_grad(-delta_p)
            probe.step_optimizer()

        if delta_pos is not None:
            probe_positions.set_grad(-delta_pos)
            probe_positions.step_optimizer(clip_update=False)

    def compute_updates(
        self, indices: torch.Tensor, y_true: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """
        Calculates the updates of the whole object, the probe, and other parameters.
        This function is called in self.update_step_module.forward.
        """
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        probe_positions = self.parameter_group.probe_positions

        self.indices = indices.cpu()
        self.positions = probe_positions.tensor[indices]

        y = self.forward_model.forward(indices)
        op = self.forward_model.intermediate_variables["obj_patches"]

        psi_far = self.forward_model.intermediate_variables["psi_far"]
        p = probe.get_opr_mode(0)  # shape (n_modes, h, w); all incoherent probe modes
        pos = self.positions
        self.current_constrained_pixel_mask = self.get_constrained_pixel_mask(y_true)[:, None]

        # sqrt of data
        d = torch.sqrt(y_true)[:, torch.newaxis]

        # Gradient for the Gaussian model
        gradF = self.gradientF(psi_far, d)

        # shortcuts
        o_opt = object_.optimization_enabled(self.current_epoch)
        p_opt = probe.optimization_enabled(self.current_epoch)
        pos_opt = probe_positions.optimization_enabled(self.current_epoch)

        # Calculate updates either of 3 problems: (o,p,pos), (o,p), or (o)
        if o_opt and p_opt and pos_opt:
            (delta_o, delta_p, delta_pos0) = self.compute_updates_object_probe_positions(
                op, p, pos, psi_far, gradF, d
            )

            # operations from the original code
            delta_o = delta_o.unsqueeze(0)
            delta_p_all_modes = delta_p[None, :, :]
            delta_pos = torch.zeros_like(probe_positions.data)
            delta_pos[indices] = self.forward_model.probe_displacements_to_object_pixels(
                delta_pos0
            )

        elif o_opt and p_opt and (not pos_opt):
            (delta_o, delta_p) = self.compute_updates_object_probe(op, p, psi_far, gradF, d)
            delta_pos = None

            # operations from the original code
            delta_o = delta_o.unsqueeze(0)
            delta_p_all_modes = delta_p[None, :, :]

        elif o_opt:
            delta_o = self.compute_updates_object(op, p, psi_far, gradF, d)
            delta_p_all_modes = None
            delta_pos = None

            # operations from the original code
            delta_o = delta_o.unsqueeze(0)

        return (delta_o, delta_p_all_modes, delta_pos), y

    def object_direction_patches(self, direction):
        """Sample a native-grid object direction at the current positions."""
        if self.forward_model.resolved_shift_operator == ShiftOperators.FULL_OBJECT:
            return self.forward_model.object_sampling_derivative(self.positions, (0, 0), direction)
        if self.forward_model.resampling_enabled:
            direction = ip.fourier_resize(direction, self.forward_model.probe_grid_object_shape)
        return self.forward_model.extract_probe_grid_patches(
            direction,
            self.positions,
            self.parameter_group.probe.get_spatial_shape(),
            integer_mode=False,
        )[:, None]

    def position_directional_derivative(
        self, patches, positions, direction, *, object_array=None, second_direction=None
    ):
        """First (D_d S) or mixed second (D_e D_d S) position derivative of sampled patches."""
        if self.forward_model.resolved_shift_operator == ShiftOperators.FULL_OBJECT:
            dy, dx = direction[:, 0, None, None, None], direction[:, 1, None, None, None]
            if second_direction is None:
                sy, sx = self.position_derivatives(patches, positions, FIRST_ORDERS, object_array)
                return dy * sy + dx * sx
            ey = second_direction[:, 0, None, None, None]
            ex = second_direction[:, 1, None, None, None]
            syy, sxy, sxx = self.position_derivatives(
                patches, positions, SECOND_ORDERS, object_array
            )
            return dy * ey * syy + (dy * ex + dx * ey) * sxy + dx * ex * sxx
        fy = torch.fft.fftfreq(patches.shape[-2], device=patches.device)
        fx = torch.fft.fftfreq(patches.shape[-1], device=patches.device)
        def factor(vector):
            return 2j * torch.pi * (
                fy[None, None, :, None] * vector[:, 0, None, None, None]
                + fx[None, None, None, :] * vector[:, 1, None, None, None]
            )
        multiplier = factor(direction)
        if second_direction is not None:
            multiplier = multiplier * factor(second_direction)
        return torch.fft.ifft2(torch.fft.fft2(patches) * multiplier)

    def compute_updates_object(self, op, p, psi_far, gradF, d):
        grad_o, grad_op = self.gradient_o(p, gradF)

        if self.options.method == "CG" and self.current_epoch > 0:
            eta_o = self.eta_o
            eta_op = self.object_direction_patches(eta_o)
            Qgg, Bge, Qee = self.calc_hessian3_object(p, d, psi_far, grad_op, eta_op)
            beta = Bge / Qee
            eta_o = -grad_o + beta * eta_o
            eta_op = -grad_op + beta * eta_op

            top = -redot(grad_o, eta_o)
            alpha = self.fused_step_length(top, beta, Qgg, Bge, Qee)
        else:
            eta_o = -grad_o
            eta_op = -grad_op
            alpha = self.calc_alpha_object(p, grad_o, eta_o, d, psi_far, eta_op)

        if self.options.method == "CG":
            self.eta_o = eta_o

        # normalize with the batch size
        alpha *= op.shape[0] / self.batch_size

        delta_o = alpha * eta_o

        return delta_o

    def compute_updates_object_probe(self, op, p, psi_far, gradF, d):
        grad_o, grad_op = self.gradient_o(p, gradF)
        grad_p = self.parameter_group.probe.options.rho * self.gradient_p(op, gradF)

        if self.options.method == "CG" and self.current_epoch > 0:
            eta_o = self.eta_o
            eta_op = self.object_direction_patches(eta_o)
            eta_p = self.eta_p
            Qgg, Bge, Qee = self.calc_hessian3_object_probe(
                p, grad_p, eta_p, d, gradF, psi_far, op, grad_op, eta_op
            )
            beta = Bge / Qee
            eta_o = -grad_o + beta * eta_o
            eta_op = -grad_op + beta * eta_op
            eta_p = -grad_p + beta * eta_p

            top = -redot(grad_o, eta_o) - redot(grad_p, eta_p)
            alpha = self.fused_step_length(top, beta, Qgg, Bge, Qee)
        else:
            eta_o = -grad_o
            eta_op = -grad_op
            eta_p = -grad_p
            alpha = self.calc_alpha_object_probe(
                p, grad_o, grad_p, eta_o, eta_p, d, gradF, psi_far, op, eta_op
            )

        if self.options.method == "CG":
            self.eta_o = eta_o
            self.eta_p = eta_p

        # normalize with the batch size
        alpha *= op.shape[0] / self.batch_size

        delta_o = alpha * eta_o
        delta_p = self.parameter_group.probe.options.rho * alpha * eta_p
        return (delta_o, delta_p)

    def compute_updates_object_probe_positions(self, op, p, pos, psi_far, gradF, d):
        grad_o, grad_op = self.gradient_o(p, gradF)
        grad_p = self.parameter_group.probe.options.rho * self.gradient_p(op, gradF)
        grad_pos, op_derivs = self.gradient_pos(op, p, pos, gradF)
        grad_pos = self.parameter_group.probe_positions.options.rho * grad_pos

        if self.options.method == "CG" and self.current_epoch > 0:
            eta_o = self.eta_o
            eta_op = self.object_direction_patches(eta_o)
            eta_p = self.eta_p
            eta_pos = self.eta_pos[self.indices]
            Qgg, Bge, Qee = self.calc_hessian3_object_probe_positions(
                p,
                grad_p,
                eta_p,
                grad_pos,
                eta_pos,
                d,
                gradF,
                psi_far,
                op,
                grad_op,
                eta_op,
                op_derivs,
                grad_o,
                eta_o,
            )
            beta = Bge / Qee
            eta_o = -grad_o + beta * eta_o
            eta_op = -grad_op + beta * eta_op
            eta_p = -grad_p + beta * eta_p
            eta_pos = -grad_pos + beta * eta_pos

            top = -redot(grad_o, eta_o) - redot(grad_p, eta_p) - torch.sum(grad_pos * eta_pos)
            alpha = self.fused_step_length(top, beta, Qgg, Bge, Qee)
        else:
            eta_o = -grad_o
            eta_op = -grad_op
            eta_p = -grad_p
            eta_pos = -grad_pos
            alpha = self.calc_alpha_object_probe_positions(
                p, grad_o, grad_p, grad_pos, eta_o, eta_p, eta_pos, d, gradF, psi_far, op, eta_op,
                op_derivs,
            )
        del op_derivs

        if self.options.method == "CG":
            self.eta_o = eta_o
            self.eta_p = eta_p
            self.eta_pos[self.indices] = eta_pos

        # normalize with the batch size
        alpha *= op.shape[0] / self.batch_size

        delta_o = alpha * eta_o
        delta_p = self.parameter_group.probe.options.rho * alpha * eta_p
        delta_pos = self.parameter_group.probe_positions.options.rho * alpha * eta_pos

        return (delta_o, delta_p, delta_pos)

    def gradientF(self, psi_far, d):
        """
        Gradient for the Gaussian model.

        `psi_far` carries a probe-mode axis (dim=1) of N_mu mutually incoherent
        modes; the measured intensity is their incoherent sum, so the amplitude
        entering the cost function is the joint norm A = sqrt(sum_mu |psi_far_mu|^2)
        rather than the per-mode modulus. This reduces to the single-mode formula
        when N_mu == 1.
        """

        A = torch.sqrt(torch.sum(torch.abs(psi_far) ** 2, dim=1, keepdim=True)) + self.eps
        td = d * (psi_far / A)
        td = psi_far - td
        td = td * self.current_constrained_pixel_mask
        # Compensate FFT normalization only for the far-field Fourier propagator.
        if isinstance(self.forward_model.free_space_propagator, fm.FourierPropagator):
            td *= psi_far.shape[-1] * psi_far.shape[-2]
        res = 2 * self.forward_model.free_space_propagator.propagate_backward(td)
        return res

    def hessianF(self, psi_far, psi_far1, psi_far2, data):
        """
        Hessian for the Gaussian model.

        As in `gradientF`, the amplitude is the joint norm A across probe modes
        (dim=1). The diagonal term is a plain per-mode sum, but the data term
        couples all modes through the shared amplitude: the mode axis must be
        summed *before* the two factors are multiplied together.
        """

        A = torch.sqrt(torch.sum(torch.abs(psi_far) ** 2, dim=1, keepdim=True)) + self.eps
        l0 = psi_far / A
        d0 = data / A
        v1 = torch.sum(self.current_constrained_pixel_mask * (1 - d0) * reprod(psi_far1, psi_far2))
        c1 = torch.sum(reprod(l0, psi_far1), dim=1, keepdim=True)
        c2 = torch.sum(reprod(l0, psi_far2), dim=1, keepdim=True)
        v2 = torch.sum(
            self.current_constrained_pixel_mask
            * d0
            * c1
            * c2
        )
        return 2 * (v1 + v2)

    def hessianF3(self, psi_far, psi_far1, psi_far2, data):
        """`hessianF` for (1, 1), (1, 2), (2, 2) in one pass."""

        A = torch.sqrt(torch.sum(torch.abs(psi_far) ** 2, dim=1, keepdim=True)) + self.eps
        l0 = psi_far / A
        d0 = data / A
        mask = self.current_constrained_pixel_mask
        w1 = mask * (1 - d0)
        w2 = mask * d0
        del A, d0
        c1 = torch.sum(reprod(l0, psi_far1), dim=1, keepdim=True)
        c2 = torch.sum(reprod(l0, psi_far2), dim=1, keepdim=True)
        del l0
        h11 = torch.sum(w1 * reprod(psi_far1, psi_far1)) + torch.sum(w2 * c1 * c1)
        h12 = torch.sum(w1 * reprod(psi_far1, psi_far2)) + torch.sum(w2 * c1 * c2)
        h22 = torch.sum(w1 * reprod(psi_far2, psi_far2)) + torch.sum(w2 * c2 * c2)
        return 2 * h11, 2 * h12, 2 * h22

    def fused_step_length(self, top, beta, Qgg, Bge, Qee):
        """Step length along eta_new = -g + beta*eta, using
        B(eta_new, eta_new) = Qgg - 2 beta Bge + beta^2 Qee; non-positive curvature
        is replaced by its absolute value to keep the step downhill."""
        bottom = beta * beta * Qee - 2 * beta * Bge + Qgg
        scale = torch.abs(beta * beta * Qee) + torch.abs(2 * beta * Bge) + torch.abs(Qgg)
        fallback = torch.where(bottom != 0, torch.abs(bottom), 1e-6 * scale)
        bottom = torch.where(bottom > 1e-6 * scale, bottom, fallback)
        return top / bottom

    def gradient_o(self, p, gradF):
        """
        Gradient with respect to the object.

        Summed over probe modes (dim=1), since the object is illuminated by
        every mutually incoherent mode. Contrast with `gradient_p`, where each
        probe mode keeps its own, unsummed gradient.
        """

        tmp = torch.conj(p) * gradF
        tmp = torch.sum(tmp, dim=1)

        grad_probe_grid = self.forward_model.place_object_patches_on_probe_grid(
            self.positions,
            tmp,
            integer_mode=False,
        )
        grad_o = self.forward_model.object_update_to_native_grid(grad_probe_grid)
        grad_o_on_probe_grid = grad_probe_grid
        if self.forward_model.resampling_enabled:
            # Evaluate curvature along the gradient applied on the native object grid.
            grad_o_on_probe_grid = ip.fourier_resize(
                grad_o, self.forward_model.probe_grid_object_shape
            )
        patches = self.forward_model.extract_probe_grid_patches(
            grad_o_on_probe_grid,
            self.positions,
            self.parameter_group.probe.get_spatial_shape(),
            integer_mode=False,
        )
        patches = patches[:, None]
        return grad_o, patches

    def gradient_p(self, op, gradF):
        """Gradient with respect to the probe"""

        return torch.sum(torch.conj(op) * gradF, axis=0)

    def gradient_pos(self, op, p, pos, gradF):
        """Gradient with respect to positions, and the first derivatives (dS/dq_y, dS/dq_x)."""

        op_derivs = tuple(self.position_derivatives(op, pos, FIRST_ORDERS))
        grad_pos = torch.stack([redot(gradF, p * t, axis=(1, 2, 3)) for t in op_derivs], dim=-1)
        return grad_pos, op_derivs

    def calc_alpha_object(self, p, do1, do2, d, psi_far, dop2):
        """Step length for the object update"""

        top = -redot(do1, do2)

        dm2 = p * dop2
        Ldm2 = self.forward_model.free_space_propagator.propagate_forward(dm2)
        bottom = self.hessianF(psi_far, Ldm2, Ldm2, d)
        return top / bottom

    def calc_alpha_object_probe(self, p, do1, dp1, do2, dp2, d, gradF, psi_far, op, dop2):
        """Step length for the object and probe update"""

        top = -redot(do1, do2) - redot(dp1, dp2)
        dp2 = self.parameter_group.probe.options.rho * dp2

        dm2 = dp2 * op + p * dop2
        d2m2 = 2 * dp2 * dop2
        Ldm2 = self.forward_model.free_space_propagator.propagate_forward(dm2)
        bottom = redot(gradF, d2m2) + self.hessianF(psi_far, Ldm2, Ldm2, d)
        return top / bottom

    def calc_alpha_object_probe_positions(
        self, p, do1, dp1, dpos1, do2, dp2, dpos2, d, gradF, psi_far, op, dop2, op_derivs
    ):
        """Step length for the object, probe and positions update"""

        top = -redot(do1, do2) - redot(dp1, dp2) - torch.sum(dpos1 * dpos2)

        dp2 = self.parameter_group.probe.options.rho * dp2
        dpos2 = self.parameter_group.probe_positions.options.rho * dpos2

        sy, sx = op_derivs
        a2 = dop2 + dpos2[:, 0, None, None, None] * sy + dpos2[:, 1, None, None, None] * sx
        Ldm2 = self.forward_model.free_space_propagator.propagate_forward(dp2 * op + p * a2)
        bottom = self.hessianF(psi_far, Ldm2, Ldm2, d)
        del Ldm2

        derivative = self.position_directional_derivative
        s = 2 * derivative(dop2, self.positions, dpos2, object_array=do2)
        s = s + derivative(op, self.positions, dpos2, second_direction=dpos2)
        bottom += redot(gradF, p * s + 2 * dp2 * a2)
        return top / bottom

    def calc_hessian3_object(self, p, d, psi_far, dop1, dop2):
        """{B(g, g), B(g, eta), B(eta, eta)} for the object."""

        propagate = self.forward_model.free_space_propagator.propagate_forward
        Ldm1 = propagate(p * dop1)
        Ldm2 = propagate(p * dop2)
        return self.hessianF3(psi_far, Ldm1, Ldm2, d)

    def calc_hessian3_object_probe(self, p, dp1, dp2, d, gradF, psi_far, op, dop1, dop2):
        """{B(g, g), B(g, eta), B(eta, eta)} for the object and probe."""

        dp1 = self.parameter_group.probe.options.rho * dp1
        dp2 = self.parameter_group.probe.options.rho * dp2

        propagate = self.forward_model.free_space_propagator.propagate_forward
        Ldm1 = propagate(dp1 * op + p * dop1)
        Ldm2 = propagate(dp2 * op + p * dop2)
        h11, h12, h22 = self.hessianF3(psi_far, Ldm1, Ldm2, d)
        del Ldm1, Ldm2

        h11 += redot(gradF, 2 * dp1 * dop1)
        h12 += redot(gradF, dp1 * dop2 + dp2 * dop1)
        h22 += redot(gradF, 2 * dp2 * dop2)
        return h11, h12, h22

    def calc_hessian3_object_probe_positions(
        self, p, dp1, dp2, dpos1, dpos2, d, gradF, psi_far, op, dop1, dop2, op_derivs, do1, do2
    ):
        """{B(g, g), B(g, eta), B(eta, eta)} for the object, probe and positions."""

        dp1 = self.parameter_group.probe.options.rho * dp1
        dp2 = self.parameter_group.probe.options.rho * dp2

        dpos1 = self.parameter_group.probe_positions.options.rho * dpos1
        dpos2 = self.parameter_group.probe_positions.options.rho * dpos2

        propagate = self.forward_model.free_space_propagator.propagate_forward

        # first order: dop_i + D_i op
        sy, sx = op_derivs
        a1 = dop1 + dpos1[:, 0, None, None, None] * sy + dpos1[:, 1, None, None, None] * sx
        a2 = dop2 + dpos2[:, 0, None, None, None] * sy + dpos2[:, 1, None, None, None] * sx
        Ldm1 = propagate(dp1 * op + p * a1)
        Ldm2 = propagate(dp2 * op + p * a2)
        h11, h12, h22 = self.hessianF3(psi_far, Ldm1, Ldm2, d)
        del Ldm1, Ldm2

        # second order: <gradF, p D_j dop_i> and <gradF, p D_i D_j op> from per-position
        # projections of the base derivatives
        r_op = self.position_projections(gradF, p, op, SECOND_ORDERS)
        r1 = self.position_projections(gradF, p, dop1, FIRST_ORDERS, do1)
        r2 = self.position_projections(gradF, p, dop2, FIRST_ORDERS, do2)

        h11 += 2 * torch.sum(r1 * dpos1) + self.second_order_term(r_op, dpos1, dpos1)
        h11 += redot(gradF, 2 * dp1 * a1)
        h12 += torch.sum(r1 * dpos2) + torch.sum(r2 * dpos1)
        h12 += self.second_order_term(r_op, dpos1, dpos2)
        h12 += redot(gradF, dp1 * a2 + dp2 * a1)
        h22 += 2 * torch.sum(r2 * dpos2) + self.second_order_term(r_op, dpos2, dpos2)
        h22 += redot(gradF, 2 * dp2 * a2)
        return h11, h12, h22

    def position_derivatives(self, patches, positions, orders, object_array=None):
        """Yield d^(m+n) S / dq_y^m dq_x^n of the sampled object for each (m, n) in orders."""
        if self.forward_model.resolved_shift_operator == ShiftOperators.FULL_OBJECT:
            for order in orders:
                yield self.forward_model.object_sampling_derivative(positions, order, object_array)
            return
        ky = 2j * torch.pi * torch.fft.fftfreq(patches.shape[-2], device=patches.device)
        kx = 2j * torch.pi * torch.fft.fftfreq(patches.shape[-1], device=patches.device)
        ky, kx = ky[:, None], kx[None, :]
        f = torch.fft.fft2(patches)
        for m, n in orders:
            yield torch.fft.ifft2(f * (ky**m * kx**n))

    def position_projections(self, gradF, p, patches, orders, object_array=None):
        """Per-position <gradF, p * d^(m+n) S> for each order, shape (batch, len(orders))."""
        return torch.stack([
            redot(gradF, p * t, axis=(1, 2, 3))
            for t in self.position_derivatives(patches, self.positions, orders, object_array)
        ], dim=-1)

    def second_order_term(self, r, a, b):
        """sum_n of <gradF, p D_a D_b S> from second-derivative projections r (batch, 3)."""
        return torch.sum(
            a[:, 0] * b[:, 0] * r[:, 0]
            + (a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0]) * r[:, 1]
            + a[:, 1] * b[:, 1] * r[:, 2]
        )
