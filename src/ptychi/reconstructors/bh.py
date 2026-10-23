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
from ptychi.utils import get_default_complex_dtype
from ptychi.api.enums import ShiftOperators
import ptychi.forward_models as fm
import ptychi.image_proc as ip

if TYPE_CHECKING:
    import ptychi.api as api
    import ptychi.data_structures.parameter_group as pg


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
            self.eta_op = torch.empty(dataset.patterns.shape,dtype=get_default_complex_dtype())[:,None,:,:]
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
            object_.optimizer.step()

        if delta_p is not None:
            probe.set_grad(-delta_p)
            probe.optimizer.step()

        if delta_pos is not None:
            probe_positions.set_grad(-delta_pos)
            probe_positions.optimizer.step()

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

    def refresh_object_direction_patches(self, direction, positions, patches):
        r"""Resample a cached object search direction at the current positions.

        FULL_OBJECT returns :math:`\widetilde{\Delta O}(u+q_y-(p_y-1)/2,
        v+q_x-(p_x-1)/2)` on the fixed retained support, with zero padded
        margins. Here :math:`q` is the current absolute probe-grid center and
        :math:`p` is the probe shape. PATCH retains its cached ``patches``.
        """
        if self.forward_model.resolved_shift_operator != ShiftOperators.FULL_OBJECT:
            return patches
        return self.forward_model.object_position_derivative(positions, (0, 0), direction)

    def position_directional_derivative(
        self, patches, positions, direction, *, object_array=None, second_direction=None
    ):
        r"""Evaluate first or mixed second position derivatives for BH curvature.

        For each position, let :math:`S(q)` denote its sampled object patch
        and :math:`d=(d_y,d_x)` the supplied ``direction`` in probe-grid pixels.
        With no ``second_direction``, return

        .. math::

            D_d S = d_y\partial_{q_y}S + d_x\partial_{q_x}S.

        With ``second_direction`` :math:`e=(e_y,e_x)`, return

        .. math::

            D_e D_d S = d_y e_y\partial_{q_y}^2 S
                + (d_y e_x+d_x e_y)\partial_{q_y}\partial_{q_x}S
                + d_x e_x\partial_{q_x}^2 S.

        Directions are held fixed during differentiation. FULL_OBJECT uses
        derivatives of the complete object before cropping, dividing out the
        native-to-probe scale factors from ``object_position_derivative``.
        ``positions`` are in native object pixels; ``object_array`` optionally
        supplies a native-grid object update direction in place of the object.

        PATCH Fourier-differentiates ``patches`` on their own periodic grid:
        the multiplier is :math:`2\pi i(f_y d_y+f_x d_x)`, multiplied by
        :math:`2\pi i(f_y e_y+f_x e_x)` for the second derivative. The output
        retains shape ``(batch_size, n_slices, probe_height, probe_width)``.
        """
        if self.forward_model.resolved_shift_operator == ShiftOperators.FULL_OBJECT:
            scale = self.forward_model.probe_grid_scale if self.forward_model.resampling_enabled else (1, 1)
            def derivative(order):
                return self.forward_model.object_position_derivative(positions, order, object_array) / (
                    scale[0] ** order[0] * scale[1] ** order[1]
                )
            dy, dx = direction[:, 0, None, None, None], direction[:, 1, None, None, None]
            if second_direction is None:
                return dy * derivative((1, 0)) + dx * derivative((0, 1))
            ey = second_direction[:, 0, None, None, None]
            ex = second_direction[:, 1, None, None, None]
            return (
                dy * ey * derivative((2, 0))
                + (dy * ex + dx * ey) * derivative((1, 1))
                + dx * ex * derivative((0, 2))
            )
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

        if self.current_epoch == 0 or self.options.method == "GD":
            eta_o = -grad_o
            eta_op = -grad_op
        elif self.options.method == "CG":
            eta_o = self.eta_o
            eta_op = self.refresh_object_direction_patches(
                eta_o, self.positions, self.eta_op[self.indices]
            )

            beta = self.calc_beta_object(p, d, psi_far, grad_op, eta_op)
            eta_o = -grad_o + beta * eta_o
            eta_op = -grad_op + beta * eta_op

        if self.options.method == "CG":
            self.eta_o = eta_o
            self.eta_op[self.indices] = eta_op

        alpha = self.calc_alpha_object(p, grad_o, eta_o, d, psi_far, eta_op)

        # normalize with the batch size
        alpha *= op.shape[0] / self.batch_size

        delta_o = alpha * eta_o

        return delta_o

    def compute_updates_object_probe(self, op, p, psi_far, gradF, d):
        grad_o, grad_op = self.gradient_o(p, gradF)
        grad_p = self.parameter_group.probe.options.rho * self.gradient_p(op, gradF)

        if self.current_epoch == 0 or self.options.method == "GD":
            eta_o = -grad_o
            eta_op = -grad_op
            eta_p = -grad_p
        elif self.options.method == "CG":
            # intermediate variables for the CG
            eta_o = self.eta_o
            eta_op = self.refresh_object_direction_patches(
                eta_o, self.positions, self.eta_op[self.indices]
            )
            eta_p = self.eta_p

            beta = self.calc_beta_object_probe(
                p, grad_p, eta_p, d, gradF, psi_far, op, grad_op, eta_op
            )
            eta_o = -grad_o + beta * eta_o
            eta_op = -grad_op + beta * eta_op
            eta_p = -grad_p + beta * eta_p

        if self.options.method == "CG":
            self.eta_o = eta_o
            self.eta_op[self.indices] = eta_op
            self.eta_p = eta_p

        alpha = self.calc_alpha_object_probe(
            p, grad_o, grad_p, eta_o, eta_p, d, gradF, psi_far, op, eta_op
        )

        # normalize with the batch size
        alpha *= op.shape[0] / self.batch_size

        delta_o = alpha * eta_o
        delta_p = self.parameter_group.probe.options.rho * alpha * eta_p
        return (delta_o, delta_p)

    def compute_updates_object_probe_positions(self, op, p, pos, psi_far, gradF, d):
        grad_o, grad_op = self.gradient_o(p, gradF)
        grad_p = self.parameter_group.probe.options.rho * self.gradient_p(op, gradF)
        grad_pos = self.parameter_group.probe_positions.options.rho * self.gradient_pos(
            op, p, pos, gradF
        )

        if self.current_epoch == 0 or self.options.method == "GD":
            eta_o = -grad_o
            eta_op = -grad_op
            eta_p = -grad_p
            eta_pos = -grad_pos
        elif self.options.method == "CG":
            # intermediate variables for the CG
            eta_o = self.eta_o
            eta_op = self.refresh_object_direction_patches(
                eta_o, self.positions, self.eta_op[self.indices]
            )
            eta_p = self.eta_p
            eta_pos = self.eta_pos[self.indices]

            beta = self.calc_beta_object_probe_positions(
                p, grad_p, eta_p, grad_pos, eta_pos, d, gradF, psi_far, op, grad_op, eta_op,
                do1=grad_o, do2=eta_o,
            )
            eta_o = -grad_o + beta * eta_o
            eta_op = -grad_op + beta * eta_op
            eta_p = -grad_p + beta * eta_p
            eta_pos = -grad_pos + beta * eta_pos

        if self.options.method == "CG":
            self.eta_o = eta_o
            self.eta_op[self.indices] = eta_op
            self.eta_p = eta_p
            self.eta_pos[self.indices] = eta_pos

        alpha = self.calc_alpha_object_probe_positions(
            p, grad_o, grad_p, grad_pos, eta_o, eta_p, eta_pos, d, gradF, psi_far, op, eta_op
        )

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
        """Gradient with respect to positions"""

        direction = torch.zeros_like(pos)
        direction[:, 0] = 1
        dt1 = self.position_directional_derivative(op, pos, direction)
        dt2 = self.position_directional_derivative(op, pos, direction.flip(-1))

        grad_pos = torch.zeros_like(pos)
        grad_pos[:, 0] = redot(gradF, p * dt1, axis=(1, 2, 3))
        grad_pos[:, 1] = redot(gradF, p * dt2, axis=(1, 2, 3))

        return grad_pos

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
        self, p, do1, dp1, dpos1, do2, dp2, dpos2, d, gradF, psi_far, op, dop2
    ):
        """Step length for the object, probe and positions update"""

        top = -redot(do1, do2) - redot(dp1, dp2) - torch.sum(dpos1 * dpos2)

        dp2 = self.parameter_group.probe.options.rho * dp2
        dpos2 = self.parameter_group.probe_positions.options.rho * dpos2

        derivative = self.position_directional_derivative
        dt2 = derivative(dop2, self.positions, dpos2, object_array=do2)
        dt = derivative(op, self.positions, dpos2)
        d2t = derivative(op, self.positions, dpos2, second_direction=dpos2)

        dm2 = dp2 * op + p * (dop2 + dt)
        d2m2 = p * (2 * dt2 + d2t) + 2 * dp2 * dop2 + 2 * dp2 * dt

        Ldm2 = self.forward_model.free_space_propagator.propagate_forward(dm2)
        bottom = redot(gradF, d2m2) + self.hessianF(psi_far, Ldm2, Ldm2, d)
        return top / bottom

    def calc_beta_object(self, p, d, psi_far, dop1, dop2):
        """Step length for the CG direction with respect to the object"""

        dm1 = p * dop1
        dm2 = p * dop2

        Ldm1 = self.forward_model.free_space_propagator.propagate_forward(dm1)
        Ldm2 = self.forward_model.free_space_propagator.propagate_forward(dm2)

        top = self.hessianF(psi_far, Ldm1, Ldm2, d)
        bottom = self.hessianF(psi_far, Ldm2, Ldm2, d)
        return top / bottom

    def calc_beta_object_probe(self, p, dp1, dp2, d, gradF, psi_far, op, dop1, dop2):
        """Step length for the CG direction with respect to the object and probe"""

        dp1 = self.parameter_group.probe.options.rho * dp1
        dp2 = self.parameter_group.probe.options.rho * dp2

        dm1 = dp1 * op + p * dop1
        dm2 = dp2 * op + p * dop2

        d2m1 = dp1 * dop2 + dp2 * dop1
        d2m2 = 2 * dp2 * dop2

        Ldm1 = self.forward_model.free_space_propagator.propagate_forward(dm1)
        Ldm2 = self.forward_model.free_space_propagator.propagate_forward(dm2)

        top = redot(gradF, d2m1) + self.hessianF(psi_far, Ldm1, Ldm2, d)
        bottom = redot(gradF, d2m2) + self.hessianF(psi_far, Ldm2, Ldm2, d)
        return top / bottom

    def calc_beta_object_probe_positions(
        self, p, dp1, dp2, dpos1, dpos2, d, gradF, psi_far, op, dop1, dop2,
        *, do1=None, do2=None,
    ):
        """Step length for the CG direction with respect to the object, probe and positions"""

        dp1 = self.parameter_group.probe.options.rho * dp1
        dp2 = self.parameter_group.probe.options.rho * dp2

        dpos1 = self.parameter_group.probe_positions.options.rho * dpos1
        dpos2 = self.parameter_group.probe_positions.options.rho * dpos2

        derivative = self.position_directional_derivative
        dt12 = derivative(dop1, self.positions, dpos2, object_array=do1)
        dt21 = derivative(dop2, self.positions, dpos1, object_array=do2)
        dt22 = derivative(dop2, self.positions, dpos2, object_array=do2)
        dt1 = derivative(op, self.positions, dpos1)
        dt2 = derivative(op, self.positions, dpos2)
        d2t1 = derivative(op, self.positions, dpos1, second_direction=dpos2)
        d2t2 = derivative(op, self.positions, dpos2, second_direction=dpos2)

        dm1 = dp1 * op + p * (dop1 + dt1)
        dm2 = dp2 * op + p * (dop2 + dt2)

        d2m1 = p * (dt12 + dt21 + d2t1) + dp1 * (dop2 + dt2) + dp2 * (dop1 + dt1)
        d2m2 = p * (2 * dt22 + d2t2) + 2 * dp2 * dop2 + 2 * dp2 * dt2

        Ldm1 = self.forward_model.free_space_propagator.propagate_forward(dm1)
        Ldm2 = self.forward_model.free_space_propagator.propagate_forward(dm2)

        top = redot(gradF, d2m1) + self.hessianF(psi_far, Ldm1, Ldm2, d)
        bottom = redot(gradF, d2m2) + self.hessianF(psi_far, Ldm2, Ldm2, d)
        return top / bottom
