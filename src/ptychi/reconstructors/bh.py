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
import ptychi.forward_models as fm

if TYPE_CHECKING:
    import ptychi.api as api
    import ptychi.data_structures.parameter_group as pg


class BHReconstructor(AnalyticalIterativePtychographyReconstructor):
    """
    Bilinear Hessian Reconstruction Method:
    Implements Gradient Descent and Conjugate Gradient algorithms (Carlsson & Nikitin, 2025, in prep).

    So far this implementation supports the reconstruction of the object, probe, and position parameters.
    Probe multi modes need to be implemented.

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
        p = probe.get_opr_mode(0)  # to do for multi modes
        pos = self.positions

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
            delta_pos[indices] = delta_pos0

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

    def compute_updates_object(self, op, p, psi_far, gradF, d):
        grad_o, grad_op = self.gradient_o(p, gradF)

        if self.current_epoch == 0 or self.options.method == "GD":
            eta_o = -grad_o
            eta_op = -grad_op
        elif self.options.method == "CG":
            eta_o = self.eta_o
            eta_op = self.eta_op[self.indices]

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
            eta_op = self.eta_op[self.indices]
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
            eta_op = self.eta_op[self.indices]
            eta_p = self.eta_p
            eta_pos = self.eta_pos[self.indices]

            beta = self.calc_beta_object_probe_positions(
                p, grad_p, eta_p, grad_pos, eta_pos, d, gradF, psi_far, op, grad_op, eta_op
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
        """Gradient for the Guassian model"""

        td = d * (psi_far / (torch.abs(psi_far) + self.eps))
        td = psi_far - td
        # Compensate FFT normalization only for the far-field Fourier propagator.
        if isinstance(self.forward_model.free_space_propagator, fm.FourierPropagator):
            td *= psi_far.shape[-1] * psi_far.shape[-2]
        res = 2 * self.forward_model.free_space_propagator.propagate_backward(td)
        return res

    def hessianF(self, psi_far, psi_far1, psi_far2, data):
        """Hessian for the Guassian model"""

        l0 = psi_far / (torch.abs(psi_far) + self.eps)
        d0 = data / (torch.abs(psi_far) + self.eps)
        v1 = torch.sum((1 - d0) * reprod(psi_far1, psi_far2))
        v2 = torch.sum(d0 * reprod(l0, psi_far1) * reprod(l0, psi_far2))
        return 2 * (v1 + v2)

    def gradient_o(self, p, gradF):
        """Gradient with respect to the object"""

        tmp = torch.conj(p) * gradF

        obj = self.parameter_group.object
        o = obj.place_patches_on_empty_buffer(
            self.positions, 
            tmp[:, 0], 
            pad_for_shift=self.options.forward_model_options.pad_for_shift
        )
        patches = obj.extract_patches_function(
            o, self.positions + obj.pos_origin_coords, 
            self.parameter_group.probe.get_spatial_shape(), 
            pad=self.options.forward_model_options.pad_for_shift
        )
        patches = patches[:, None]
        return o, patches

    def gradient_p(self, op, gradF):
        """Gradient with respect to the probe"""

        return torch.sum(torch.conj(op) * gradF, axis=0)

    def gradient_pos(self, op, p, pos, gradF):
        """Gradient with respect to positions"""

        xi1 = torch.fft.fftfreq(op.shape[-1])
        xi2, xi1 = torch.meshgrid(xi1, xi1, indexing="xy")

        tmp = torch.fft.fft2(op)
        dt1 = 2 * torch.pi * 1j * torch.fft.ifft2(xi1 * tmp)
        dt2 = 2 * torch.pi * 1j * torch.fft.ifft2(xi2 * tmp)

        grad_pos = torch.zeros([len(pos), 2])
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

        xi1 = torch.fft.fftfreq(p.shape[-1])
        [xi2, xi1] = torch.meshgrid(xi1, xi1, indexing="xy")

        dpos2 = dpos2[:, :, None, None, None]
        w1 = xi1 * dpos2[:, 0] + xi2 * dpos2[:, 1]
        w2 = (
            xi1**2 * dpos2[:, 0] ** 2
            + 2 * xi1 * xi2 * (dpos2[:, 0] * dpos2[:, 1])
            + xi2**2 * dpos2[:, 1] ** 2
        )

        tmp = torch.fft.fft2(dop2)
        dt2 = 2 * torch.pi * 1j * torch.fft.ifft2(w1 * tmp)

        tmp = torch.fft.fft2(op)
        dt = 2 * torch.pi * 1j * torch.fft.ifft2(w1 * tmp)
        d2t = -4 * torch.pi**2 * torch.fft.ifft2(w2 * tmp)

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
        self, p, dp1, dp2, dpos1, dpos2, d, gradF, psi_far, op, dop1, dop2
    ):
        """Step length for the CG direction with respect to the object, probe and positions"""

        dp1 = self.parameter_group.probe.options.rho * dp1
        dp2 = self.parameter_group.probe.options.rho * dp2

        dpos1 = self.parameter_group.probe_positions.options.rho * dpos1
        dpos2 = self.parameter_group.probe_positions.options.rho * dpos2

        xi1 = torch.fft.fftfreq(p.shape[-1])
        [xi2, xi1] = torch.meshgrid(xi1, xi1, indexing="xy")

        dpos1 = dpos1[:, :, None, None, None]
        dpos2 = dpos2[:, :, None, None, None]

        w1 = xi1 * dpos1[:, 0] + xi2 * dpos1[:, 1]
        w2 = xi1 * dpos2[:, 0] + xi2 * dpos2[:, 1]
        w12 = (
            xi1**2 * dpos1[:, 0] * dpos2[:, 0]
            + xi1 * xi2 * (dpos1[:, 0] * dpos2[:, 1] + dpos1[:, 1] * dpos2[:, 0])
            + xi2**2 * dpos1[:, 1] * dpos2[:, 1]
        )
        w22 = (
            xi1**2 * dpos2[:, 0] ** 2
            + 2 * xi1 * xi2 * (dpos2[:, 0] * dpos2[:, 1])
            + xi2**2 * dpos2[:, 1] ** 2
        )

        tmp = torch.fft.fft2(dop1)
        dt12 = 2 * torch.pi * 1j * torch.fft.ifft2(w2 * tmp)

        tmp = torch.fft.fft2(dop2)
        dt21 = 2 * torch.pi * 1j * torch.fft.ifft2(w1 * tmp)
        dt22 = 2 * torch.pi * 1j * torch.fft.ifft2(w2 * tmp)

        tmp = torch.fft.fft2(op)
        dt1 = 2 * torch.pi * 1j * torch.fft.ifft2(w1 * tmp)
        dt2 = 2 * torch.pi * 1j * torch.fft.ifft2(w2 * tmp)

        d2t1 = -4 * torch.pi**2 * torch.fft.ifft2(w12 * tmp)
        d2t2 = -4 * torch.pi**2 * torch.fft.ifft2(w22 * tmp)

        dm1 = dp1 * op + p * (dop1 + dt1)
        dm2 = dp2 * op + p * (dop2 + dt2)

        d2m1 = p * (dt12 + dt21 + d2t1) + dp1 * (dop2 + dt2) + dp2 * (dop1 + dt1)
        d2m2 = p * (2 * dt22 + d2t2) + 2 * dp2 * dop2 + 2 * dp2 * dt2

        Ldm1 = self.forward_model.free_space_propagator.propagate_forward(dm1)
        Ldm2 = self.forward_model.free_space_propagator.propagate_forward(dm2)

        top = redot(gradF, d2m1) + self.hessianF(psi_far, Ldm1, Ldm2, d)
        bottom = redot(gradF, d2m2) + self.hessianF(psi_far, Ldm2, Ldm2, d)
        return top / bottom
