# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Optional, TYPE_CHECKING
import logging
import math

import torch
from torch.utils.data import Dataset
import torch.distributed as dist

from ptychi.reconstructors.base import (
    AnalyticalIterativePtychographyReconstructor,
    LossTracker,
)
import ptychi.forward_models as fm
import ptychi.maths as pmath
import ptychi.api.enums as enums
from ptychi.timing.timer_utils import timer
import ptychi.image_proc as ip
from ptychi.parallel import MultiprocessMixin

if TYPE_CHECKING:
    import ptychi.data_structures.parameter_group as pg
    import ptychi.api as api

logger = logging.getLogger(__name__)


class LSQMLReconstructor(AnalyticalIterativePtychographyReconstructor):
    """
    The least square maximum likelihood (LSQ-ML) algorithm described in

    Odstrčil, M., Menzel, A., & Guizar-Sicairos, M. (2018). Iterative
    least-squares solver for generalized maximum-likelihood ptychography.
    Optics Express, 26(3), 3108–3123. doi:10.1364/oe.26.003108

    This implementation uses automatic differentiation to get necessary gradients,
    but other steps, including the solving of the step size, are done analytically.
    """

    parameter_group: "pg.PlanarPtychographyParameterGroup"
    options: "api.LSQMLReconstructorOptions"

    def __init__(
        self,
        parameter_group: "pg.PlanarPtychographyParameterGroup",
        dataset: Dataset,
        options: Optional["api.LSQMLReconstructorOptions"] = None,
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

        noise_model_params = (
            {} if options.noise_model == "poisson" else {"sigma": options.gaussian_noise_std}
        )
        self.noise_model = {
            "gaussian": fm.PtychographyGaussianNoiseModel,
            "poisson": fm.PtychographyPoissonNoiseModel,
        }[options.noise_model](
            **noise_model_params, valid_pixel_mask=self.dataset.valid_pixel_mask.clone()
        )

        self.alpha_psi_far = 0.5
        self.alpha_psi_far_all_pos = None

        self.indices = []

        self.object_momentum_params = {}
        self.probe_momentum_params = {}


        # Fourier error for momentum acceleration.
        self.accumulated_fourier_error = 0.0
        
        # Create buffers.
        alpha_object_all_pos_all_slices = torch.ones(
            [self.parameter_group.probe_positions.shape[0], self.parameter_group.object.n_slices],
            device=torch.get_default_device(),
        )
        alpha_probe_all_pos = torch.ones(
            self.parameter_group.probe_positions.shape[0], device=torch.get_default_device()
        )
        self.reconstructor_buffers.register_buffer("alpha_probe_all_pos", alpha_probe_all_pos, dist.ReduceOp.SUM)
        self.reconstructor_buffers.register_buffer("alpha_object_all_pos_all_slices", alpha_object_all_pos_all_slices, dist.ReduceOp.SUM)
        self.reconstructor_buffers.register_buffer("fourier_errors", [], dist.ReduceOp.AVG)
        self.reconstructor_buffers.register_buffer("accumulated_true_intensity", 0, dist.ReduceOp.SUM)
        self.reconstructor_buffers.register_buffer("accumulated_pred_intensity", 0, dist.ReduceOp.SUM)

    def check_inputs(self, *args, **kwargs):
        if self.parameter_group.opr_mode_weights.optimizer is not None:
            logger.warning(
                "Selecting optimizer for OPRModeWeights is not supported for "
                "LSQMLReconstructor and will be disregarded."
            )
        if (
            self.parameter_group.opr_mode_weights.n_opr_modes > 1
            and self.parameter_group.opr_mode_weights.data[:, 1:].abs().max() < 1e-9
        ):
            raise ValueError(
                "Weights of eigenmodes (the second and following OPR modes) in LSQMLReconstructor "
                "should not be all zero, which can cause numerical instability!"
            )

    def build(self) -> None:
        super().build()
        self.build_cached_variables()
        self.build_noise_model()

    def build_loss_tracker(self):
        f = (
            self.noise_model.nll
            if self.displayed_loss_function is None
            else self.displayed_loss_function
        )
        self.loss_tracker = LossTracker(metric_function=f)

    def build_noise_model(self):
        self.noise_model = self.noise_model.to(torch.get_default_device())

    def build_cached_variables(self):
        self.alpha_psi_far_all_pos = torch.full(
            size=(self.parameter_group.probe_positions.shape[0],), fill_value=0.5
        )

    @timer()
    def get_psi_far_step_size(self, y_pred, y_true, indices, eps=1e-5):
        if isinstance(self.noise_model, fm.PtychographyGaussianNoiseModel):
            alpha = torch.tensor(0.5, device=y_pred.device)  # Eq. 16
        elif isinstance(self.noise_model, fm.PtychographyPoissonNoiseModel):
            # This implementation reproduces PtychoShelves (gradient_descent_xi_solver)
            # and is different from Eq. 17 of Odstrcil (2018).
            xi = 1 - y_true / (y_pred + eps)
            for _ in range(2):
                alpha_prev = self.alpha_psi_far_all_pos[indices].mean()
                alpha = (xi * (y_pred - y_true / (1 - alpha_prev * xi))).sum(-1).sum(-1)
                alpha = alpha / (xi**2 * y_pred).sum(-1).sum(-1)
                # Use previous step size as momentum.
                alpha = 0.5 * alpha_prev + 0.5 * alpha
                alpha = alpha.clamp(0, 1)
                self.alpha_psi_far_all_pos[indices] = alpha
            # Add perturbation.
            alpha = alpha + torch.randn(alpha.shape, device=alpha.device) * 1e-2
            self.alpha_psi_far_all_pos[indices] = alpha
        return alpha
    
    @timer()
    def compute_reconstruction_parameter_updates(self, y_pred, y_true, indices):
        """Calculate the update vectors for unknown parameters. These update
        vectors are stored in the `grad` attribute of the corresponding
        `ReconstructionParameter` objects. Actual updates are NOT performed here.
        """
        psi_opt = self.run_reciprocal_space_step(y_pred, y_true, indices)
        self.run_real_space_step(psi_opt, indices)

    @timer()
    def run_reciprocal_space_step(self, y_pred, y_true, indices):
        """
        Run step 1 of LSQ-ML, which updates `psi`.

        Returns
        -------
        psi_opt : Tensor
            A (batch_size, n_probe_modes, h, w) complex tensor.
        """
        # gradient as in Eq. 12a/b
        psi_far_0 = self.forward_model.intermediate_variables["psi_far"]
        dl_dpsi_far = self.noise_model.backward_to_psi_far(y_pred, y_true, psi_far_0)
        self.alpha_psi_far = self.get_psi_far_step_size(y_pred, y_true, indices)
        psi_far = psi_far_0 - self.alpha_psi_far.view(-1, 1, 1, 1) * dl_dpsi_far  # Eq. 14

        psi_opt = self.forward_model.free_space_propagator.propagate_backward(psi_far)
        return psi_opt

    @timer()
    def run_real_space_step(self, psi_opt, indices):
        """
        Run real space step of LSQ-ML, which updates the object, probe, and other variables
        using psi updated in the reciprocal space step and backpropagated to real space.

        Parameters
        ----------
        psi_opt : Tensor
            A (batch_size, n_probe_modes, h, w) complex tensor.
            Should be the psi updated in the reciprocal space step.
        """
        positions = self.forward_model.intermediate_variables["positions"]
        psi_0 = self.forward_model.intermediate_variables["psi"]
        # Shape of chi:           (batch_size, n_probe_modes, h, w)
        chi = psi_opt - psi_0  # Eq, 19
        obj_patches = self.forward_model.intermediate_variables["obj_patches"]

        self.calculate_update_vectors(indices, chi, obj_patches, positions)

    @timer()
    def calculate_update_vectors(self, indices, chi, obj_patches, positions):
        """
        Calculate the update vectors for the object, probe, and other reconstruction
        parameters. These update vectors are stored in the `grad` attribute of the
        corresponding `ReconstructionParameter` objects. Actual updates are NOT
        performed here.

        Parameters
        ----------
        indices : Tensor.
            A tensor of indices of diffraction patterns processed in the current batch.
        chi : Tensor.
            A (batch_size, n_modes, h, w) complex tensor giving the exit wave difference
            (psi_opt - psi_0) at the exit plane.
        obj_patches : Tensor.
            A (batch_size, h, w) complex tensor giving the object patches.
        positions : Tensor.
            A (batch_size, 2) tensor giving the probe positions in the current batch.
        gamma : float
            Damping factor for solving the step size linear equations.
        """
        object_ = self.parameter_group.object
        self._initialize_object_gradient()
        self.parameter_group.probe.initialize_grad()
        self.parameter_group.probe_positions.initialize_grad()
        self.parameter_group.opr_mode_weights.initialize_grad()
        self._initialize_object_step_size_buffer()
        self._initialize_probe_step_size_buffer()
        self._initialize_momentum_buffers()

        for i_slice in range(object_.n_slices - 1, -1, -1):
            if i_slice < object_.n_slices - 1:
                chi = self.forward_model.propagate_to_previous_slice(chi, slice_index=i_slice + 1)

            # Get unique probes, or the wavefield at the current slice before modulation.
            probe_current_slice = self._get_incident_wavefields_for_slice(i_slice)

            # Caluclate various object-related update directions.
            delta_o_comb, delta_o_precond, delta_o_i, delta_o_i_mode_0 = (
                self.calculate_object_update_directions(
                    chi, probe_current_slice, positions, i_slice
                )
            )
            # Record object update directions.
            # In compact batching mode, object is updated at the end of an epoch using gradients
            # accumulated over all minibatches.
            if self.options.batching_mode in [
                enums.BatchingModes.RANDOM,
                enums.BatchingModes.UNIFORM,
            ]:
                self._record_object_slice_gradient(i_slice, delta_o_precond, add_to_existing=False)
            else:
                self._record_object_slice_gradient(i_slice, delta_o_comb, add_to_existing=False)

            # Calculate probe update direction.
            delta_p_i_before_adj_shift = self._calculate_probe_update_direction(
                chi, obj_patches=obj_patches, slice_index=i_slice, probe_mode_index=None
            )  # Eq. 24a
            delta_p_i = self.adjoint_shift_probe_update_direction(
                indices, delta_p_i_before_adj_shift, first_mode_only=True
            )
            delta_p_hat = self._precondition_probe_update_direction(delta_p_i)  # Eq. 25a
            self._record_probe_gradient(delta_p_hat)

            # Calculate update vectors for OPR modes and weights.
            if i_slice == 0:
                if self.parameter_group.opr_mode_weights.optimization_enabled(self.current_epoch):
                    self.parameter_group.opr_mode_weights.update_variable_probe(
                        self.parameter_group.probe,
                        indices,
                        chi,
                        delta_p_i,
                        delta_p_hat,
                        obj_patches,
                        self.current_epoch,
                        probe_mode_index=0,
                        apply_updates=False,
                    )

            # Update buffered data for momentum acceleration.
            if i_slice == 0:
                self._update_momentum_buffers(delta_p_hat)

            # Calculate optimal step sizes.
            self.calculate_optimal_step_sizes(
                indices,
                chi,
                obj_patches,
                delta_o_i,
                delta_p_hat,
                delta_o_i_mode_0,
                probe=probe_current_slice,
                slice_index=i_slice,
            )

            if self.parameter_group.probe_positions.optimization_enabled(
                self.current_epoch
            ) and i_slice == self.parameter_group.probe_positions.get_slice_for_correction(
                object_.n_slices
            ):
                mean_alpha_o = pmath.trim_mean(
                    self.reconstructor_buffers.alpha_object_all_pos_all_slices[indices, i_slice], 0.1, dim=0
                )
                self.update_probe_positions(
                    chi,
                    indices,
                    obj_patches[:, i_slice : i_slice + 1],
                    mean_alpha_o * delta_o_i,
                    self.forward_model.intermediate_variables.shifted_unique_probes[i_slice],
                    apply_updates=False,
                )

            # Set chi to conjugate-modulated wavefield.
            chi = delta_p_i_before_adj_shift

    @timer()
    def apply_reconstruction_parameter_updates(self, indices: torch.Tensor):
        """Perform updates for reconstruction parameters using the
        update vectors stored in the `grad` attribute of the corresponding
        `ReconstructionParameter` objects.

        Parameters
        ----------
        indices : torch.Tensor
            The indices of the diffraction patterns processed in the current batch.
        """
        # Update object.
        mean_alpha_o_all_slices = pmath.trim_mean(
            self.reconstructor_buffers.alpha_object_all_pos_all_slices[indices], 0.1, dim=0
        )
        if self.parameter_group.object.optimization_enabled(
            self.current_epoch
        ) and self.options.batching_mode in [
            enums.BatchingModes.RANDOM,
            enums.BatchingModes.UNIFORM,
        ]:
            self._apply_object_update(mean_alpha_o_all_slices, None)
            
        # Update probe.
        alpha_p_i = self.reconstructor_buffers.alpha_probe_all_pos[indices]
        if self.parameter_group.probe.optimization_enabled(self.current_epoch):
            self._apply_probe_update(alpha_p_i, -self.parameter_group.probe.get_grad()[0])
            
        # Update probe positions.
        if self.parameter_group.probe_positions.optimization_enabled(self.current_epoch):
            self.parameter_group.probe_positions.step_optimizer()
            
        # Update OPR modes and weights.
        if self.parameter_group.opr_mode_weights.optimization_enabled(self.current_epoch):
            self.parameter_group.opr_mode_weights.step_optimizer()

    @timer()
    def _get_incident_wavefields_for_slice(self, i_slice):
        r"""
        Get the incident wavefields for a specified slice. For the first slice,
        this is just the unique probes. For other slices in a multislice object,
        this is the wavefield at the given slice before modulation.

        Parameters
        ----------
        i_slice : int
            The current slice index.
        indices : Tensor
            A tensor of indices of diffraction patterns processed in the current batch.

        Returns
        -------
        Tensor
            A (batch_size, n_modes, h, w) tensor giving the incident wavefields.
        """
        incident_wavefields = self.forward_model.intermediate_variables.shifted_unique_probes[
            i_slice
        ]
        return incident_wavefields

    @timer()
    def calculate_object_update_directions(self, chi, probe, positions, i_slice=0):
        """
        Calculate various update directions related to the object. These include
        the preconditioned patch-wise and combined object update directions,
        calculated with the first and all probe modes, respectively.

        Parameters
        ----------
        chi : Tensor
            A (batch_size, n_probe_modes, h, w) tensor giving the exit wave update.
        probe : Tensor
            A (batch_size, n_probe_modes, h, w) tensor giving the unique probe for
            each point.
        positions : Tensor
            A (batch_size, 2) tensor giving the probe positions.
        i_slice : int, optional
            The object slice index.

        Returns
        -------
        delta_o_comb : Tensor
            A (1, h, w) tensor giving the combined object update direction, but not
            yet preconditioned.
        delta_o_precond : Tensor
            A (1, h, w) tensor giving the combined and preconditioned object update
            direction.
        delta_o_i : Tensor
            A (batch_size, 1, h, w) tensor giving the object patch update directions
            after preconditioning.
        delta_o_i_mode_0 : Tensor, optional
            A (batch_size, 1, h, w) tensor giving the object patch update directions
            for the first probe mode after preconditioning (used for step size
            calculation).
        """
        if (
            self.options.solve_step_sizes_only_using_first_probe_mode
            or not self.parameter_group.object.options.multimodal_update
        ):
            # If object step size is to be solved with only the first probe mode,
            # then the delta_o_i used should also be calculated using only the first
            # probe mode.
            delta_o_i_mode_0_raw = self._calculate_object_patch_update_direction(
                chi, incident_wavefields=probe, probe_mode_index=0
            )
            delta_o_comb_mode_0 = self._combine_object_patch_update_directions(
                delta_o_i_mode_0_raw, positions, onto_accumulated=True, slice_index=i_slice
            )
            _, delta_o_i_mode_0 = self._precondition_object_update_direction(
                delta_o_comb_mode_0, 
                positions, 
                alpha_mix=self.options.preconditioning_damping_factor
            )
        else:
            delta_o_i_mode_0 = None

        # Calculate object update direction and precondition it.
        if self.parameter_group.object.options.multimodal_update:
            delta_o_i_raw = self._calculate_object_patch_update_direction(
                chi, incident_wavefields=probe, probe_mode_index=None
            )
            delta_o_comb = self._combine_object_patch_update_directions(
                delta_o_i_raw, positions, onto_accumulated=True, slice_index=i_slice
            )
        else:
            delta_o_i_raw = delta_o_i_mode_0_raw
            delta_o_comb = delta_o_comb_mode_0
        delta_o_precond, delta_o_i = self._precondition_object_update_direction(
            delta_o_comb, 
            positions, 
            alpha_mix=self.options.preconditioning_damping_factor
        )
        return delta_o_comb, delta_o_precond, delta_o_i, delta_o_i_mode_0

    @timer()
    def calculate_optimal_step_sizes(
        self,
        indices,
        chi,
        obj_patches,
        delta_o_i,
        delta_p_hat,
        delta_o_i_mode_0,
        probe=None,
        slice_index=0,
    ):
        """Calculate the optimal step sizes for the object and probem, and update
        the step size buffers.

        Parameters
        ----------
        indices : Tensor
            A tensor of indices of diffraction patterns processed in the current batch.
        chi : Tensor
            A (batch_size, n_probe_modes, h, w) tensor giving the exit wave difference.
        obj_patches : Tensor
            A (batch_size, h, w) tensor giving the object patches.
        delta_o_i : Tensor
            A (batch_size, 1, h, w) tensor giving the object patch update directions,
            calculated using all probe modes.
        delta_p_hat : Tensor
            A (n_probe_modes, h, w) tensor giving the preconditioned probe update direction.
        delta_o_i_mode_0 : Tensor, optional
            A (batch_size, 1, h, w) tensor giving the object patch update directions,
            calculated using the first probe mode.
        probe : Tensor, optional
            A (batch_size, n_probe_modes, h, w) tensor giving the probe.
        slice_index : int, optional
            The slice index of the object.
        """
        object_ = self.parameter_group.object
        if (not object_.is_multislice) or (
            object_.is_multislice
            and slice_index == 0
            and self.options.solve_obj_prb_step_size_jointly_for_first_slice_in_multislice
        ):
            (alpha_o_i, alpha_p_i) = self.calculate_object_and_probe_update_step_sizes(
                chi,
                obj_patches,
                (
                    delta_o_i_mode_0
                    if self.options.solve_step_sizes_only_using_first_probe_mode
                    else delta_o_i
                ),
                delta_p_hat,
                probe=probe,
                probe_mode_index=(
                    0 if self.options.solve_step_sizes_only_using_first_probe_mode else None
                ),
            )
        else:
            alpha_o_i = self.calculate_object_update_step_sizes(
                chi,
                (
                    delta_o_i_mode_0
                    if self.options.solve_step_sizes_only_using_first_probe_mode
                    else delta_o_i
                ),
                probe=probe,
                probe_mode_index=(
                    0 if self.options.solve_step_sizes_only_using_first_probe_mode else None
                ),
            )
            alpha_p_i = self.calculate_probe_update_step_sizes(
                chi,
                obj_patches,
                delta_p_hat,
                probe_mode_index=(
                    0 if self.options.solve_step_sizes_only_using_first_probe_mode else None
                ),
            )

        self.reconstructor_buffers.alpha_object_all_pos_all_slices[indices, slice_index] = alpha_o_i
        self.reconstructor_buffers.alpha_probe_all_pos[indices] = alpha_p_i

    @timer()
    def calculate_object_and_probe_update_step_sizes(
        self,
        chi,
        obj_patches,
        delta_o_i,
        delta_p_hat,
        probe=None,
        slice_index=0,
        probe_mode_index=None,
    ):
        """
        Jointly calculate the update step sizes for object and probe according to Eq. 22 of Odstrcil (2018).
        This routine builds a (batch_size, 2, 2) batch matrix, batch-invert them to get the update step sizes.
        """
        mode_slicer = self.parameter_group.probe._get_probe_mode_slicer(probe_mode_index)

        obj_patches = obj_patches[:, slice_index]
        delta_o_i = delta_o_i[:, 0]

        if probe is None:
            probe = self.forward_model.intermediate_variables.shifted_unique_probes[0]
        # When no OPR mode is present, probe is (n_modes, h, w). We add a batch dimension here.
        if probe.ndim == 3:
            probe = probe[None, ...]

        probe = probe[:, mode_slicer]
        chi = chi[:, mode_slicer]
        # TODO: consolidate
        delta_p_hat = delta_p_hat[None, ...]
        if delta_p_hat.shape[1] > 1:
            delta_p_hat = delta_p_hat[:, mode_slicer]

        lambda_0 = 1.2e-7 / (probe.shape[-2] * probe.shape[-1])
        lambda_lsq = 0.1

        # Shape of delta_p_o/o_p:     (batch_size, n_probe_modes or 1, h, w)
        delta_p_o = delta_p_hat * obj_patches[:, None, :, :]
        delta_o_patches_p = delta_o_i[:, None, :, :] * probe

        # Shape of aij:               (batch_size,)
        a11 = torch.sum((delta_o_patches_p.abs() ** 2 + lambda_0), dim=(-1, -2, -3))
        a11 = a11 + lambda_lsq * torch.mean(a11, dim=0)
        a12 = torch.sum((delta_o_patches_p * delta_p_o.conj()), dim=(-1, -2, -3))
        a21 = a12.conj()
        a22 = torch.sum((delta_p_o.abs() ** 2 + lambda_0), dim=(-1, -2, -3))
        a22 = a22 + lambda_lsq * torch.mean(a22, dim=0)
        b1 = torch.sum(torch.real(delta_o_patches_p.conj() * chi), dim=(-1, -2, -3))
        b2 = torch.sum(torch.real(delta_p_o.conj() * chi), dim=(-1, -2, -3))

        a_mat = torch.stack([a11, a12, a21, a22], dim=1).view(-1, 2, 2)
        b_vec = torch.stack([b1, b2], dim=1).view(-1, 2).type(a_mat.dtype)
        alpha_vec = torch.linalg.solve(a_mat, b_vec)
        alpha_vec = alpha_vec.real.clip(0, None)

        alpha_o_i = alpha_vec[:, 0]
        alpha_p_i = alpha_vec[:, 1]

        alpha_o_i = alpha_o_i * self.parameter_group.object.options.optimal_step_size_scaler
        alpha_p_i = alpha_p_i * self.parameter_group.probe.options.optimal_step_size_scaler

        alpha_o_i = alpha_o_i / self.parameter_group.object.n_slices
        if self.parameter_group.object.options.multimodal_update:
            alpha_o_i = alpha_o_i / self.parameter_group.probe.n_modes

        return alpha_o_i, alpha_p_i

    @timer()
    def calculate_object_update_step_sizes(self, chi, delta_o_i, probe=None, probe_mode_index=None):
        """
        Calculate the update step sizes just for the object using Eq. 23b of Odstrcil (2018).
        """
        # Just take the first slice.
        delta_o_i = delta_o_i[:, 0]

        mode_slicer = self.parameter_group.probe._get_probe_mode_slicer(probe_mode_index)

        if probe is None:
            probe = self.forward_model.intermediate_variables.shifted_unique_probes[0]

        probe = probe[:, mode_slicer]
        chi = chi[:, mode_slicer]

        # Shape of delta_p_o/o_p:     (batch_size, n_probe_modes or 1, h, w)
        delta_o_patches_p = delta_o_i[:, None, :, :] * probe

        numerator = 0.5 * torch.sum(torch.real(delta_o_patches_p.conj() * chi), dim=(-1, -2, -3))
        denominator = torch.sum(delta_o_patches_p.abs() ** 2, dim=(-1, -2, -3))

        alpha_o_i = numerator / denominator
        alpha_o_i = alpha_o_i * self.parameter_group.object.options.optimal_step_size_scaler
        alpha_o_i = alpha_o_i / self.parameter_group.object.n_slices
        if self.parameter_group.object.options.multimodal_update:
            alpha_o_i = alpha_o_i / self.parameter_group.probe.n_modes

        alpha_o_i = alpha_o_i.clamp(0, None)

        return alpha_o_i

    @timer()
    def calculate_probe_update_step_sizes(
        self, chi, obj_patches, delta_p_hat, probe_mode_index=None
    ):
        """
        Calculate the update step sizes just for the probe using Eq. 23a of Odstrcil (2018).
        """
        # Just take the first slice.
        obj_patches = obj_patches[:, 0]

        mode_slicer = self.parameter_group.probe._get_probe_mode_slicer(probe_mode_index)

        delta_p_hat = delta_p_hat[None, mode_slicer]
        chi = chi[:, mode_slicer]

        # Shape of delta_p_o/o_p:     (batch_size, n_probe_modes, h, w)
        delta_p_o = delta_p_hat * obj_patches[:, None, :, :]

        # Shape of aij:               (batch_size,)
        numerator = 0.5 * torch.sum(torch.real(delta_p_o.conj() * chi), dim=(-1, -2, -3))
        denominator = torch.sum(delta_p_o.abs() ** 2, dim=(-1, -2, -3))

        alpha_p_i = numerator / denominator
        alpha_p_i = alpha_p_i * self.parameter_group.probe.options.optimal_step_size_scaler

        alpha_p_i = alpha_p_i.clamp(0, None)

        return alpha_p_i

    @timer()
    def _calculate_probe_update_direction(
        self, chi, obj_patches=None, slice_index=0, probe_mode_index=None
    ):
        """
        Calculate probe update direction using Eq. 24a of Odstrcil (2018).

        Parameters
        ----------
        chi: torch.Tensor
            A (batch_size, n_probe_modes, h, w) tensor giving the difference of exit waves.
        obj_patches: torch.Tensor
            A (batch_size, h, w) tensor giving the object patches. If None, just return
            chi as it is. This behavior is intended for multislice.
        slice: int
            The slice of the object patches used to calculate the update direction.
        """
        mode_slicer = self.parameter_group.probe._get_probe_mode_slicer(probe_mode_index)

        if obj_patches is not None:
            obj_patches = obj_patches[:, slice_index]
            delta_p = chi[:, mode_slicer] * obj_patches.conj()[:, None, :, :]  # Eq. 24a
        else:
            delta_p = chi[:, mode_slicer]
        return delta_p

    @timer()
    def _precondition_probe_update_direction(self, delta_p):
        """
        Eq. 25a of Odstrcil, 2018.

        Parameters
        ----------
        delta_p : Tensor
            A (batch_size, n_probe_modes, h, w) tensor giving the probe update direction.

        Returns
        -------
        Tensor
            A (n_probe_modes, h, w) tensor giving the preconditioned probe update direction.
        """
        # Shape of delta_p_hat:  (n_probe_modes, h, w)
        delta_p_hat = torch.sum(delta_p, dim=0)  # Eq. 25a
        # PtychoShelves code simply takes the average. This is different from the paper
        # which does delta_p_hat = delta_p_hat / ((object_.abs() ** 2).sum() + delta),
        # but this seems to work better.
        delta_p_hat = delta_p_hat / delta_p.shape[0]
        return delta_p_hat

    @timer()
    def _apply_probe_update(self, alpha_p_i, delta_p_hat, probe_mode_index=None):
        """
        Apply update to the probe.
        
        Parameters
        ----------
        alpha_p_i : torch.Tensor
            A (batch_size,) tensor giving the probe step size calculated for 
            each diffraction pattern.
        delta_p_hat : torch.Tensor
            A (n_probe_modes, h, w) tensor giving the probe update direction.
        probe_mode_index : int, optional
            The index of the probe mode to update. If None, all probe modes are updated.
        """
        # PtychoShelves code simply multiplies delta_p_hat with averaged step size.
        # This is different from the paper which does the following:
        #     update_vec = delta_p_hat * obj_patches[:, None, :, :].abs() ** 2
        #     update_vec = update_vec * alpha_p_i[:, None, None, None]
        #     update_vec = update_vec / ((obj_patches.abs() ** 2).sum(0) + delta)

        # Just apply the update to the main OPR mode of each incoherent mode.
        # To do this, we pad the update vector with zeros in the OPR mode dimension.
        mode_slicer = self.parameter_group.probe._get_probe_mode_slicer(probe_mode_index)

        if self.options.batching_mode == enums.BatchingModes.COMPACT:
            # In compact mode, object is updated only once per epoch. To match the probe to this,
            # we divide the probe step size by the number of minibatches before each probe update.
            alpha_p_i = alpha_p_i / len(self.dataloader)
        alpha_p_mean = torch.mean(alpha_p_i)
        self.parameter_group.probe.set_grad(-delta_p_hat * alpha_p_mean, slicer=(0, mode_slicer))
        self.parameter_group.probe.optimizer.step()

    @timer()
    def _apply_probe_momentum(self, alpha_p_mean, delta_p_hat):
        """
        Apply momentum acceleration to the probe (only the first OPR mode). This is a
        special momentum acceleration used in PtychoShelves, which behaves somewhat
        differently from the momentum in `torch.optim.SGD`.

        Parameters
        ----------
        alpha_p_mean: float
            A scalar giving the mean probe step size.
        delta_p_hat: torch.Tensor
            A (n_probe_modes, h, w) tensor giving the accumulated probe update direction
            of the first OPR mode.
        """
        delta_p_hat = delta_p_hat * alpha_p_mean

        probe = self.parameter_group.probe
        if "update_direction_history" not in self.probe_momentum_params.keys():
            self.probe_momentum_params["update_direction_history"] = []
            self.probe_momentum_params["velocity_map"] = torch.zeros_like(delta_p_hat)

        upd = delta_p_hat / (pmath.mnorm(delta_p_hat, dim=(-1, -2), keepdims=True) + 1e-15)
        self.probe_momentum_params["update_direction_history"].append(upd)

        momentum_memory = 3

        if len(self.probe_momentum_params["update_direction_history"]) > momentum_memory + 1:
            # Remove the oldest momentum update.
            self.probe_momentum_params["update_direction_history"].pop(0)
            # PtychoShelves only applies momentum to the first mode.
            for i_mode in range(1):
                # Project older updates to the latest one, ordered from recent to old.
                projected_updates = [
                    (
                        self.probe_momentum_params["update_direction_history"][i][i_mode]
                        * self.probe_momentum_params["update_direction_history"][-1][i_mode].conj()
                    )
                    for i in range(len(self.probe_momentum_params["update_direction_history"]) - 1)
                ][::-1]
                # Shape of corr_level: (momentum_memory, n_probe_modes)
                corr_level = [
                    torch.mean(projected_updates[k]).real for k in range(len(projected_updates))
                ]
                corr_level = torch.tensor(corr_level, device=delta_p_hat.device)

                if self._fourier_error_ok() and torch.all(corr_level > 0):
                    # Estimate optimal friction.
                    p = pmath.polyfit(
                        torch.arange(0.0, momentum_memory + 1),
                        torch.concat([torch.zeros([1]), torch.log(corr_level)], dim=0).reshape(-1),
                        deg=1,
                    )
                    friction = 0.5 * (-p[0]).clip(0, None)

                    m = self.options.momentum_acceleration_gradient_mixing_factor
                    m = friction if m is None else m
                    self.probe_momentum_params["velocity_map"][i_mode] = (
                        1 - friction
                    ) * self.probe_momentum_params["velocity_map"][i_mode] + m * delta_p_hat[i_mode]
                    probe.set_data(
                        probe.data[0, i_mode]
                        + self.options.momentum_acceleration_gain
                        * self.probe_momentum_params["velocity_map"][i_mode],
                        slicer=(0, i_mode),
                    )
                else:
                    self.probe_momentum_params["velocity_map"][i_mode] = (
                        self.probe_momentum_params["velocity_map"][i_mode] / 2.0
                    )

    @timer()
    def _calculate_object_patch_update_direction(
        self, chi, incident_wavefields=None, probe_mode_index=None
    ):
        r"""
        Calculate the update direction for object patches, implementing
        Eq. 24b of Odstrcil, 2018. This function works in both 2D mode and
        multislice mode:

        - When `incident_wavefields` is None, 2D mode is assumed. `chi` is multiplied with the
            complex conjugate of the probe.
        - When `incident_wavefields` is not None, multislice mode is assumed. `chi` is multiplied
            with the complex conjugate of `incident_wavefields`.

        Parameters
        ----------
        indices : Tensor
            Indices of diffraction patterns in the current batch.
        chi : Tensor
            A (batch_size, n_modes, h, w) tensor giving the difference of exit waves.
            For multislice, this should be the exiting-plane `chi` backpropagated to the
            current slice.
        incident_wavefields : Tensor
            A (batch_size, n_modes, h, w) tensor giving $\psi_{i - 1}$, the wavefield
            modulated by the previous slice and propagated to the current slice -- in
            other words, the incident wavefield on the current slice. This quantity is
            used in lieu of the probe if given.

        Returns
        -------
        Tensor
            A (batch_size, 1, h, w) tensor giving the update direction for object patches.
            The dimension of size 1 is to match the slice dimension in the object patch
            tensor.
        """
        mode_slicer = self.parameter_group.probe._get_probe_mode_slicer(probe_mode_index)

        if incident_wavefields is None:
            p = self.forward_model.intermediate_variables.shifted_unique_probes[0]
        else:
            p = incident_wavefields

        if p.ndim == 3:
            p = p[None, ...]

        p = p[:, mode_slicer]
        chi = chi[:, mode_slicer]
        # Shape of chi:          (batch_size, n_probe_modes, h, w)
        # Shape delta_o_patches: (batch_size, h, w)
        # Multiply and sum over probe mode dimension
        delta_o_patches = torch.sum(chi * p.conj(), dim=1)  # Eq. 24b

        # Add slice dimension.
        return delta_o_patches[:, None, :, :]

    @timer()
    def _combine_object_patch_update_directions(
        self, delta_o_patches, positions, onto_accumulated=False, slice_index=0
    ):
        """
        Combine the update directions of object patches into a buffer with the
        same size as the whole object.

        Parameters
        ----------
        delta_o_patches : Tensor
            A (batch_size, 1, h, w) tensor giving the update direction for object patches.
        onto_accumulated : bool
            If True, add the update direction to the accumulated update direction stored in
            `object.grad`. Otherwise, just return the update direction accumulated on an empty
            buffer.

        Returns
        -------
        Tensor
            A (1, h, w) tensor giving the combined update direction for the whole object.
        """
        delta_o_patches = delta_o_patches[:, 0]

        # Stitch all delta O patches on the object buffer
        # Shape of delta_o_hat:  (h_whole, w_whole)
        delta_o_hat = self.parameter_group.object.place_patches_on_empty_buffer(
            positions.round().int(), delta_o_patches, integer_mode=True
        )
        delta_o_hat = delta_o_hat[None, ...]
        if onto_accumulated:
            delta_o_hat = delta_o_hat + (
                -self.parameter_group.object.get_grad()[slice_index : slice_index + 1]
            )
        return delta_o_hat

    @timer()
    def _precondition_object_update_direction(
        self, delta_o_hat, positions=None, alpha_mix=0.1, slice_index=0
    ):
        """
        Eq. 25b of Odstrcil, 2018.

        Returns
        -------
        Tensor
            A (1, h, w) tensor giving the preconditioned update direction for the whole object.
        Tensor
            A (batch_size, 1, h, w) tensor giving the preconditioned update direction for object patches.
            Only returned when `positions` is not None.
        """
        delta_o_hat = delta_o_hat[slice_index]

        preconditioner = self.parameter_group.object.preconditioner
        delta_o_hat = delta_o_hat / torch.sqrt(
            preconditioner**2 + (preconditioner.max() * alpha_mix) ** 2
        )

        # Re-extract delta O patches
        if positions is not None:
            delta_o_patches = ip.extract_patches_integer(
                delta_o_hat,
                positions.round().int() + self.parameter_group.object.pos_origin_coords,
                self.parameter_group.probe.shape[-2:],
            )

            return delta_o_hat[None, ...], delta_o_patches[:, None, :, :]
        return delta_o_hat[None, ...]

    @timer()
    def _precondition_accumulated_object_update_direction(self):
        """
        Sequentially precondition the object update direction accumulated over minibatches
        and stored in `object.grad`. This is only used in compact batching mode, where the
        object is only updated at the end of each epoch.
        """
        delta_o_hat_full = []
        for i_slice in range(self.parameter_group.object.n_slices):
            delta_o_hat = self._precondition_object_update_direction(
                -self.parameter_group.object.get_grad()[i_slice : i_slice + 1], 
                positions=None,
                alpha_mix=self.options.preconditioning_damping_factor
            )
            delta_o_hat_full.append(delta_o_hat)
        delta_o_hat_full = torch.cat(delta_o_hat_full, dim=0)
        return delta_o_hat_full

    @timer()
    def _initialize_object_gradient(self):
        """
        Initialize object gradient with zeros. This method is called at the beginning of the
        real-space step of a minibatch. If batching mode is "random/uniform", the gradient is always
        re-initialized when this method is called. If batching mode is "compact", gradient
        is only initialized if the current minibatch is the first in the current epoch.
        """
        if self.options.batching_mode in [enums.BatchingModes.RANDOM, enums.BatchingModes.UNIFORM]:
            self.parameter_group.object.initialize_grad()
        else:
            if self.current_minibatch == 0:
                self.parameter_group.object.initialize_grad()

    def _initialize_object_step_size_buffer(self):
        if self.current_minibatch == 0:
            self.reconstructor_buffers.alpha_object_all_pos_all_slices[...] = 1

    def _initialize_probe_step_size_buffer(self):
        if self.current_minibatch == 0:
            self.reconstructor_buffers.alpha_probe_all_pos[...] = 1

    def _initialize_momentum_buffers(self):
        """Initialize momentum buffers.

        Only the probe's accumulated update direction is initialized here. The accumulated update
        direction for the object is stored in `object.grad`.
        """
        if self.options.batching_mode != enums.BatchingModes.COMPACT or self.current_minibatch == 0:
            self.probe_momentum_params["accumulated_update_direction"] = 0

    @timer()
    def _update_momentum_buffers(self, delta_p_hat):
        """Update momentum buffer for probe after each minibatch using the update direction calculated
        in that minibatch.

        We do not track the update direction for the object here, because it is already recorded in
        `object.grad`.

        Parameters
        ----------
        delta_p_hat : Tensor
            A (n_opr_modes, n_probe_modes, h, w) tensor giving the update direction for the probe.
        """
        self.probe_momentum_params["accumulated_update_direction"] += delta_p_hat / len(
            self.dataloader
        )

    @timer()
    def _record_object_slice_gradient(self, i_slice, delta_o_hat, add_to_existing=False):
        """
        Record the gradient of one slice of a multislice object.
        """
        if not add_to_existing:
            self.parameter_group.object.set_grad(-delta_o_hat[0], slicer=i_slice)
        else:
            self.parameter_group.object.set_grad(
                self.parameter_group.object.get_grad()[i_slice] - delta_o_hat[0], slicer=i_slice
            )
            
    @timer()
    def _record_probe_gradient(self, delta_p_hat):
        """
        Record the gradient of the probe.
        
        Parameters
        ----------
        delta_p_hat : Tensor
            A (n_opr_modes, n_probe_modes, h, w) or (n_probe_modes, h, w) tensor 
            giving the update direction for the probe. If the input tensor is 3D,
            it is assumed to be the latter case and will be expanded with 0s for
            the OPR modes.
        """
        if delta_p_hat.ndim == 3:
            delta_p_hat_full = torch.zeros_like(self.parameter_group.probe.data)
            delta_p_hat_full[0] = delta_p_hat
        else:
            delta_p_hat_full = delta_p_hat
        self.parameter_group.probe.set_grad(-delta_p_hat_full)

    @timer()
    def _apply_object_update(self, alpha_o_mean_all_slices, delta_o_hat=None):
        """
        Apply object update using Eq. 27b of Odstrcil, 2018.

        If both `alpha_o_mean` and `delta_o_hat` are given, the object's gradient is set
        using averaged step size and `delta_o_hat`. Otherwise, we assume the gradient
        is already set previously, and just simply run the optimizer step.

        Parameters
        ----------
        alpha_o_mean_all_slices : Tensor
            A (n_slices,) tensor giving the averaged step size for each object slice.
        delta_o_hat : Tensor
            A (n_slices, h, w) tensor giving the update direction for the whole object. If None,
            use the (negative) update direction stored in `object.grad`.
        """
        if delta_o_hat is None:
            delta_o_hat = -self.parameter_group.object.get_grad()
        self.parameter_group.object.set_grad(-alpha_o_mean_all_slices[:, None, None] * delta_o_hat)
        self.parameter_group.object.optimizer.step()

    @timer()
    def _apply_object_momentum(self, alpha_o_mean_all_slices, delta_o_hat):
        """
        Apply momentum acceleration to the object. This is a special momentum acceleration used
        in PtychoShelves, which behaves somewhat differently from the momentum in `torch.optim.SGD`.
        """
        # Scale object update by step size at the beginning to match the behavior in PtychoShelves,
        # In PtychoShelves, this scaling happens in `update_object.m`.
        delta_o_hat = delta_o_hat * alpha_o_mean_all_slices[:, None, None]

        object_ = self.parameter_group.object
        if "update_direction_history" not in self.object_momentum_params.keys():
            self.object_momentum_params["update_direction_history"] = []
            self.object_momentum_params["velocity_map"] = torch.zeros_like(object_.data)

        object_roi_bbox = self.parameter_group.object.roi_bbox.get_bbox_with_top_left_origin()
        object_roi_slicer = object_roi_bbox.get_slicer()
        upd = delta_o_hat * alpha_o_mean_all_slices[:, None, None]
        upd = upd[:, *object_roi_slicer]
        upd = upd / pmath.mnorm(upd, dim=(-1, -2), keepdims=True)
        self.object_momentum_params["update_direction_history"].append(upd)

        momentum_memory = 2

        if len(self.object_momentum_params["update_direction_history"]) > momentum_memory + 1:
            # Remove the oldest momentum update.
            self.object_momentum_params["update_direction_history"].pop(0)
            for i_slice in range(object_.n_slices):
                if self._fourier_error_ok():
                    # Project older updates to the latest one, ordered from recent to old.
                    projected_updates = [
                        (
                            self.object_momentum_params["update_direction_history"][i][i_slice]
                            * self.object_momentum_params["update_direction_history"][-1][
                                i_slice
                            ].conj()
                        )
                        for i in range(
                            len(self.object_momentum_params["update_direction_history"]) - 1
                        )
                    ][::-1]
                    corr_level = [
                        torch.mean(projected_updates[k]).real for k in range(len(projected_updates))
                    ]
                    corr_level = torch.tensor(corr_level, device=delta_o_hat.device)

                if self._fourier_error_ok() and torch.all(corr_level > 0):
                    # Estimate optimal friction.
                    p = pmath.polyfit(
                        torch.arange(0.0, momentum_memory + 1.0, device=delta_o_hat.device),
                        torch.tensor([0, *torch.log(corr_level)], device=delta_o_hat.device),
                        deg=1,
                    )
                    # If correlation drops fast as one goes away from the current epoch, p[0]
                    # is more negative, and friction is larger. Accumulated velocity is weighted
                    # less to allow the update direction to change more quickly.
                    friction = 0.5 * (-p[0]).clip(0, None)

                    w = object_.preconditioner / (
                        0.1 * object_.preconditioner.max() + object_.preconditioner
                    )
                    m = self.options.momentum_acceleration_gradient_mixing_factor
                    m = friction if m is None else m
                    self.object_momentum_params["velocity_map"][i_slice] = (
                        1 - friction
                    ) * self.object_momentum_params["velocity_map"][i_slice] + m * delta_o_hat[
                        i_slice
                    ]
                    object_.set_data(
                        object_.data[i_slice]
                        + w
                        * self.options.momentum_acceleration_gain
                        * self.object_momentum_params["velocity_map"][i_slice],
                        slicer=i_slice,
                    )
                else:
                    self.object_momentum_params["velocity_map"][i_slice] = (
                        self.object_momentum_params["velocity_map"][i_slice] / 2.0
                    )

    @timer()
    def _fourier_error_ok(self):
        if len(self.reconstructor_buffers.fourier_errors) < 3:
            return True
        return max(self.reconstructor_buffers.fourier_errors[-3:-1]) > min(self.reconstructor_buffers.fourier_errors[-2:])

    @timer()
    def update_probe_positions(
        self, chi, indices, obj_patches, delta_o_patches, unique_probes, apply_updates=True
    ):
        """
        Update the probe positions.

        Parameters
        ----------
        apply_updates : bool
            If True, the data of the probe positions are modified with the
            update vectors. Otherwise, the update vectors will be saved in the
            ``grad`` attribute of the probe positions object.
        """
        delta_pos = self.parameter_group.probe_positions.position_correction.get_update(
            chi,
            obj_patches,
            delta_o_patches,
            unique_probes,
            self.parameter_group.object.optimizer_params["lr"],
        )
        delta_pos_full = torch.zeros_like(self.parameter_group.probe_positions.tensor)
        delta_pos_full[indices] = delta_pos
        self.parameter_group.probe_positions.set_grad(-delta_pos_full)
        if apply_updates:
            self.parameter_group.probe_positions.step_optimizer()

    @timer()
    def _calculate_final_object_update_step_size(self):
        """
        Given the patch-wise step sizes, calculate the final step size for the whole object used
        in compact-mode update.

        This routine follows the same logic as in PtychoSheleves. With the `(n_pos, n_slices)`
        tensor that sotres the step sizes for all object patches and all slices, we take the
        10-th percentile trimmed mean of the step sizes for each minibatch. We then take the
        minimum of the step sizes across all minibatches for each slice to use as the step size
        for updating the object.

        Returns
        -------
        Tensor
            A (n_slices,) tensor giving the final step size for each slice.
        """
        alpha_object_all_minibatches = []
        for inds in self.indices:
            alpha_current_batch = pmath.trim_mean(
                self.reconstructor_buffers.alpha_object_all_pos_all_slices[inds], 0.1, dim=0, keepdim=False
            )
            alpha_object_all_minibatches.append(alpha_current_batch)
        alpha_object_all_minibatches = torch.stack(alpha_object_all_minibatches, dim=0)
        alpha_object_all_slices = torch.min(
            alpha_object_all_minibatches, dim=0, keepdim=False
        ).values
        return alpha_object_all_slices

    @timer()
    def update_accumulated_intensities(self, y_true, y_pred):
        self.reconstructor_buffers.accumulated_true_intensity = self.reconstructor_buffers.accumulated_true_intensity + torch.sum(y_true)
        self.reconstructor_buffers.accumulated_pred_intensity = self.reconstructor_buffers.accumulated_pred_intensity + torch.sum(y_pred)

    @timer()
    def _apply_probe_intensity_scaling_correction(self):
        corr = math.sqrt(self.reconstructor_buffers.accumulated_true_intensity / self.reconstructor_buffers.accumulated_pred_intensity)
        self.parameter_group.probe.set_data(self.parameter_group.probe.data * corr)

    @timer()
    def update_fourier_error(self, y_pred, y_true):
        self.accumulated_fourier_error += torch.mean(
            (torch.sqrt(y_pred) - torch.sqrt(y_true)) ** 2, dim=(-2, -1)
        ).sum()
        if self.current_minibatch == len(self.dataloader) - 1:
            e = self.accumulated_fourier_error / self.parameter_group.probe_positions.shape[0]
            self.reconstructor_buffers.fourier_errors.append(e.item())
            self.accumulated_fourier_error = 0.0

    def run_pre_run_hooks(self) -> None:
        self.prepare_data()

    def run_pre_epoch_hooks(self) -> None:
        super().run_pre_epoch_hooks()
        self.accumulated_fourier_error = 0.0
        self.indices = []

    @timer()
    def run_post_epoch_hooks(self) -> None:
        if self.current_epoch > 0 or (not self.options.rescale_probe_intensity_in_first_epoch):
            if (
                self.parameter_group.object.optimization_enabled(self.current_epoch)
                and self.options.batching_mode == enums.BatchingModes.COMPACT
            ):
                # Take the 10-th percentile of the object step sizes across all minibatches for
                # each slice to use as the step size for updating the object.
                alpha_object_all_slices = self._calculate_final_object_update_step_size()
                delta_o_hat_full = self._precondition_accumulated_object_update_direction()
                self._apply_object_update(alpha_object_all_slices, delta_o_hat_full)

                if self.options.momentum_acceleration_gain > 0:
                    # Momentum acceleration for object is only applied for compact batching.
                    self._apply_object_momentum(alpha_object_all_slices, delta_o_hat_full)

            if (
                self.parameter_group.probe.optimization_enabled(self.current_epoch)
                and self.options.momentum_acceleration_gain > 0
            ):
                self._apply_probe_momentum(
                    torch.mean(self.reconstructor_buffers.alpha_probe_all_pos),
                    self.probe_momentum_params["accumulated_update_direction"],
                )
        else:
            # In epoch 0, only correct probe intensity.
            self._apply_probe_intensity_scaling_correction()
        return super().run_post_epoch_hooks()

    @timer()
    def run_minibatch(self, input_data, y_true, *args, **kwargs) -> None:
        indices = input_data[0]
        self.indices.append(indices)
        y_pred = self.forward_model(*input_data)
        self.update_fourier_error(y_pred, y_true)
        if self.current_epoch == 0 and self.options.rescale_probe_intensity_in_first_epoch:
            self.update_accumulated_intensities(y_true, y_pred)
        else:
            self.compute_reconstruction_parameter_updates(y_pred, y_true, indices)
            self.apply_reconstruction_parameter_updates(indices)

        self.loss_tracker.update_batch_loss_with_metric_function(y_pred, y_true)


class MultiprocessLSQMLReconstructor(LSQMLReconstructor, MultiprocessMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_sizes_all_ranks = None

    def run_minibatch(self, input_data, y_true, *args, **kwargs) -> None:
        indices = input_data[0]
        full_indices = indices.clone()
        indices, self.chunk_sizes_all_ranks = self.get_chunk_of_current_rank(
            full_indices, return_chunk_sizes=True
        )
        if len(indices) == 0:
            raise ValueError(
                "A rank didn't get any data in this minibatch. This happens when "
                "the total number of samples in the last minibatch is smaller "
                "than the number of ranks. Adjust your batch size or number of ranks "
                "to avoid this error."
            )
        self.indices.append(indices)
        input_data[0] = indices
        
        y_true = self.get_chunk_of_current_rank(y_true)
        
        y_pred = self.forward_model(*input_data)
        
        self.update_fourier_error(y_pred, y_true)
        if self.current_minibatch == len(self.dataloader) - 1:
            self.reconstructor_buffers.synchronize(["fourier_errors"])
        
        if self.current_epoch == 0 and self.options.rescale_probe_intensity_in_first_epoch:
            self.update_accumulated_intensities(y_true, y_pred)
            if self.current_minibatch == len(self.dataloader) - 1:
                self.reconstructor_buffers.synchronize(["accumulated_true_intensity", "accumulated_pred_intensity"])
        else:
            self.compute_reconstruction_parameter_updates(y_pred, y_true, indices)
            self.reconstructor_buffers.synchronize(
                ["alpha_object_all_pos_all_slices", "alpha_probe_all_pos"], indices=full_indices
            )
            # Sync gradients.
            self.parameter_group.synchronize_optimizable_parameter_gradients(
                op=dist.ReduceOp.AVG,
                names_to_exclude=["object"]
            )
            if self.options.batching_mode != enums.BatchingModes.COMPACT:
                # Object gradient should always be summed because it is not batch-averaged.
                self.parameter_group.synchronize_optimizable_parameter_gradients(
                    op=dist.ReduceOp.SUM,
                    names_to_include=["object"]
                )
            
            self.apply_reconstruction_parameter_updates(indices)

        self.loss_tracker.update_batch_loss_with_metric_function(y_pred, y_true)
        self.loss_tracker.synchronize_accumulated_losses()

    def _combine_object_patch_update_directions(
        self, delta_o_patches, positions, onto_accumulated=False, slice_index=0
    ):
        """
        Combine the update directions of object patches into a buffer with the
        same size as the whole object.
        
        This method overrides the parent method to synchronize buffer across ranks.

        Parameters
        ----------
        delta_o_patches : Tensor
            A (batch_size, 1, h, w) tensor giving the update direction for object patches.
        onto_accumulated : bool
            If True, add the update direction to the accumulated update direction stored in
            `object.grad`. Otherwise, just return the update direction accumulated on an empty
            buffer.

        Returns
        -------
        Tensor
            A (1, h, w) tensor giving the combined update direction for the whole object.
        """
        delta_o_patches = delta_o_patches[:, 0]

        # Stitch all delta O patches on the object buffer
        # Shape of delta_o_hat:  (h_whole, w_whole)
        delta_o_hat = self.parameter_group.object.place_patches_on_empty_buffer(
            positions.round().int(), delta_o_patches, integer_mode=True
        )
        
        # Synchronize buffer across ranks.
        delta_o_hat = self.sync_buffer(
            delta_o_hat,
            op=dist.ReduceOp.SUM,
        )
        
        delta_o_hat = delta_o_hat[None, ...]
        if onto_accumulated:
            delta_o_hat = delta_o_hat + (
                -self.parameter_group.object.get_grad()[slice_index : slice_index + 1]
            )
        return delta_o_hat
    
    
    def _precondition_probe_update_direction(self, delta_p):
        """
        Eq. 25a of Odstrcil, 2018.
        
        This method overrides the parent method to synchronize buffer across ranks.

        Parameters
        ----------
        delta_p : Tensor
            A (batch_size, n_probe_modes, h, w) tensor giving the probe update direction.

        Returns
        -------
        Tensor
            A (n_probe_modes, h, w) tensor giving the preconditioned probe update direction.
        """
        # Shape of delta_p_hat:  (n_probe_modes, h, w)
        delta_p_hat = torch.sum(delta_p, dim=0)  # Eq. 25a
        delta_p_hat = self.sync_buffer(
            delta_p_hat,
            op=dist.ReduceOp.SUM,
        )
        bsize_current_rank = delta_p.shape[0]
        bsize_all_ranks = self.sync_buffer(
            bsize_current_rank,
            op=dist.ReduceOp.SUM,
        )
        delta_p_hat = delta_p_hat / bsize_all_ranks
        return delta_p_hat

    def apply_reconstruction_parameter_updates(self, indices: torch.Tensor):
        """Perform updates for reconstruction parameters using the
        update vectors stored in the `grad` attribute of the corresponding
        `ReconstructionParameter` objects.

        This method overrides the parent method to synchronize buffer across ranks.

        Parameters
        ----------
        indices : torch.Tensor
            The indices of the diffraction patterns processed in the current batch.
        """
        tensor_list = [
            torch.zeros(
                self.chunk_sizes_all_ranks[r],
                dtype=indices.dtype,
                device=torch.get_default_device(),
            )
            for r in range(self.n_ranks)
        ]
        dist.all_gather(tensor_list, indices.to(torch.get_default_device()))
        indices_all_ranks = torch.cat(tensor_list)
        super().apply_reconstruction_parameter_updates(indices_all_ranks)

    def run_post_epoch_hooks(self) -> None:
        if self.current_epoch == 0 and self.options.rescale_probe_intensity_in_first_epoch:
            super().run_post_epoch_hooks()
            return
        
        # Sync accumulated object gradients if batching mode is compact.
        if self.options.batching_mode == enums.BatchingModes.COMPACT:
            self.parameter_group.synchronize_optimizable_parameter_gradients(
                op=dist.ReduceOp.SUM,
                names_to_include=["object"]
            )
        super().run_post_epoch_hooks()
        self.parameter_group.synchronize_optimizable_parameter_data(
            source_rank=0, 
        )
