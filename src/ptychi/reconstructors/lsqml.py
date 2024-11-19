from typing import Optional, TYPE_CHECKING
import logging

import torch
from torch.utils.data import Dataset

from ptychi import maps
import ptychi.data_structures.object
from ptychi.reconstructors.base import (
    AnalyticalIterativePtychographyReconstructor,
    LossTracker,
)
import ptychi.data_structures.base as ds
import ptychi.forward_models as fm
from ptychi.utils import chunked_processing
import ptychi.maths as pmath
import ptychi.api.enums as enums

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
        self.alpha_psi_far_all_points = None
        self.alpha_object_all_slices = torch.tensor([1.0 for _ in range(self.parameter_group.object.n_slices)])
        self.alpha_probe_all_minibatches = []
        

    def check_inputs(self, *args, **kwargs):
        if self.parameter_group.opr_mode_weights.optimizer is not None:
            logger.warning(
                "Selecting optimizer for OPRModeWeights is not supported for "
                "LSQMLReconstructor and will be disregarded."
            )
        if not isinstance(self.parameter_group.opr_mode_weights, ds.DummyParameter):
            if self.parameter_group.opr_mode_weights.data[:, 1:].abs().max() < 1e-5:
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
        self.alpha_psi_far_all_points = torch.full(
            size=(self.parameter_group.probe_positions.shape[0],), fill_value=0.5
        )

    def prepare_data(self, *args, **kwargs):
        self.parameter_group.probe.normalize_eigenmodes()
        logger.info("Probe eigenmodes normalized.")

    def get_psi_far_step_size(self, y_pred, y_true, indices, eps=1e-5):
        if isinstance(self.noise_model, fm.PtychographyGaussianNoiseModel):
            alpha = torch.tensor(0.5, device=y_pred.device)  # Eq. 16
        elif isinstance(self.noise_model, fm.PtychographyPoissonNoiseModel):
            # This implementation reproduces PtychoShelves (gradient_descent_xi_solver)
            # and is different from Eq. 17 of Odstrcil (2018).
            xi = 1 - y_true / (y_pred + eps)
            for _ in range(2):
                alpha_prev = self.alpha_psi_far_all_points[indices].mean()
                alpha = (xi * (y_pred - y_true / (1 - alpha_prev * xi))).sum(-1).sum(-1)
                alpha = alpha / (xi**2 * y_pred).sum(-1).sum(-1)
                # Use previous step size as momentum.
                alpha = 0.5 * alpha_prev + 0.5 * alpha
                alpha = alpha.clamp(0, 1)
                self.alpha_psi_far_all_points[indices] = alpha
            # Add perturbation.
            alpha = alpha + torch.randn(alpha.shape, device=alpha.device) * 1e-2
            self.alpha_psi_far_all_points[indices] = alpha
            logger.debug("poisson alpha_psi_far: mean = {}".format(torch.mean(alpha)))
        return alpha

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

        psi_opt = self.forward_model.far_field_propagator.propagate_backward(psi_far)
        return psi_opt

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

        if self.parameter_group.object.is_multislice:
            delta_o_patches = self.update_object_and_probe_multislice(
                indices, chi, obj_patches, positions
            )
        else:
            delta_o_patches = self.update_object_and_probe(indices, chi, obj_patches, positions)
        if self.parameter_group.probe_positions.optimization_enabled(self.current_epoch):
            self.update_probe_positions(chi, indices, obj_patches, delta_o_patches)

    def update_object_and_probe(self, indices, chi, obj_patches, positions, gamma=1e-5):
        # TODO: avoid unnecessary computations when not both of object and probe are optimizable
        self._initialize_object_gradient()
        self._initialize_object_step_size_buffer()
        self.alpha_probe_all_minibatches = []
        
        delta_p_i = self._calculate_probe_update_direction(
            chi, obj_patches, slice_index=0
        )  # Eq. 24a
        delta_p_hat = self._precondition_probe_update_direction(delta_p_i)  # Eq. 25a

        delta_o_i = self._calculate_object_patch_update_direction(indices, chi)
        delta_o_comb = self._combine_object_patch_update_directions(delta_o_i, positions)
        delta_o_precond, delta_o_i = self._precondition_object_update_direction(
            delta_o_comb, positions
        )

        alpha_o_i, alpha_p_i = self.calculate_object_and_probe_update_step_sizes(
            indices, chi, obj_patches, delta_o_i, delta_p_i, gamma=gamma
        )
        self.alpha_object_all_slices[0] += pmath.trim_mean(alpha_o_i)

        if self.parameter_group.probe.optimization_enabled(self.current_epoch):
            self._apply_probe_update(alpha_p_i, delta_p_hat)

        if (
            self.parameter_group.object.optimization_enabled(self.current_epoch)
            and self.options.batching_mode == enums.BatchingModes.RANDOM
        ):
            self._apply_object_update(self.alpha_object_all_slices, delta_o_precond)
        else:
            self._record_object_slice_gradient(
                0, delta_o_comb, add_to_existing=True
            )

        self.update_variable_probe(indices, chi, delta_p_i, delta_p_hat, obj_patches)

        delta_o_patches = self.alpha_object_all_slices[0] * delta_o_i
        return delta_o_patches

    def update_object_and_probe_multislice(self, indices, chi, obj_patches, positions, gamma=1e-5):
        """
        Update the object and probe for multislice reconstruction.

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
        self._initialize_object_step_size_buffer()
        self.alpha_probe_all_minibatches = []

        for i_slice in range(object_.n_slices - 1, -1, -1):
            if i_slice < object_.n_slices - 1:
                chi = self.forward_model.propagate_to_previous_slice(chi, slice_index=i_slice + 1)

            psi_im1 = self._get_psi_im1(i_slice, indices)
            delta_o_i = self._calculate_object_patch_update_direction(indices, chi, psi_im1=psi_im1)
            delta_o_comb = self._combine_object_patch_update_directions(delta_o_i, positions)
            delta_o_precond, delta_o_i = self._precondition_object_update_direction(
                delta_o_comb, positions
            )
            alpha_o_i = self.calculate_object_update_step_sizes(
                indices, chi, delta_o_i, gamma=gamma
            )
            # In compact batching mode, object is updated at the end of an epoch using gradients
            # accumulated over all minibatches.
            if self.options.batching_mode == enums.BatchingModes.RANDOM:
                self._record_object_slice_gradient(
                    i_slice, delta_o_precond, add_to_existing=False
                )
            else:
                self._record_object_slice_gradient(
                    i_slice, delta_o_comb, add_to_existing=True
                )

            delta_p_i = self._calculate_probe_update_direction(
                chi, obj_patches=obj_patches, slice_index=i_slice
            )  # Eq. 24a

            if i_slice == 0:
                delta_p_hat = self._precondition_probe_update_direction(delta_p_i)  # Eq. 25a

                if self.options.solve_obj_prb_step_size_jointly_for_first_slice_in_multislice:
                    (alpha_o_i, alpha_p_i) = self.calculate_object_and_probe_update_step_sizes(
                        indices, chi, obj_patches, delta_o_i, delta_p_i, gamma=gamma, slice_index=0
                    )
                    if self.options.batching_mode == enums.BatchingModes.RANDOM:
                        self._record_object_slice_gradient(0, delta_o_precond)
                else:
                    alpha_p_i = self.calculate_probe_update_step_sizes(
                        chi, obj_patches, delta_p_i, gamma=gamma
                    )

                if self.parameter_group.probe.optimization_enabled(self.current_epoch):
                    self._apply_probe_update(alpha_p_i, delta_p_hat)
                    
            self.alpha_object_all_slices[i_slice] += pmath.trim_mean(alpha_o_i)
            chi = delta_p_i

        if (
            self.parameter_group.object.optimization_enabled(self.current_epoch)
            and self.options.batching_mode == enums.BatchingModes.RANDOM
        ):
            self._apply_object_update(self.alpha_object_all_slices, None)

        self.update_variable_probe(indices, chi, delta_p_i, delta_p_hat, obj_patches)

        delta_o_patches = self.alpha_object_all_slices[0] * delta_o_i
        return delta_o_patches

    def _get_psi_im1(self, i_slice, indices):
        r"""
        Get $\psi_{i-1}$, the wavefield modulated by the previous slice and then propagated,
        for multislice reconstruction. If `i_slice == 0`, return the probe instead.

        Parameters
        ----------
        i_slice : int
            The current slice index.
        indices : Tensor
            A tensor of indices of diffraction patterns processed in the current batch.

        Returns
        -------
        Tensor
            The previous wavefield. If `i_slice == 0` and the probe does not have OPR modes,
            the returned tensor will be (n_modes, h, w); otherwise it will be
            (batch_size, n_modes, h, w).
        """
        if i_slice > 0:
            psi_im1 = self.forward_model.intermediate_variables["slice_psis"][:, i_slice - 1]
        else:
            if self.parameter_group.probe.has_multiple_opr_modes:
                psi_im1 = self.parameter_group.probe.get_unique_probes(
                    weights=self.parameter_group.opr_mode_weights.get_weights(indices),
                    mode_to_apply=0,
                )
            else:
                psi_im1 = self.parameter_group.probe.get_opr_mode(0)
        return psi_im1

    def calculate_object_and_probe_update_step_sizes(
        self, indices, chi, obj_patches, delta_o_i, delta_p_i, gamma=1e-5, slice_index=0
    ):
        """
        Jointly calculate the update step sizes for object and probe according to Eq. 22 of Odstrcil (2018).
        This routine builds a (batch_size, 2, 2) batch matrix, batch-invert them to get the update step sizes.
        """
        obj_patches = obj_patches[:, slice_index]
        delta_o_i = delta_o_i[:, 0]

        if self.parameter_group.probe.has_multiple_opr_modes:
            # Shape of probe:         (n_batch, n_modes, h, w)
            probe = self.parameter_group.probe.get_unique_probes(
                self.parameter_group.opr_mode_weights.get_weights(indices), mode_to_apply=0
            )
        else:
            # Shape of probe:         (1, n_modes, h, w)
            probe = self.parameter_group.probe.get_opr_mode(0, keepdim=True)

        if self.options.solve_step_sizes_only_using_first_probe_mode:
            probe = probe[:, 0:1]
            delta_p_i = delta_p_i[:, 0:1]

        # Shape of delta_p_o/o_p:     (batch_size, n_probe_modes or 1, h, w)
        delta_p_o = delta_p_i * obj_patches[:, None, :, :]
        delta_o_patches_p = delta_o_i[:, None, :, :] * probe

        # Shape of aij:               (batch_size,)
        a11 = (delta_o_patches_p.abs() ** 2).sum(-1).sum(-1).sum(-1) 
        a11 = a11 + 0.1 * torch.mean(a11, dim=0)
        a12 = (delta_o_patches_p * delta_p_o.conj()).sum(-1).sum(-1).sum(-1)
        a21 = a12.conj()
        a22 = (delta_p_o.abs() ** 2).sum(-1).sum(-1).sum(-1)
        a22 = a22 + 0.1 * torch.mean(a22, dim=0)
        b1 = torch.real(delta_o_patches_p.conj() * chi).sum(-1).sum(-1).sum(-1)
        b2 = torch.real(delta_p_o.conj() * chi).sum(-1).sum(-1).sum(-1)

        a_mat = torch.stack([a11, a12, a21, a22], dim=1).view(-1, 2, 2)
        b_vec = torch.stack([b1, b2], dim=1).view(-1, 2).type(a_mat.dtype)
        alpha_vec = torch.linalg.solve(a_mat, b_vec)
        alpha_vec = alpha_vec.real.clip(0, None)

        alpha_o_i = alpha_vec[:, 0]
        alpha_p_i = alpha_vec[:, 1]

        alpha_o_i = self.options.beta_LSQ * alpha_o_i / ( self.parameter_group.object.n_slices * self.parameter_group.probe.n_modes )
        alpha_p_i = self.options.beta_LSQ * alpha_p_i

        logger.debug("alpha_p_i: min={}, max={}".format(alpha_p_i.min(), alpha_p_i.max()))
        logger.debug("alpha_o_i: min={}, max={}".format(alpha_o_i.min(), alpha_o_i.max()))

        return alpha_o_i, alpha_p_i

    def calculate_object_update_step_sizes(self, indices, chi, delta_o_i, gamma=1e-5):
        """
        Calculate the update step sizes just for the object using Eq. 23b of Odstrcil (2018).
        """
        # Just take the first slice.
        delta_o_i = delta_o_i[:, 0]

        if self.parameter_group.probe.has_multiple_opr_modes:
            # Shape of probe:         (n_batch, n_modes, h, w)
            probe = self.parameter_group.probe.get_unique_probes(
                self.parameter_group.opr_mode_weights.get_weights(indices), mode_to_apply=0
            )
        else:
            # Shape of probe:         (1, n_modes, h, w)
            probe = self.parameter_group.probe.get_opr_mode(0, keepdim=True)

        if self.options.solve_step_sizes_only_using_first_probe_mode:
            probe = probe[:, 0:1]

        # Shape of delta_p_o/o_p:     (batch_size, n_probe_modes or 1, h, w)
        delta_o_patches_p = delta_o_i[:, None, :, :] * probe

        numerator = 0.5 * torch.sum(torch.real(delta_o_patches_p.conj() * chi), dim=(-1, -2, -3))
        denominator = torch.sum(delta_o_patches_p.abs() ** 2, dim=(-1, -2, -3))

        alpha_o_i = numerator / (denominator + gamma)

        alpha_o_i = self.options.beta_LSQ * alpha_o_i / ( self.parameter_group.object.n_slices * self.parameter_group.probe.n_modes )

        alpha_o_i = alpha_o_i.clamp(
            0, self.parameter_group.object.options.solved_step_size_upper_bound
        )

        logger.debug(
            "alpha_o_i: min={}, max={}, trim_mean={}".format(
                alpha_o_i.min(), alpha_o_i.max(), pmath.trim_mean(alpha_o_i)
            )
        )
        return alpha_o_i

    def calculate_probe_update_step_sizes(self, chi, obj_patches, delta_p_i, gamma=1e-5):
        """
        Calculate the update step sizes just for the probe using Eq. 23a of Odstrcil (2018).
        """
        # Just take the first slice.
        obj_patches = obj_patches[:, 0]

        if self.options.solve_step_sizes_only_using_first_probe_mode:
            delta_p_i = delta_p_i[:, 0:1]

        # Shape of delta_p_o/o_p:     (batch_size, n_probe_modes, h, w)
        delta_p_o = delta_p_i * obj_patches[:, None, :, :]

        # Shape of aij:               (batch_size,)
        numerator = 0.5 * torch.sum(torch.real(delta_p_o.conj() * chi), dim=(-1, -2, -3))
        denominator = torch.sum(delta_p_o.abs() ** 2, dim=(-1, -2, -3))

        alpha_p_i = numerator / (denominator + gamma)
        alpha_p_i = self.options.beta_LSQ * alpha_p_i
        
        alpha_p_i = alpha_p_i.clamp(
            0, self.parameter_group.object.options.solved_step_size_upper_bound
        )

        logger.debug("alpha_p_i: min={}, max={}".format(alpha_p_i.min(), alpha_p_i.max()))
        return alpha_p_i

    def _calculate_probe_update_direction(self, chi, obj_patches=None, slice_index=0):
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
        if obj_patches is not None:
            obj_patches = obj_patches[:, slice_index]
            delta_p = chi * obj_patches.conj()[:, None, :, :]  # Eq. 24a
        else:
            delta_p = chi
        return delta_p

    def _precondition_probe_update_direction(self, delta_p):
        """
        Eq. 25a of Odstrcil, 2018.
        """
        # Shape of delta_p_hat:  (n_probe_modes, h, w)
        delta_p_hat = torch.sum(delta_p, dim=0)  # Eq. 25a
        # PtychoShelves code simply takes the average. This is different from the paper
        # which does delta_p_hat = delta_p_hat / ((object_.abs() ** 2).sum() + delta),
        # but this seems to work better.
        delta_p_hat = delta_p_hat / delta_p.shape[0]
        return delta_p_hat

    def _apply_probe_update(self, alpha_p_i, delta_p_hat):
        """
        Eq. 27a of Odstrcil, 2018.
        """
        # Shape of alpha_p_i:        (batch_size,)
        # Shape of delta_p_hat:      (n_probe_modes, h, w)
        # PtychoShelves code simply multiplies delta_p_hat with averaged step size.
        # This is different from the paper which does the following:
        #     update_vec = delta_p_hat * obj_patches[:, None, :, :].abs() ** 2
        #     update_vec = update_vec * alpha_p_i[:, None, None, None]
        #     update_vec = update_vec / ((obj_patches.abs() ** 2).sum(0) + delta)

        # Just apply the update to the main OPR mode of each incoherent mode.
        # To do this, we pad the update vector with zeros in the OPR mode dimension.
        delta_p_hat = delta_p_hat[None, :, :, :]
        if self.parameter_group.probe.has_multiple_opr_modes:
            delta_p_hat = torch.nn.functional.pad(
                delta_p_hat,
                pad=(0, 0, 0, 0, 0, 0, 0, self.parameter_group.probe.n_opr_modes - 1),
                mode="constant",
                value=0.0,
            )

        if self.options.batching_mode == enums.BatchingModes.COMPACT:
            # In compact mode, object is updated only once per epoch. To match the probe to this,
            # we divide the probe step size by the number of minibatches before each probe update.
            alpha_p_i = alpha_p_i / len(self.dataloader)
        alpha_p_mean = torch.mean(alpha_p_i)
        self.alpha_probe_all_minibatches.append(alpha_p_mean)
        self.parameter_group.probe.set_grad(-delta_p_hat * alpha_p_mean)
        self.parameter_group.probe.optimizer.step()

    def _calculate_object_patch_update_direction(self, indices, chi, psi_im1=None):
        r"""
        Calculate the update direction for object patches, implementing
        Eq. 24b of Odstrcil, 2018. This function works in both 2D mode and
        multislice mode:

        - When `psi_im1` is None, 2D mode is assumed. `chi` is multiplied with the
            complex conjugate of the probe.
        - When `psi_im1` is not None, multislice mode is assumed. `chi` is multiplied
            with the complex conjugate of `psi_im1`.

        Parameters
        ----------
        indices : Tensor
            Indices of diffraction patterns in the current batch.
        chi : Tensor
            A (batch_size, n_modes, h, w) tensor giving the difference of exit waves.
            For multislice, this should be the exiting-plane `chi` backpropagated to the
            current slice.
        psi_im1 : Tensor
            A (batch_size, n_modes, h, w) tensor giving $\psi_{i - 1}$, the wavefield
            modulated by the previous slice and propagated to the current slice. If this
            is given, multislice mode is assumed.

        Returns
        -------
        Tensor
            A (batch_size, 1, h, w) tensor giving the update direction for object patches.
            The dimension of size 1 is to match the slice dimension in the object patch
            tensor.
        """
        if psi_im1 is None:
            if self.parameter_group.probe.has_multiple_opr_modes:
                # Shape of p:    (batch_size, n_probe_modes, h, w)
                p = self.parameter_group.probe.get_unique_probes(
                    weights=self.parameter_group.opr_mode_weights.get_weights(indices),
                    mode_to_apply=0,
                )
            else:
                # Shape of p:    (n_probe_modes, h, w)
                p = self.parameter_group.probe.get_opr_mode(0)
        else:
            p = psi_im1
        # Shape of chi:          (batch_size, n_probe_modes, h, w)
        # Shape delta_o_patches: (batch_size, h, w)
        # Multiply and sum over probe mode dimension
        delta_o_patches = torch.sum(chi * p.conj(), dim=1)  # Eq. 24b
        return delta_o_patches[:, None, :, :]

    def _combine_object_patch_update_directions(self, delta_o_patches, positions):
        """
        Combine the update directions of object patches into a buffer with the
        same size as the whole object.

        Parameters
        ----------
        delta_o_patches : Tensor
            A (batch_size, 1, h, w) tensor giving the update direction for object patches.

        Returns
        -------
        Tensor
            A (1, h, w) tensor giving the combined update direction for the whole object.
        """
        delta_o_patches = delta_o_patches[:, 0]

        # Stitch all delta O patches on the object buffer
        # Shape of delta_o_hat:  (h_whole, w_whole)
        delta_o_hat = self.parameter_group.object.place_patches_on_empty_buffer(
            positions, delta_o_patches
        )
        return delta_o_hat[None, ...]

    def _precondition_object_update_direction(self, delta_o_hat, positions=None, alpha_mix=0.1, slice_index=0):
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
            delta_o_patches = self.parameter_group.object.extract_patches_function(
                delta_o_hat,
                positions + self.parameter_group.object.center_pixel,
                self.parameter_group.probe.shape[-2:],
            )
    
            return delta_o_hat[None, ...], delta_o_patches[:, None, :, :]
        return delta_o_hat[None, ...]
    
    def _initialize_object_gradient(self):
        """
        Initialize object gradient with zeros. This method is called at the beginning of the
        real-space step of a minibatch. If batching mode is "random", the gradient is always
        re-initialized when this method is called. If batching mode is "compact", gradient
        is only initialized if the current minibatch is the first in the current epoch.
        """
        if self.options.batching_mode == enums.BatchingModes.RANDOM:
            self.parameter_group.object.initialize_grad()
        else:
            if self.current_minibatch == 0:
                self.parameter_group.object.initialize_grad()
                
    def _initialize_object_step_size_buffer(self):
        """
        Initialize object step size with zeros. This method is called at the beginning of the
        real-space step of a minibatch. If batching mode is "random", the buffer is always
        re-initialized when this method is called. If batching mode is "compact", the buffer
        is only initialized if the current minibatch is the first in the current epoch so
        that it can be accumulated across minibatches.
        """
        if self.options.batching_mode == enums.BatchingModes.RANDOM:
            self.alpha_object_all_slices[...] = 0
        else:
            if self.current_minibatch == 0:
                self.alpha_object_all_slices[...] = 0

    def _record_object_slice_gradient(
        self, i_slice, delta_o_hat, add_to_existing=False
    ):
        """
        Record the gradient of one slice of a multislice object.
        """
        if not add_to_existing:
            self.parameter_group.object.set_grad(-delta_o_hat[0], slicer=i_slice)
        else:
            self.parameter_group.object.set_grad(
                self.parameter_group.object.get_grad()[i_slice] - delta_o_hat[0], slicer=i_slice
            )

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
            then the (negative) update direction should be stored in `object.grad`.
        """
        if delta_o_hat is None:
            delta_o_hat = -self.parameter_group.object.get_grad()
        self.parameter_group.object.set_grad(-alpha_o_mean_all_slices[:, None, None] * delta_o_hat)
        self.parameter_group.object.optimizer.step()

    def update_probe_positions(self, chi, indices, obj_patches, delta_o_patches):
        delta_pos = self.parameter_group.probe_positions.position_correction.get_update(
            chi,
            obj_patches,
            delta_o_patches,
            self.parameter_group.probe,
            self.parameter_group.opr_mode_weights,
            indices,
            self.parameter_group.object.optimizer_params["lr"],
        )
        self._apply_probe_position_update(delta_pos, indices)

    def _apply_probe_position_update(self, delta_pos, indices):
        # TODO: allow setting step size or use adaptive step size
        if self.parameter_group.probe_positions.update_magnitude_limit > 0:
            lim = self.parameter_group.probe_positions.update_magnitude_limit
            delta_pos = torch.clamp(delta_pos, -lim, lim)

        delta_pos_full = torch.zeros_like(self.parameter_group.probe_positions.tensor)
        delta_pos_full[indices] = delta_pos
        self.parameter_group.probe_positions.set_grad(-delta_pos_full)
        self.parameter_group.probe_positions.optimizer.step()

    def _calculate_fourier_probe_position_update_direction(self, chi, positions, obj_patches):
        """
        Eq. 28 of Odstrcil (2018).
        """
        raise NotImplementedError
        probe = self.parameter_group.probe.tensor.complex()
        f_probe = torch.fft.fft2(probe)

        # coord_ramp = torch.fft.fftfreq(probe.shape[-2])
        # dp = 2j * torch.pi * coord_ramp[:, None] * obj_patches[:, None, :, :] * probe[None, :, :, :]
        # nom_y = (dp.conj() * chi).real()
        # denom_y = dp.abs() ** 2

        # coord_ramp = torch.fft.fftfreq(probe.shape[-1])
        # dp = 2j * torch.pi * coord_ramp[None, :] * obj_patches[:, None, :, :] * probe[None, :, :, :]
        # nom_x = (dp.conj() * chi).real()
        # denom_x = dp.abs() ** 2

        coord_ramp = torch.fft.fftfreq(probe.shape[-2])
        delta_p_y = torch.ifft2(2 * torch.pi * coord_ramp[:, None] * 1j * f_probe)

        coord_ramp = torch.fft.fftfreq(probe.shape[-1])
        delta_p_x = torch.ifft2(2 * torch.pi * coord_ramp[None, :] * 1j * f_probe)

    def update_variable_probe(self, indices, chi, delta_p_i, delta_p_hat, obj_patches):
        if self.parameter_group.probe.has_multiple_opr_modes and (
            self.parameter_group.probe.optimization_enabled(self.current_epoch)
            or (
                not self.parameter_group.opr_mode_weights.is_dummy
                and self.parameter_group.opr_mode_weights.eigenmode_weight_optimization_enabled(
                    self.current_epoch
                )
            )
        ):
            self.update_opr_probe_modes_and_weights(
                indices, chi, delta_p_i, delta_p_hat, obj_patches
            )

        if (
            not self.parameter_group.opr_mode_weights.is_dummy
            and self.parameter_group.opr_mode_weights.intensity_variation_optimization_enabled(
                self.current_epoch
            )
        ):
            delta_weights_int = self._calculate_intensity_variation_update_direction(
                indices, chi, obj_patches
            )
            self._apply_variable_intensity_updates(delta_weights_int)

    def update_opr_probe_modes_and_weights(self, indices, chi, delta_p_i, delta_p_hat, obj_patches):
        """
        Update the eigenmodes of the first incoherent mode of the probe, and update the OPR mode weights.

        This implementation is adapted from PtychoShelves code (update_variable_probe.m) and has some
        differences from Eq. 31 of Odstrcil (2018).
        """
        probe = self.parameter_group.probe.data
        weights = self.parameter_group.opr_mode_weights.data

        batch_size = len(delta_p_i)
        n_points_total = self.parameter_group.probe_positions.shape[0]
        # FIXME: reduced relax_u/v by a factor of 10 for stability, but PtychoShelves works without this.
        relax_u = (
            min(0.1, batch_size / n_points_total)
            * self.parameter_group.probe.options.eigenmode_update_relaxation
        )
        relax_v = self.parameter_group.opr_mode_weights.options.update_relaxation
        # Shape of delta_p_i:       (batch_size, n_probe_modes, h, w)
        # Shape of delta_p_hat:     (n_probe_modes, h, w)
        # Just use the first incoherent mode.
        delta_p_i = delta_p_i[:, 0, :, :]
        delta_p_hat = delta_p_hat[0, :, :]
        residue_update = delta_p_i - delta_p_hat

        # Start from the second OPR mode which is the first after the main mode - i.e., the first eigenmode.
        for i_opr_mode in range(1, self.parameter_group.probe.n_opr_modes):
            # Just take the first incoherent mode.
            eigenmode_i = self.parameter_group.probe.get_mode_and_opr_mode(
                mode=0, opr_mode=i_opr_mode
            )
            weights_i = self.parameter_group.opr_mode_weights.get_weights(indices)[:, i_opr_mode]
            eigenmode_i, weights_i = self._update_first_eigenmode_and_weight(
                residue_update,
                eigenmode_i,
                weights_i,
                relax_u,
                relax_v,
                obj_patches,
                chi,
                update_eigenmode=self.parameter_group.probe.optimization_enabled(
                    self.current_epoch
                ),
                update_weights=self.parameter_group.opr_mode_weights.eigenmode_weight_optimization_enabled(
                    self.current_epoch
                ),
            )

            # Project residue on this eigenmode, then subtract it.
            residue_update = (
                residue_update - pmath.project(residue_update, eigenmode_i) * eigenmode_i
            )

            probe[i_opr_mode, 0, :, :] = eigenmode_i
            weights[indices, i_opr_mode] = weights_i

        if self.parameter_group.probe.optimization_enabled(self.current_epoch):
            self.parameter_group.probe.set_data(probe)
        if self.parameter_group.opr_mode_weights.eigenmode_weight_optimization_enabled(
            self.current_epoch
        ):
            self.parameter_group.opr_mode_weights.set_data(weights)

    def _update_first_eigenmode_and_weight(
        self,
        residue_update,
        eigenmode_i,
        weights_i,
        relax_u,
        relax_v,
        obj_patches,
        chi,
        eps=1e-5,
        update_eigenmode=True,
        update_weights=True,
    ):
        # Shape of residue_update:          (batch_size, h, w)
        # Shape of eigenmode_i:             (h, w)
        # Shape of weights_i:               (batch_size,)

        obj_patches = obj_patches[:, 0]

        # Update eigenmode.
        # Shape of proj:                    (batch_size, h, w)
        # FIXME: What happens when weights is zero!?
        proj = ((residue_update.conj() * eigenmode_i).real + weights_i[:, None, None]) / pmath.norm(
            weights_i
        ) ** 2

        if update_eigenmode:
            # Shape of eigenmode_update:        (h, w)
            eigenmode_update = torch.mean(
                residue_update * torch.mean(proj, dim=(-2, -1), keepdim=True), dim=0
            )
            eigenmode_i = eigenmode_i + relax_u * eigenmode_update / (
                pmath.mnorm(eigenmode_update.view(-1)) + eps
            )
            eigenmode_i = eigenmode_i / pmath.mnorm(eigenmode_i.view(-1) + eps)

        if update_weights:
            # Update weights using Eq. 23a.
            # Shape of psi:                     (batch_size, h, w)
            psi = eigenmode_i * obj_patches
            # The denominator can get smaller and smaller as eigenmode_i goes down.
            # Weight update needs to be clamped.
            denom = torch.mean((torch.abs(psi) ** 2), dim=(-2, -1))
            num = torch.mean((chi[:, 0, :, :] * psi.conj()).real, dim=(-2, -1))
            weight_update = num / (denom + 0.1 * torch.mean(denom))
            weight_update = weight_update.clamp(max=10)
            weights_i = weights_i + relax_v * weight_update

        return eigenmode_i, weights_i

    def _calculate_intensity_variation_update_direction(self, indcies, chi, obj_patches):
        """
        Update variable intensity scaler - i.e., the OPR mode weight corresponding to the main mode.

        This implementation is adapted from PtychoShelves code (update_variable_probe.m) and has some
        differences from Eq. 31 of Odstrcil (2018).
        """
        # Just take the first slice.
        obj_patches = obj_patches[:, 0]

        mean_probe = self.parameter_group.probe.get_mode_and_opr_mode(mode=0, opr_mode=0)
        op = obj_patches * mean_probe
        num = torch.real(op.conj() * chi[:, 0, ...])
        denom = op.abs() ** 2
        delta_weights_int_i = torch.sum(num, dim=(-2, -1)) / torch.sum(denom, dim=(-2, -1))
        # Pad it to the same shape as opr_mode_weights.
        delta_weights_int = torch.zeros_like(self.parameter_group.opr_mode_weights.data)
        delta_weights_int[indcies, 0] = delta_weights_int_i
        return delta_weights_int

    def _apply_variable_intensity_updates(self, delta_weights_int):
        weights = self.parameter_group.opr_mode_weights
        weights.set_data(weights.data + 0.1 * delta_weights_int)

    def run_pre_run_hooks(self) -> None:
        self.prepare_data()

    def run_pre_epoch_hooks(self) -> None:
        self.update_preconditioners()

    def run_post_epoch_hooks(self) -> None:
        if (
            self.parameter_group.object.optimization_enabled(self.current_epoch)
            and self.options.batching_mode == enums.BatchingModes.COMPACT
        ):
            # Average object step sizes.
            self.alpha_object_all_slices = self.alpha_object_all_slices / self.current_minibatch
            delta_o_hat_full = []
            for i_slice in range(self.parameter_group.object.n_slices):
                delta_o_hat = self._precondition_object_update_direction(
                    -self.parameter_group.object.get_grad()[i_slice:i_slice + 1], 
                    positions=None
                )
                delta_o_hat_full.append(delta_o_hat)
            delta_o_hat_full = torch.cat(delta_o_hat_full, dim=0)
            self._apply_object_update(self.alpha_object_all_slices[:, None, None], delta_o_hat_full)
        return super().run_post_epoch_hooks()

    def run_minibatch(self, input_data, y_true, *args, **kwargs) -> None:
        indices = input_data[0]
        y_pred = self.forward_model(*input_data)

        psi_opt = self.run_reciprocal_space_step(y_pred, y_true, indices)
        self.run_real_space_step(psi_opt, indices)

        self.loss_tracker.update_batch_loss_with_metric_function(y_pred, y_true)

    def get_config_dict(self) -> dict:
        d = super().get_config_dict()
        d.update({"noise_model": self.noise_model.noise_statistics})
        return d
