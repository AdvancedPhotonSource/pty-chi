# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Union, TYPE_CHECKING, Optional
import logging

import torch
from torch import Tensor

import ptychi.data_structures.base as dsbase
import ptychi.image_proc as ip
import ptychi.maths as pmath
from ptychi.timing.timer_utils import timer

if TYPE_CHECKING:
    import ptychi.api as api
    from ptychi.data_structures.probe import Probe

logger = logging.getLogger(__name__)


class OPRModeWeights(dsbase.ReconstructParameter):
    # TODO: update_relaxation is only used for LSQML. We should create dataclasses
    # to contain additional options for ReconstructParameter classes, and subclass them for specific
    # reconstruction algorithms - for example, OPRModeWeightsOptions -> LSQMLOPRModeWeightsOptions.
    options: "api.options.base.OPRModeWeightsOptions"

    def __init__(
        self,
        *args,
        name="opr_weights",
        options: "api.options.base.OPRModeWeightsOptions" = None,
        **kwargs,
    ):
        """
        Weights of OPR modes for each scan point.

        Parameters
        ----------
        name : str
            Name of the parameter.
        update_relaxation : float
            Relaxation factor, or effectively the step size, for the update step in LSQML.
        """
        super().__init__(*args, name=name, options=options, is_complex=False, **kwargs)
        if len(self.shape) != 2:
            raise ValueError("OPR weights must be of shape (n_scan_points, n_opr_modes).")

        if self.optimizable:
            if not (options.optimize_eigenmode_weights or options.optimize_intensity_variation):
                raise ValueError(
                    "When OPRModeWeights is optimizable, at least 1 of "
                    "optimize_eigenmode_weights and optimize_intensity_variation "
                    "should be set to True."
                )

        self.optimize_eigenmode_weights = options.optimize_eigenmode_weights
        self.optimize_intensity_variation = options.optimize_intensity_variation

    @property
    def n_opr_modes(self):
        return self.tensor.shape[1]

    @property
    def n_scan_points(self):
        return self.tensor.shape[0]

    @property
    def n_eigenmodes(self):
        return self.tensor.shape[1] - 1

    def build_optimizer(self):
        if self.optimizer_class is None:
            return
        if self.optimizable:
            if isinstance(self.tensor, dsbase.ComplexTensor):
                self.optimizer = self.optimizer_class([self.tensor.data], **self.optimizer_params)
            else:
                self.optimizer = self.optimizer_class([self.tensor], **self.optimizer_params)

    def get_weights(self, indices: Union[tuple[int, ...], slice]) -> Tensor:
        return self.data[indices]

    def optimization_enabled(self, epoch: int):
        enabled = super().optimization_enabled(epoch)
        return enabled and (self.optimize_eigenmode_weights or self.optimize_intensity_variation)

    def eigenmode_weight_optimization_enabled(self, epoch: int):
        enabled = super().optimization_enabled(epoch)
        return enabled and self.optimize_eigenmode_weights

    def intensity_variation_optimization_enabled(self, epoch: int):
        enabled = super().optimization_enabled(epoch)
        return enabled and self.optimize_intensity_variation

    @timer()
    def update_variable_probe(
        self,
        probe: "Probe",
        adjoint_shift_probe_update_direction,   # what do I do for type hint here?
        indices: Tensor,
        chi: Tensor,
        delta_p_i: Tensor,
        delta_p_hat: Tensor,
        obj_patches: Tensor,
        current_epoch: int,
        probe_mode_index: Optional[int] = None,
    ):
        # TODO: OPR updates are calculated using chi with uniquely shifted probes, 
        # probe without shift, and probe updates that are adjoint-shifted. This is
        # not accurate, but PtychoShelves does the same. We might need to revisit this.
        mode_slicer = probe._get_probe_mode_slicer(probe_mode_index)
        chi = chi[:, mode_slicer]
        delta_p_i = delta_p_i[:, mode_slicer]
        delta_p_hat = delta_p_hat[mode_slicer]

        if probe.has_multiple_opr_modes and (
            probe.optimization_enabled(current_epoch)
            or (self.eigenmode_weight_optimization_enabled(current_epoch))
        ):
            self.update_opr_probe_modes_and_weights(probe, 
                                                    adjoint_shift_probe_update_direction, 
                                                    indices, 
                                                    chi, 
                                                    delta_p_i, 
                                                    delta_p_hat, 
                                                    obj_patches, 
                                                    current_epoch
            )

        if self.intensity_variation_optimization_enabled(current_epoch):
            delta_weights_int = self._calculate_intensity_variation_update_direction(
                probe,
                indices,
                chi,
                obj_patches,
            )
            self._apply_variable_intensity_updates(delta_weights_int)

    @timer()
    def update_opr_probe_modes_and_weights(
        self,
        probe: "Probe",
        adjoint_shift_probe_update_direction,   # what do I do for type hint here?
        indices: Tensor,
        chi: Tensor,
        delta_p_i: Tensor,
        delta_p_hat: Tensor,
        obj_patches: Tensor,
        current_epoch: int,
    ):
        """
        Update the eigenmodes of the first incoherent mode of the probe, and update the OPR mode weights.

        The default (for self.options.use_optimal_update = False) implementation below is adapted from 
        PtychoShelves code (update_variable_probe.m) and has some differences from Eq. 31 of Odstrcil (2018).
        """
        probe_data = probe.data
        weights_data = self.data
        
        batch_size = len(delta_p_i)
        n_points_total = self.n_scan_points

        # If there is only one sample in the batch, `residue_update` would become 0
        # which causes division by zero error. We skip the update if that's the case.
        if batch_size == 1:
            return

        update_eigenmode = probe.optimization_enabled(current_epoch)        # why is this needed again? To even get into this function, we need this to already be true?
        update_eigenmode_weights = self.eigenmode_weight_optimization_enabled(current_epoch)

        if self.options.use_optimal_update:
            
            rc = obj_patches.shape[-2] * obj_patches.shape[-1]
            n_spos = obj_patches.shape[0]
            
            U = probe_data[1:, 0, ...]
            
            Ws = (weights_data[ indices, 1:]).to(torch.complex64)
            
            Tsconj_chi = (obj_patches[:,0,...].conj() * chi[:,0,...])
            Tsconj_chi = adjoint_shift_probe_update_direction( indices, Tsconj_chi[:,None,...], first_mode_only=True)
            
            chi = adjoint_shift_probe_update_direction( indices, chi, first_mode_only=True)
            
            U = torch.reshape(U, (U.shape[0], rc))
            chi_vec = torch.reshape(chi[:,0,...], (n_spos, rc))
            Ts = torch.reshape(obj_patches[:,0,...], (n_spos, rc))   
            Tsconj_chi = torch.reshape(Tsconj_chi[:,0,...], (n_spos, rc)).T   
            
            # Optimal OPR weight updates
            
            if update_eigenmode_weights:
                
                delta_Ws = -2 * torch.real(U.conj() @ Tsconj_chi).to(torch.complex64)

                Ts_U_deltaWs = Ts.T * (U.T @ delta_Ws)
                numer = torch.sum(torch.real(chi_vec * Ts_U_deltaWs.H))
                denom = torch.sum(torch.real( Ts_U_deltaWs.conj() * Ts_U_deltaWs ))
                optimal_step_deltaWs = self.options.update_relaxation * (numer / denom)

                Ws = (Ws + optimal_step_deltaWs * delta_Ws.T)
 
            if (probe.representation == "sparse_code"
                and probe.options.experimental.sdl_probe_options.enabled_opr):
                
                # Optimal sparse code OPR mode updates
                    
                delta_U = -1 * Tsconj_chi @ Ws

                delta_sparse_code_probe_opr = probe.dictionary_matrix.H @ delta_U
                
                Gs = probe.dictionary_matrix @ delta_sparse_code_probe_opr @ Ws.T
                TsHGsH = Ts.H * Gs.conj()
                numer = torch.sum( torch.real(TsHGsH * chi_vec.T))
                denom = torch.sum( torch.real(TsHGsH * TsHGsH.conj()))
                optimal_step_sparse_code_probe_opr = probe.options.eigenmode_update_relaxation * (numer / denom)

                sparse_code_probe_opr = probe.get_sparse_code_probe_opr_weights()
                
                optimal_sparse_code_probe_opr = (sparse_code_probe_opr 
                                                 + optimal_step_sparse_code_probe_opr * delta_sparse_code_probe_opr.T)
                
                # Enforce sparsity constraint on sparse code
                abs_sparse_code = torch.abs(optimal_sparse_code_probe_opr)
                abs_sparse_code_sorted = torch.sort(abs_sparse_code, dim=-1, descending=True)
                sel = abs_sparse_code_sorted[0][:, probe.sparse_code_probe_nnz]
                sparse_code_mask = (abs_sparse_code >= sel[:,None])
                
                # Hard or Soft thresholding
                if probe.options.experimental.sdl_probe_options.thresholding_type_opr == 'hard':
                    optimal_sparse_code_probe_opr = optimal_sparse_code_probe_opr * sparse_code_mask
                elif probe.options.experimental.sdl_probe_options.thresholding_type_opr == 'soft':
                    optimal_sparse_code_probe_opr = ( abs_sparse_code - sel[:,None] ) * sparse_code_mask * torch.exp(1j * torch.angle(optimal_sparse_code_probe_opr))

                probe.set_sparse_code_probe_opr(optimal_sparse_code_probe_opr)
                
                # Back to dense OPR representation
                U = (probe.dictionary_matrix @ optimal_sparse_code_probe_opr.T).T
                
                # the OPR modes must have L2 norm = torch.sqrt(torch.tensor(rc))
                U = U * torch.sqrt(torch.tensor(rc)) / torch.sqrt(torch.sum(torch.abs(U)**2, -1))[:,None]

                U = torch.reshape(U, (U.shape[0], obj_patches.shape[-2], obj_patches.shape[-1]))

                probe_data[1:, 0, :, :] = U
                weights_data[indices, 1:] = Ws.real
                
                # DELETE THIS FOR FINAL MERGING
                # DELETE THIS FOR FINAL MERGING
                
                # Test the rank of the new scan position dependent probe:
                
                # probe_data_TEST = torch.reshape(probe_data[:,0,...], (probe_data.shape[0], probe_data.shape[-1] * probe_data.shape[-2]))
                # Z1 = torch.sum(probe_data[:, 0, :, :][None,...] * weights_data[indices][...,None,None], 1)
                # Z1 = torch.reshape(Z1, (Z1.shape[0], Z1.shape[1] * Z1.shape[2]))
                # Z2 = probe_data_TEST.T @ weights_data[indices, :].T.to(torch.complex64)
                # print( torch.linalg.matrix_rank(Z1) )
                # print( torch.linalg.matrix_rank(Z2) )
                
                # DELETE THIS FOR FINAL MERGING
                # DELETE THIS FOR FINAL MERGING
                
            else: 
                                
                # Optimal dense OPR mode updates:
            
                delta_U = -1 * Tsconj_chi @ Ws

                Ts_deltaU_Ws = Ts.T * (delta_U @ Ws.T)
                numer = torch.sum(torch.real(chi_vec * Ts_deltaU_Ws.H))
                denom = torch.sum(torch.real( Ts_deltaU_Ws.conj() * Ts_deltaU_Ws ))
                optimal_step_deltaU = probe.options.eigenmode_update_relaxation * (numer / denom)
                
                U = U + optimal_step_deltaU * delta_U.T
                
                # the OPR modes must have L2 norm = torch.sqrt(torch.tensor(rc))
                U = U * torch.sqrt(torch.tensor(rc)) / torch.sqrt(torch.sum(torch.abs(U)**2, -1))[:,None]
                
                U = torch.reshape(U, (U.shape[0], obj_patches.shape[-2], obj_patches.shape[-1]))
                
                probe_data[1:, 0, :, :] = U
                weights_data[indices, 1:] = Ws.real
                
        else:
            
            # Ptychoshelves method for OPR updates
            
            # FIXME: reduced relax_u/v by a factor of 10 for stability, but PtychoShelves works without this.
            relax_u = min(0.1, batch_size / n_points_total) * probe.options.eigenmode_update_relaxation
            relax_v = self.options.update_relaxation
            # Shape of delta_p_i:       (batch_size, n_probe_modes, h, w)
            # Use only the first incoherent mode
            delta_p_i = delta_p_i[:, 0, :, :]
            delta_p_hat = delta_p_hat[0, :, :]
            residue_update = delta_p_i - delta_p_hat

            # Start from the second OPR mode which is the first after the main mode - i.e., the first eigenmode.
            for i_opr_mode in range(1, probe.n_opr_modes):
                # Just take the first incoherent mode.
                eigenmode_i = probe.get_mode_and_opr_mode(mode=0, opr_mode=i_opr_mode)
                weights_i = self.get_weights(indices)[:, i_opr_mode]
                eigenmode_i, weights_i = self._update_first_eigenmode_and_weight(
                    residue_update,
                    eigenmode_i,
                    weights_i,
                    relax_u,
                    relax_v,
                    obj_patches,
                    chi,
                    update_eigenmode=update_eigenmode,
                    update_weights=self.eigenmode_weight_optimization_enabled(current_epoch),
                )

                # Project residue on this eigenmode, then subtract it.
                if i_opr_mode < probe.n_opr_modes - 1:
                    residue_update = residue_update - pmath.project(
                        residue_update, eigenmode_i, dim=(-2, -1)
                    )

                probe_data[i_opr_mode, 0, :, :] = eigenmode_i
                weights_data[indices, i_opr_mode] = weights_i
        
        if update_eigenmode:
            probe.set_data(probe_data)      
                        
        if update_eigenmode_weights:
            self.set_data(weights_data)

    @timer()
    def _update_first_eigenmode_and_weight(
        self,
        residue_update: Tensor,
        eigenmode_i: Tensor,
        weights_i: Tensor,
        relax_u: Tensor,
        relax_v: Tensor,
        obj_patches: Tensor,
        chi: Tensor,
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
                pmath.mnorm(eigenmode_update.view(-1))
            )
            eigenmode_i = eigenmode_i / pmath.mnorm(eigenmode_i.view(-1))

        if update_weights:
            # Update weights using Eq. 23a.
            # Shape of psi:                     (batch_size, h, w)
            psi = eigenmode_i * obj_patches
            # The denominator can get smaller and smaller as eigenmode_i goes down.
            # Weight update needs to be clamped.
            denom = torch.mean((torch.abs(psi) ** 2), dim=(-2, -1))
            num = torch.mean((chi[:, 0, :, :] * psi.conj()).real, dim=(-2, -1))
            weight_update = num / (denom + 0.1 * torch.mean(denom))
            # weight_update = weight_update.clamp(max=10)
            weights_i = weights_i + relax_v * weight_update

        return eigenmode_i, weights_i

    @timer()
    def _calculate_intensity_variation_update_direction(
        self,
        probe: "Probe",
        indices: Tensor,
        chi: Tensor,
        obj_patches: Tensor,
    ):
        """
        Update variable intensity scaler - i.e., the OPR mode weight corresponding to the main mode.

        This implementation is adapted from PtychoShelves code (update_variable_probe.m) and has some
        differences from Eq. 31 of Odstrcil (2018).
        """
        # Just take the first slice.
        obj_patches = obj_patches[:, 0]

        mean_probe = probe.get_mode_and_opr_mode(mode=0, opr_mode=0)
        op = obj_patches * mean_probe
        num = torch.real(op.conj() * chi[:, 0, ...])
        denom = op.abs() ** 2
        delta_weights_int_i = torch.sum(num, dim=(-2, -1)) / torch.sum(denom, dim=(-2, -1))
        # Pad it to the same shape as opr_mode_weights.
        delta_weights_int = torch.zeros_like(self.data)
        delta_weights_int[indices, 0] = delta_weights_int_i
        return delta_weights_int

    @timer()
    def _apply_variable_intensity_updates(self, delta_weights_int: Tensor):
        self.set_data(self.data + 0.1 * delta_weights_int)

    @timer()
    def smooth_weights(self):
        """
        Smooth the weights with a median filter.
        """
        weights = self.data
        if self.options.smoothing.method == "median":
            if self.n_scan_points < 81:
                logger.warning(
                    "OPR weight smoothing with median filter could "
                    "not run because the number of scan points is less than 81."
                )
                return
            weights = ip.median_filter_1d(weights.T, window_size=81).T
        elif self.options.smoothing.method == "polynomial":
            if self.n_scan_points < self.options.smoothing.polynomial_degree:
                logger.warning(
                    "OPR weight smoothing with polynomial filter could "
                    "not run because the number of scan points is less than the "
                    "polynomial smoothing degree ({}).".format(
                        self.options.smoothing.polynomial_degree
                    )
                )
                return
            inds = torch.arange(self.n_scan_points, device=weights.device, dtype=weights.dtype)
            for i_opr_mode in range(1, self.n_opr_modes):
                weights_current_mode = weights[:, i_opr_mode]
                fit_coeffs = pmath.polyfit(
                    inds, weights_current_mode, deg=self.options.smoothing.polynomial_degree
                )
                weights_smoothed = pmath.polyval(inds, fit_coeffs)
                weights[:, i_opr_mode] = 0.5 * weights_current_mode + 0.5 * weights_smoothed
        self.set_data(weights)

    @timer()
    def remove_outliers(self):
        aevol = torch.abs(self.data)
        weights = torch.min(aevol, 1.5 * torch.quantile(aevol, 0.95)) * torch.sign(self.data)
        self.set_data(weights)
