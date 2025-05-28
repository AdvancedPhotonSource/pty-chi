# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Optional, TYPE_CHECKING

import torch
from torch.utils.data import Dataset
from torch import Tensor

from ptychi.reconstructors.base import (
    AnalyticalIterativePtychographyReconstructor,
)
from ptychi.metrics import MSELossOfSqrt
from ptychi.timing.timer_utils import timer
import ptychi.image_proc as ip

if TYPE_CHECKING:
    import ptychi.api as api
    import ptychi.data_structures.parameter_group as pg


class PIEReconstructor(AnalyticalIterativePtychographyReconstructor):
    """
    The ptychographic iterative engine (PIE), as described in:

    Andrew Maiden, Daniel Johnson, and Peng Li, "Further improvements to the
    ptychographical iterative engine," Optica 4, 736-745 (2017)

    Object and probe updates are calculated using the formulas in table 1 of
    Maiden (2017).

    The `step_size` parameter is equivalent to gamma in Eq. 22 of Maiden (2017)
    when `optimizer == SGD`.
    """

    parameter_group: "pg.PlanarPtychographyParameterGroup"

    def __init__(
        self,
        parameter_group: "pg.PlanarPtychographyParameterGroup",
        dataset: Dataset,
        options: Optional["api.options.pie.PIEReconstructorOptions"] = None,
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

    def build_loss_tracker(self):
        if self.displayed_loss_function is None:
            self.displayed_loss_function = MSELossOfSqrt()
        return super().build_loss_tracker()

    def check_inputs(self, *args, **kwargs):
        for var in self.parameter_group.get_optimizable_parameters():
            if "lr" not in var.optimizer_params.keys():
                raise ValueError(
                    "Optimizable parameter {} must have 'lr' in optimizer_params.".format(var.name)
                )

    @timer()
    def run_minibatch(self, input_data, y_true, *args, **kwargs):
        self.parameter_group.probe.initialize_grad()
        (delta_o, delta_p_i, delta_pos), y_pred = self.compute_updates(
            *input_data, y_true, self.dataset.valid_pixel_mask
        )
        self.apply_updates(delta_o, delta_p_i, delta_pos)
        self.loss_tracker.update_batch_loss_with_metric_function(y_pred, y_true)

    @timer()
    def compute_updates(
        self, indices: torch.Tensor, y_true: torch.Tensor, valid_pixel_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """
        Calculates the updates of the whole object, the probe, and other parameters.
        This function is called in self.update_step_module.forward.
        """
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        probe_positions = self.parameter_group.probe_positions
        opr_mode_weights = self.parameter_group.opr_mode_weights

        indices = indices.cpu()
        positions = probe_positions.tensor[indices]

        y = self.forward_model.forward(indices)
        obj_patches = self.forward_model.intermediate_variables["obj_patches"]
        psi = self.forward_model.intermediate_variables["psi"]
        psi_far = self.forward_model.intermediate_variables["psi_far"]
        unique_probes = self.forward_model.intermediate_variables.shifted_unique_probes
        
        psi_prime = self.replace_propagated_exit_wave_magnitude(psi_far, y_true)
        # Do not swap magnitude for bad pixels.
        psi_prime = torch.where(
            valid_pixel_mask.repeat(psi_prime.shape[0], probe.n_modes, 1, 1), psi_prime, psi_far
        )
        psi_prime = self.forward_model.free_space_propagator.propagate_backward(psi_prime)

        delta_exwv_i = (psi_prime - psi)
        delta_o = torch.zeros_like(object_.data)
  
        for i_slice in range(object_.n_slices - 1, -1, -1):
        
            if object_.optimization_enabled(self.current_epoch):
                step_weight = self.calculate_object_step_weight(unique_probes[i_slice])
                delta_o_patches = step_weight * delta_exwv_i
                delta_o_patches = delta_o_patches.sum(1, keepdim=True)
                delta_o_i = ip.place_patches_integer(
                    torch.zeros_like(object_.get_slice(0)),
                    positions.round().int() + object_.pos_origin_coords,
                    delta_o_patches[:, 0],
                    op="add",
                )
                
                delta_o[i_slice, ...] = delta_o_i

            delta_pos = None
            if (probe_positions.optimization_enabled(self.current_epoch) 
                and object_.optimizable
                and i_slice == self.parameter_group.probe_positions.get_slice_for_correction(object_.n_slices)
            ):
                delta_pos = torch.zeros_like(probe_positions.data)
                delta_pos[indices] = probe_positions.position_correction.get_update(
                    delta_exwv_i,
                    obj_patches[:, i_slice : i_slice + 1, ...],    
                    delta_o_patches,
                    self.forward_model.intermediate_variables.shifted_unique_probes[i_slice],
                    object_.optimizer_params["lr"],
                )

            delta_p_i = None
            if (i_slice == 0) and (probe.optimization_enabled(self.current_epoch)):
                if (self.parameter_group.probe.representation == "sparse_code"):
                    
                    # rc = delta_exwv_i.shape[-1] * delta_exwv_i.shape[-2]
                    # n_scpm = delta_exwv_i.shape[-3]
                    # n_spos = delta_exwv_i.shape[-4]
   
                    # obj_patches_vec = torch.reshape(obj_patches[:, i_slice, ...], (n_spos, 1, rc ))
                    # obj_patches_vec = torch.swapaxes(obj_patches_vec, 0, -1)
                    
                    # abs2_obj_patches = torch.abs(obj_patches_vec) ** 2
                    # z = torch.sum(abs2_obj_patches, dim = -1)
                    # z_max = torch.max(z)
                    # w = 0.9 * (z_max - z)
                    
                    # z_plus_w = torch.asarray( torch.swapaxes(z + w, 0, 1), dtype = torch.complex64 )
                    # denom = (self.parameter_group.probe.dictionary_matrix_H @ (z_plus_w * self.parameter_group.probe.dictionary_matrix))

                    # delta_exwv_i_vec = torch.reshape( delta_exwv_i, (n_scpm * n_spos, rc)).T
                    # numer = self.parameter_group.probe.dictionary_matrix_H @ delta_exwv_i_vec

                    # delta_sparse_code = torch.linalg.solve(denom, numer)     
         
                    # probe_vec = unique_probes[i_slice]
                    # sz = unique_probes[i_slice].shape
                    # probe_vec = torch.reshape( probe_vec, ( n_spos, n_scpm, sz[2] * sz[3] ))
                    # probe_vec = torch.swapaxes( probe_vec, 0, -1)
                    # probe_vec = torch.reshape( probe_vec, ( probe_vec.shape[0], probe_vec.shape[1] * probe_vec.shape[2] ))
                    # sparse_code_probe_vs_spos = self.parameter_group.probe.dictionary_matrix_pinv @ probe_vec
   
                    # sparse_code = sparse_code_probe_vs_spos + 1e-0 * delta_sparse_code
                    
                    #=====
                    
                    rc = delta_exwv_i.shape[-1] * delta_exwv_i.shape[-2]
                    n_scpm = delta_exwv_i.shape[-3]
                    n_spos = delta_exwv_i.shape[-4]
            
                    obj_patches_vec = torch.reshape(obj_patches[:, i_slice, ...], (n_spos, 1, rc ))
                    abs2_obj_patches = torch.abs(obj_patches_vec) ** 2
                    
                    z = torch.sum(abs2_obj_patches, dim = 0)
                    #z = abs2_obj_patches
                    z_max = torch.max(z)
                    w = 1.0 * (z_max - z)
                    z_plus_w = torch.swapaxes(z + w, 0, 1)
                    
                    delta_exwv = self.adjoint_shift_probe_update_direction(indices, delta_exwv_i, first_mode_only=True)
                    delta_exwv =  delta_exwv_i
                    delta_exwv = torch.sum(delta_exwv, 0)
                    #delta_exwv = delta_exwv.mean(0)
                    delta_exwv = torch.reshape( delta_exwv, (n_scpm, rc)).T
                    
                    denom = (self.parameter_group.probe.dictionary_matrix_H @ (z_plus_w * self.parameter_group.probe.dictionary_matrix))
                    numer = self.parameter_group.probe.dictionary_matrix_H @ delta_exwv
                    
                    delta_sparse_code = torch.linalg.solve(denom, numer)
                    # self.parameter_group.probe.sparse_code_probe
                    
                    delta_p = self.parameter_group.probe.dictionary_matrix @ delta_sparse_code
                    delta_p = torch.reshape( delta_p.T, (  n_scpm, delta_exwv_i.shape[-1] , delta_exwv_i.shape[-2]))
                    delta_p_i = torch.tile(delta_p, (n_spos,1,1,1))
                    
                    # # Undo subpixel shift in probe update directions.
                    # delta_p_i = self.adjoint_shift_probe_update_direction(indices, delta_p_i, first_mode_only=True)   
                                        
                    # Update sparse code in probe class
                    sparse_code = self.parameter_group.probe.get_sparse_code_weights()
                    sparse_code = sparse_code + 1e-0 * delta_sparse_code[...,None]
                    self.parameter_group.probe.set_sparse_code(sparse_code)





                    #delta_p_i = self.forward_model.shift_unique_probes(indices, delta_p_i, first_mode_only=True)
                    
                    # probe_vec = self.parameter_group.probe.dictionary_matrix @ sparse_code
                    # probe_sdl = torch.reshape( torch.swapaxes( probe_vec, 0, -1), *[delta_exwv_i.shape] )

                    # delta_p_i = probe_sdl - unique_probes[i_slice]
                    
                    # sparse_code = sparse_code_probe + delta_sparse_code
                    
                    # new_probe = self.parameter_group.probe.dictionary_matrix @ sparse_code
                    # new_probe = torch.reshape( new_probe, ( sz[2] , sz[3], n_scpm ))
                    
                    
                    
                    '''
                    
                    #=====
                    
                    probe_vec = unique_probes[i_slice]
                    probe_vec = self.adjoint_shift_probe_update_direction(indices, probe_vec, first_mode_only=True)
                    
                    probe_vec = probe_vec.mean(0)
                    #probe_vec = torch.sum( probe_vec, 0 )
                    
                    sz = unique_probes[i_slice].shape
                    #probe_vec = torch.reshape( probe_vec, ( n_spos, n_scpm, sz[2] * sz[3] ))
                    probe_vec = torch.reshape( probe_vec, ( n_scpm, sz[2] * sz[3] ))
                    probe_vec = torch.swapaxes( probe_vec, 0, -1)
                    #probe_vec = torch.reshape( probe_vec, ( probe_vec.shape[0], probe_vec.shape[1] * probe_vec.shape[2] ))
                    
                    sparse_code_probe = self.parameter_group.probe.dictionary_matrix_pinv @ probe_vec
                    
                    #sparse_code_probe = torch.reshape( sparse_code_probe, (sparse_code_probe.shape[0], n_spos, n_scpm ))
                    
                    self.parameter_group.probe.sparse_code_probe
                    
                    
                    #sparse_code_probe = torch.sum( sparse_code_probe, 1 )
                    sparse_code_probe = sparse_code_probe.mean(1)
                            
                    sparse_code = sparse_code_probe + delta_sparse_code
                    
                    new_probe = self.parameter_group.probe.dictionary_matrix @ sparse_code
                    new_probe = torch.reshape( new_probe, ( sz[2] , sz[3], n_scpm ))

                    
 
                    
                    #from PySide6.QtWidgets import QApplication
                    import numpy as np
                    import matplotlib.pyplot as plt
                    import matplotlib as mpl
                    from mpl_toolkits.axes_grid1 import make_axes_locatable
                    #mpl.use('Agg')
                    #mpl.use('TKAgg')
                    #mpl.use('WebAgg')
                    mpl.use('QTAgg')
                    mpl.rcParams['agg.path.chunksize'] = 10000

                    # A = sparse_code_probe.cpu().numpy()
                    # #A = delta_sparse_code.cpu().numpy()
                    
                    # fig, ( ax1, ax2 ) = plt.subplots( nrows = 2, ncols = 1, )
                    # plt.subplots_adjust(wspace=0.5, hspace=0.3)
                    
                    # B = np.real( A )
                    # pos1 = ax1.plot( B  ) # vmin=0.5, vmax=1.2

                    # # divider = make_axes_locatable(ax1)
                    # # cax = divider.append_axes("right", size="5%", pad=0.05)
                    # # plt.colorbar(pos1, cax=cax)
                    # # #plt.show( block = False ) 
                                
                    # B = np.imag( A )
                    # pos2 = ax2.plot( B  ) # vmin=0.5, vmax=1.2

                    # # divider = make_axes_locatable(ax2)
                    # # cax = divider.append_axes("right", size="5%", pad=0.05)
                    # # plt.colorbar(pos2, cax=cax)
                    # # #plt.show( block = False )     
                           
                    # plt.savefig( '/net/s8iddata/export/8-id-ECA/Analysis/atripath/pty-chi/tests/outputs/sparsecode.png', dpi = 300 )
                    # plt.close('all')
                          
                          
                    #A = np.conj( delta_p[:,:,0].cpu().numpy() )
                    #A = delta_p[:,:,0].cpu().numpy() 
                    A = probe_vec[0,:,:].cpu().numpy()
                    
                    #A = self.parameter_group.probe.data[0,0,...].cpu().numpy()
                    #A = delta_sparse_code.cpu().numpy()
            
                    
                    fig, ( ax1, ax2 ) = plt.subplots( nrows = 2, ncols = 1, )
                    plt.subplots_adjust(wspace=0.5, hspace=0.3)
                    
                    B = np.abs( A )
                    pos1 = ax1.imshow( B  ) # vmin=0.5, vmax=1.2

                    divider = make_axes_locatable(ax1)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(pos1, cax=cax)
                    #plt.show( block = False ) 
                                
                    B = np.angle( A )
                    pos2 = ax2.imshow( B  ) # vmin=0.5, vmax=1.2

                    divider = make_axes_locatable(ax2)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(pos2, cax=cax)
                    #plt.show( block = False )     
                          
                          
                    plt.savefig( '/net/s8iddata/export/8-id-ECA/Analysis/atripath/pty-chi/tests/outputs/probe_vec2.png', dpi = 300 )            
                    #plt.savefig( '/net/s8iddata/export/8-id-ECA/Analysis/atripath/pty-chi/tests/outputs/d_probe.png', dpi = 300 )            
                    #plt.savefig( '/net/s8iddata/export/8-id-ECA/Analysis/atripath/pty-chi/tests/outputs/unique_probe.png', dpi = 300 )
                    plt.close('all')
    
        
                    # # Update sparse code in probe object
                    # self.parameter_group.probe.set_sparse_code(sparse_code)
                        
                    '''
                    
                    #===========================================
                    # Enforce sparsity constraint on sparse code
                    
                    # abs_sparse_code = torch.abs(sparse_code)
                    # sparse_code_sorted = torch.sort(abs_sparse_code, dim=0, descending=True)
                    
                    # sel = sparse_code_sorted[0][self.parameter_group.probe.probe_sparse_code_nnz, :]
                    
                    # # hard thresholding: 
                    # sparse_code = sparse_code * (abs_sparse_code >= sel)
                    
                    #(TODO: soft thresholding option)

                    #=====================================
                    # Generate new probes vs scan position 

                    # probe_vec = self.parameter_group.probe.dictionary_matrix @ sparse_code
                    # probe_sdl = torch.reshape( torch.swapaxes( probe_vec, 0, -1), *[delta_exwv_i.shape] )

                    # delta_p_i = probe_sdl - unique_probes[i_slice]

                else:
                    step_weight = self.calculate_probe_step_weight((obj_patches[:, i_slice, ...])[:, None, ...])
                    delta_p_i = step_weight * delta_exwv_i # get delta p at each position
                    
                    # Undo subpixel shift in probe update directions.
                    delta_p_i = self.adjoint_shift_probe_update_direction(indices, delta_p_i, first_mode_only=True)
                        
                # Calculate and apply opr mode updates
                if self.parameter_group.opr_mode_weights.optimization_enabled(self.current_epoch):
                    opr_mode_weights.update_variable_probe(
                        probe,
                        indices,
                        delta_exwv_i,
                        delta_p_i,                  
                        delta_p_i.mean(0),
                        obj_patches,
                        self.current_epoch,
                        probe_mode_index=0,
                    )
                    
                # # Update sparse code in probe class
                # sparse_code = self.parameter_group.probe.get_sparse_code_weights()
                # self.parameter_group.probe.set_sparse_code(sparse_code)

            if i_slice > 0:
                delta_exwv_i = delta_exwv_i * obj_patches[:, i_slice : i_slice + 1,...].conj()
                delta_exwv_i = self.forward_model.propagate_to_previous_slice(delta_exwv_i, slice_index=i_slice) 

        return (delta_o, delta_p_i, delta_pos), y

    @timer()
    def calculate_object_step_weight(self, p: Tensor):
        """
        Calculate the weight for the object update step.

        Parameters
        ----------
        p : Tensor
            A (batch_size, n_modes, h, w) tensor giving the first OPR mode of the probe.

        Returns
        -------
        Tensor
            A (batch_size, n_modes, h, w) tensor giving the weight for the object update step.
        """
        numerator = p.abs() * p.conj()
        denominator = p.abs().sum(1, keepdim=True).max() * (
            (p.abs() ** 2).sum(1, keepdim=True) + self.parameter_group.object.options.alpha * (p.abs() ** 2).sum(1, keepdim=True).max()
        )
        step_weight = numerator.sum(1, keepdim=True) / denominator
        return step_weight

    @timer()
    def calculate_probe_step_weight(self, obj_patches: Tensor):
        """
        Calculate the weight for the probe update step.

        Parameters
        ----------
        obj_patches : Tensor
            A (batch_size, n_slices, h, w) tensor giving the object patches.

        Returns
        -------
        Tensor
        """
        # Just take the first slice.
        obj_patches = obj_patches[:, 0]

        obj_max = (torch.abs(obj_patches) ** 2).max(-1).values.max(-1).values.view(-1, 1, 1)
        numerator = obj_patches.abs() * obj_patches.conj()
        denominator = obj_max * (
            obj_patches.abs() ** 2 + self.parameter_group.probe.options.alpha * obj_max
        )
        step_weight = numerator / denominator
        return step_weight[:, None, ...]

    @timer()
    def apply_updates(self, delta_o, delta_p_i, delta_pos, *args, **kwargs):
        """
        Apply updates to optimizable parameters given the updates calculated by self.compute_updates.

        Parameters
        ----------
        delta_o : Tensor
            A (h, w, 2) tensor of object update vector.
        delta_p_i : Tensor
            A (n_patches, n_opr_modes, n_modes, h, w, 2) tensor of probe update vector.
        delta_pos : Tensor
            A (n_positions, 2) tensor of probe position vectors.
        """
        object_ = self.parameter_group.object
        probe_positions = self.parameter_group.probe_positions

        if delta_o is not None:
            object_.set_grad(-delta_o)
            object_.optimizer.step()

        if delta_p_i is not None:
            mode_slicer = self.parameter_group.probe._get_probe_mode_slicer(None)
            self.parameter_group.probe.set_grad(-delta_p_i.mean(0), slicer=(0, mode_slicer))
            self.parameter_group.probe.optimizer.step()

        if delta_pos is not None:
            probe_positions.set_grad(-delta_pos)
            probe_positions.step_optimizer(
                limit=probe_positions.options.correction_options.update_magnitude_limit
            )


class EPIEReconstructor(PIEReconstructor):
    """
    The extended ptychographic iterative engine (ePIE), as described in:

    Andrew Maiden, Daniel Johnson, and Peng Li, "Further improvements to the
    ptychographical iterative engine," Optica 4, 736-745 (2017)

    Object and probe updates are calculated using the formulas in table 1 of
    Maiden (2017).

    The `step_size` parameter is equivalent to gamma in Eq. 22 of Maiden (2017)
    when `optimizer == SGD`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @timer()
    def calculate_object_step_weight(self, p: Tensor):
        p_max = (torch.abs(p) ** 2).sum(1, keepdim=True).max()
        step_weight = self.parameter_group.object.options.alpha * p.conj().sum(1, keepdim=True) / p_max
        return step_weight

    @timer()
    def calculate_probe_step_weight(self, obj_patches: Tensor):
        # Just take the first slice.
        obj_patches = obj_patches[:, 0]

        obj_max = (torch.abs(obj_patches) ** 2).max(-1).values.max(-1).values.view(-1, 1, 1)
        step_weight = self.parameter_group.probe.options.alpha * obj_patches.conj() / obj_max
        step_weight = step_weight[:, None]
        return step_weight


class RPIEReconstructor(PIEReconstructor):
    """
    The regularized ptychographic iterative engine (rPIE), as described in:

    Andrew Maiden, Daniel Johnson, and Peng Li, "Further improvements to the
    ptychographical iterative engine," Optica 4, 736-745 (2017)

    Object and probe updates are calculated using the formulas in table 1 of
    Maiden (2017).

    The `step_size` parameter is equivalent to gamma in Eq. 22 of Maiden (2017)
    when `optimizer == SGD`.

    To get the momentum-accelerated PIE (mPIE), use `optimizer == SGD` and use
    the optimizer settings `{'momentum': eta, 'nesterov': True}` where `eta` is
    the constant used in  Eq. 19 of Maiden (2017).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @timer()
    def calculate_object_step_weight(self, p: Tensor):
        
        # apply multimodal update
        abs2_p = (torch.abs(p) ** 2).sum(1)
        p_max = abs2_p.max()
        step_weight = p.conj().sum(1) / (
            (1 - self.parameter_group.object.options.alpha) * abs2_p
            + self.parameter_group.object.options.alpha * p_max
        )
        
        # # use only first mode
        # abs2_p = (torch.abs(p) ** 2)[:, 0]
        # p_max = abs2_p.max()
        # step_weight = p.conj()[:, 0] / (
        #     (1 - self.parameter_group.object.options.alpha) * abs2_p
        #     + self.parameter_group.object.options.alpha * p_max
        # )
        
        return step_weight[:, None,...]

    @timer()
    def calculate_probe_step_weight(self, obj_patches: Tensor):
        # Just take the first slice.
        obj_patches = obj_patches[:, 0]

        obj_max = (torch.abs(obj_patches) ** 2).max(-1).values.max(-1).values.view(-1, 1, 1)
        step_weight = obj_patches.conj() / (
            (1 - self.parameter_group.probe.options.alpha) * (torch.abs(obj_patches) ** 2)
            + self.parameter_group.probe.options.alpha * obj_max
        )
        step_weight = step_weight[:, None]
        return step_weight
