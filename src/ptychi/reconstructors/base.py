# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Optional, Tuple, Sequence, TYPE_CHECKING
import logging

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import tqdm
from ptychi.timing.timer_utils import timer
import ptychi.movies as movies

from ptychi.utils import to_numpy
import ptychi.maps as maps
import ptychi.forward_models as fm
import ptychi.api.enums as enums
import ptychi.io_handles as io
import ptychi.image_proc as ip
if TYPE_CHECKING:
    import ptychi.data_structures.parameter_group as pg
    import ptychi.api as api

logger = logging.getLogger(__name__)


class LossTracker:
    def __init__(
        self,
        metric_function: Optional[torch.nn.Module] = None,
        always_compute_loss: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        metric_function : callable, optional
            A function that takes y_pred and y_true and returns a loss.
        always_compute_loss : bool, optional
            Determines the behavior of update_batch_loss. If True,
            the loss is computed using the metric function as long as y_pred and y_true
            are given. Otherwise, the tracker logs the provided loss value if it is given,
            only computing the loss when it is not.
        """
        super().__init__(*args, **kwargs)
        self.table = pd.DataFrame(columns=["epoch", "loss", "reg_loss"])
        self.table["epoch"] = self.table["epoch"].astype(int)
        self.metric_function = metric_function
        self.epoch_loss = 0.0
        self.accumulated_num_batches = 0
        self.epoch_reg_loss = 0.0
        self.accumulated_num_reg_batches = 0
        self.epoch = 0
        self.always_compute_loss = always_compute_loss

    def conclude_epoch(self, epoch: Optional[int] = None) -> None:
        self.epoch_loss = self.epoch_loss / self.accumulated_num_batches
        self.epoch_reg_loss = self.epoch_reg_loss / max(self.accumulated_num_reg_batches, 1)
        if epoch is None:
            epoch = self.epoch
            self.epoch += 1
        else:
            self.epoch = epoch + 1
        self.table.loc[len(self.table)] = [epoch, self.epoch_loss, self.epoch_reg_loss]
        self.epoch_loss = 0.0
        self.epoch_reg_loss = 0.0
        self.accumulated_num_batches = 0
        self.accumulated_num_reg_batches = 0

    def update_batch_loss(
        self,
        y_pred: Optional[torch.Tensor] = None,
        y_true: Optional[torch.Tensor] = None,
        loss: Optional[float] = None,
    ) -> None:
        """
        Update the loss after processing a minibatch. This routine decides whether
        to compute the loss using the metric function or a provided loss value:

        - If always_compute_loss is True, the loss is always computed using the metric
          function. An exception is thrown if data (y_pred and y_true) are not provided,
          or if the metric function is not provided.
        - If always_compute_loss is False, the loss is updated using the provided loss
          as long as it is provided.
        - Otherwise, the loss is computed using the metric function and provided data.

        At least one of (y_pred, y_true) and loss must be provided. Also, in the former
        case, metric_function must be provided.

        Parameters
        ----------
        y_pred : Optional[Tensor]
            The predicted or simulated data.
        y_true : Optional[Tensor]
            The measured or labeled data.
        loss : Optional[float]
            The precalculated loss value.
        """
        data_provided = (
            y_pred is not None and y_true is not None and self.metric_function is not None
        )
        loss_provided = loss is not None
        if self.always_compute_loss:
            if not data_provided:
                raise ValueError(
                    "Always_compute_loss requires (y_pred, y_true) and metric_function to be provided."
                )
        if not (data_provided or loss_provided):
            raise ValueError(
                "One of (y_pred, y_true) and (loss,) must be provided. Also, in the former case, metric_function must be provided."
            )

        if loss_provided and not self.always_compute_loss:
            self.update_batch_loss_with_value(loss)
        else:
            self.update_batch_loss_with_metric_function(y_pred, y_true)

    def update_batch_loss_with_metric_function(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> None:
        if self.metric_function is None:
            raise ValueError("update_batch_loss_with_metric_function requires a metric function.")
        batch_loss = self.metric_function(y_pred, y_true)
        batch_loss = to_numpy(batch_loss)
        self.epoch_loss = self.epoch_loss + batch_loss
        self.accumulated_num_batches = self.accumulated_num_batches + 1

    def update_batch_loss_with_value(self, loss: float) -> None:
        loss = to_numpy(loss)
        self.epoch_loss = self.epoch_loss + loss
        self.accumulated_num_batches = self.accumulated_num_batches + 1
        
    def update_batch_regularization_loss(self, reg_loss: float) -> None:
        reg_loss = to_numpy(reg_loss)
        self.epoch_reg_loss = self.epoch_reg_loss + reg_loss
        self.accumulated_num_reg_batches = self.accumulated_num_reg_batches + 1

    def print(self) -> None:
        print(self.table)

    def print_latest(self) -> None:
        logger.info(
            "Epoch: {}, Loss: {}".format(int(self.table.iloc[-1].epoch), self.table.iloc[-1].loss)
        )

    def to_csv(self, path: str) -> None:
        self.table.to_csv(path, index=False)


class Reconstructor:
    def __init__(
        self,
        parameter_group: "pg.ParameterGroup",
        options: Optional["api.base.ReconstructorOptions"] = None,
    ) -> None:
        self.loss_tracker = LossTracker()
        self.parameter_group = parameter_group

        if options is None:
            options = self.get_option_class()()
        self.options = options

    def check_inputs(self, *args, **kwargs):
        pass

    def build(self) -> None:
        self.check_inputs()

    def get_option_class(self):
        try:
            return self.__class__.__init__.__annotations__["options"]
        except KeyError:
            return api.options.base.ReconstructorOptions


class PtychographyReconstructor(Reconstructor):
    parameter_group: "pg.PtychographyParameterGroup"

    def __init__(
        self,
        parameter_group: "pg.PtychographyParameterGroup",
        options: Optional["api.base.ReconstructorOptions"] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(parameter_group, options=options, *args, **kwargs)
        
        self.parameter_group.object.update_pos_origin_coordinates(
            self.parameter_group.probe_positions.data
        )
        self.parameter_group.object.build_roi_bounding_box(
            self.parameter_group.probe_positions
        )


class IterativeReconstructor(Reconstructor):
    def __init__(
        self,
        parameter_group: "pg.ParameterGroup",
        dataset: Dataset,
        options: Optional["api.base.ReconstructorOptions"] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        parameter_group : ds.ParameterGroup
            The ParameterGroup containing optimizable and non-optimizable parameters.
        dataset : Dataset
            The dataset containing diffraction patterns.
        options : ReconstructorOptions
            The options for the reconstruction.
        """
        super().__init__(parameter_group, options=options, *args, **kwargs)
        self.batch_size = options.batch_size
        self.dataset = dataset
        self.n_epochs = options.num_epochs
        self.dataloader = None
        self.displayed_loss_function = options.displayed_loss_function
        if self.displayed_loss_function is not None:
            self.displayed_loss_function = maps.get_loss_function_by_enum(
                options.displayed_loss_function
            )()
        self.current_epoch = 0
        self.current_minibatch = 0

    def build(self) -> None:
        super().build()
        self.build_dataloader()
        self.build_loss_tracker()
        self.build_counter()

    def build_dataloader(self, batch_sampler=None):
        data_loader_kwargs = {
            "dataset": self.dataset,
            "generator": torch.Generator(device=torch.get_default_device()),
        }
        if batch_sampler is not None:
            data_loader_kwargs["batch_sampler"] = batch_sampler
        else:
            data_loader_kwargs["batch_size"] = self.batch_size
            data_loader_kwargs["shuffle"] = True

        self.dataloader = DataLoader(**data_loader_kwargs)
        self.dataset.move_attributes_to_device(torch.get_default_device())

    def build_loss_tracker(self):
        self.loss_tracker = LossTracker(metric_function=self.displayed_loss_function)

    def build_counter(self):
        self.pbar = tqdm.tqdm(
            total=self.options.num_epochs, 
            disable=logger.level > logging.INFO, 
            leave=False
        )
        self.current_epoch = 0
    
    def prepare_batch_data(self, batch_data: Sequence[Tensor]) -> Tuple[Sequence[Tensor], Tensor]:
        input_data = batch_data[:-1]
        
        if input_data[0].ndim != 1:
            raise ValueError(
                "The first returned value of the dataset should be a 1D tensor "
                "giving the indices, but got a tensor of shape {}.".format(input_data[0].shape)
            )
            
        # Put input data on CPU so that DataParallel can distribute them later.
        input_data = [x.cpu() for x in input_data]
        
        # Put measured data on device.
        y_true = batch_data[-1]
        if y_true.device.type != torch.get_default_device().type:
            y_true = y_true.to(torch.get_default_device())
        return input_data, y_true

    def run_minibatch(self, input_data: Sequence[Tensor], y_true: Tensor, *args, **kwargs) -> None:
        """
        Process batch, update parameters, calculate loss, and update loss tracker.

        Parameters
        ----------
        input_data : list of Tensor
            A list of input data. In many cases it is [indices].
        y_true : Tensor
            The measured data.
        """
        raise NotImplementedError

    def run_pre_run_hooks(self) -> None:
        pass

    def run_pre_update_hooks(self) -> None:
        pass

    def run_pre_epoch_hooks(self) -> None:
        pass

    def run_post_update_hooks(self) -> None:
        pass

    def run_post_epoch_hooks(self) -> None:
        pass

    @timer()
    def run(self, n_epochs: Optional[int] = None, *args, **kwargs):
        if self.current_epoch == 0:
            self.run_pre_run_hooks()

        n_epochs = n_epochs if n_epochs is not None else self.n_epochs
        for _ in range(n_epochs):
            self.run_pre_epoch_hooks()
            self.current_minibatch = 0
            for batch_data in self.dataloader:
                input_data, y_true = self.prepare_batch_data(batch_data)
                self.run_pre_update_hooks()
                self.run_minibatch(input_data, y_true)
                self.run_post_update_hooks()
                self.current_minibatch += 1
            self.run_post_epoch_hooks()
            self.loss_tracker.conclude_epoch(epoch=self.current_epoch)
            self.loss_tracker.print_latest()

            if movies.MOVIES_INSTALLED:
                movies.api.update_movie_builders(self)

            self.current_epoch += 1
            self.pbar.update(1)


class IterativePtychographyReconstructor(IterativeReconstructor, PtychographyReconstructor):
    parameter_group: "pg.PtychographyParameterGroup"

    def __init__(
        self,
        parameter_group: "pg.PtychographyParameterGroup",
        options: Optional["api.base.ReconstructorOptions"] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(parameter_group, options=options, *args, **kwargs)

    def build_dataloader(self):
        batch_sampler = None
        if self.options.batching_mode == enums.BatchingModes.COMPACT:
            batch_sampler = io.PtychographyCompactBatchSampler(
                self.parameter_group.probe_positions.data.cpu(), self.batch_size
            )
        elif self.options.batching_mode == enums.BatchingModes.UNIFORM:
            batch_sampler = io.PtychographyUniformBatchSampler(
                self.parameter_group.probe_positions.data.cpu(), self.batch_size
            )
        return super().build_dataloader(batch_sampler=batch_sampler)

    def update_preconditioners(self):
        # Update preconditioner of the object only if:
        # - the preconditioner does not exist, or
        # - it is within the 10 epochs after the probe starts being optimized, or
        # - it has been 10 epochs since the preconditioner was last updated.
        if self.parameter_group.object.preconditioner is None or (
            (self.current_epoch > self.parameter_group.probe.optimization_plan.start)
            and (
                (self.current_epoch - self.parameter_group.probe.optimization_plan.start < 10)
                or (
                    (self.current_epoch - self.parameter_group.probe.optimization_plan.start) % 10
                    == 0
                )
            )
        ):
            self.parameter_group.object.update_preconditioner(
                probe=self.parameter_group.probe,
                probe_positions=self.parameter_group.probe_positions,
                patterns=self.dataset.patterns,
                use_all_modes=self.parameter_group.object.options.build_preconditioner_with_all_modes
            )
            
    def run_pre_epoch_hooks(self) -> None:
        with torch.no_grad():
            self.update_preconditioners()
            

    def run_post_epoch_hooks(self) -> None:
        with torch.no_grad():
            probe = self.parameter_group.probe
            object_ = self.parameter_group.object
            positions = self.parameter_group.probe_positions

            # Apply probe power constraint.
            if probe.options.power_constraint.is_enabled_on_this_epoch(self.current_epoch):
                probe.constrain_probe_power(
                    self.parameter_group.object, self.parameter_group.opr_mode_weights
                )

            # Apply incoherent mode orthogonality constraint.
            if probe.options.orthogonalize_incoherent_modes.is_enabled_on_this_epoch(
                self.current_epoch
            ):
                probe.constrain_incoherent_modes_orthogonality()

            # Apply OPR orthogonality constraint.
            if probe.options.orthogonalize_opr_modes.is_enabled_on_this_epoch(self.current_epoch):
                probe.constrain_opr_mode_orthogonality(
                    self.parameter_group.opr_mode_weights, update_weights_in_place=True
                )

            # Regularize multislice reconstruction.
            if object_.options.multislice_regularization.is_enabled_on_this_epoch(
                self.current_epoch
            ):
                object_.regularize_multislice()

            # Apply smoothness constraint.
            if object_.options.smoothness_constraint.is_enabled_on_this_epoch(self.current_epoch):
                object_.constrain_smoothness()

            # Remove grid artifacts.
            if object_.options.remove_grid_artifacts.is_enabled_on_this_epoch(self.current_epoch):
                object_.remove_grid_artifacts()

            # Apply position constraint.
            if positions.position_mean_constraint_enabled(self.current_epoch):
                positions.constrain_position_mean()

            # Update compact mode clustering.
            if (
                self.options.batching_mode == enums.BatchingModes.COMPACT
                and self.options.compact_mode_update_clustering
                and positions.optimization_enabled(self.current_epoch)
                and (self.current_epoch - positions.optimization_plan.start)
                % self.options.compact_mode_update_clustering_stride
                == 0
            ):
                self.dataloader.batch_sampler.update_clusters(positions.data.detach().cpu())
                
            # Apply probe support constraint.
            if probe.options.support_constraint.is_enabled_on_this_epoch(self.current_epoch):
                probe.constrain_support()
                
            # Apply probe center constraint.
            if probe.options.center_constraint.is_enabled_on_this_epoch(self.current_epoch):
                probe.center_probe()

            # Remove object-probe ambiguity.
            if object_.options.remove_object_probe_ambiguity.is_enabled_on_this_epoch(self.current_epoch):
                object_.remove_object_probe_ambiguity(probe, update_probe_in_place=True)
            
            # Smooth OPR weights.
            opr_mode_weights = self.parameter_group.opr_mode_weights
            if opr_mode_weights.options.smoothing.is_enabled_on_this_epoch(self.current_epoch):
                opr_mode_weights.smooth_weights()
            opr_mode_weights.remove_outliers()
            
            # Position affine transformation constraint.
            if (
                positions.options.affine_transform_constraint.is_position_weight_update_enabled_on_this_epoch(
                    self.current_epoch
                )
            ):
                positions.update_position_weights(probe, object_, self.options.batch_size)
            if positions.options.affine_transform_constraint.is_enabled_on_this_epoch(self.current_epoch):
                positions.update_affine_transform_matrix()
                if positions.options.affine_transform_constraint.apply_constraint:
                    positions.apply_affine_transform_constraint()


class AnalyticalIterativeReconstructor(IterativeReconstructor):
    def __init__(
        self,
        parameter_group: "pg.ParameterGroup",
        dataset: Dataset,
        options: Optional["api.base.ReconstructorOptions"] = None,
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
        self.update_step_module: torch.nn.Module = None

    def build(self) -> None:
        super().build()
        self.build_update_step_module()

    def build_update_step_module(self, *args, **kwargs):
        update_step_func = self.compute_updates
        par_group = self.parameter_group

        class EncapsulatedUpdateStep(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.parameter_module_dict = torch.nn.ModuleDict(par_group.__dict__)

            def forward(self, *args, **kwargs):
                return update_step_func(self, *args, **kwargs)

            def process_updates(self, *args):
                ret = []
                for v in args:
                    if v is not None:
                        v = v.unsqueeze(0)
                        if v.is_complex():
                            v = torch.stack([v.real, v.imag], dim=-1)
                    ret.append(v)
                return tuple(ret)

        self.update_step_module = EncapsulatedUpdateStep()
        if not torch.get_default_device().type == "cpu":
            # TODO: use CUDA stream instead of DataParallel for non-AD reconstructor.
            # https://poe.com/s/NZUVScEEGLxBE5ZDmKc0
            self.update_step_module = torch.nn.DataParallel(self.update_step_module)
            self.update_step_module.to(torch.get_default_device())

    @staticmethod
    def compute_updates(
        update_step_module: torch.nn.Module, *args, **kwargs
    ) -> Tuple[Tuple[Tensor, ...], float]:
        """
        Calculate the update vectors of optimizable parameters that should be
        applied later.

        Parameters
        ----------
        update_step_module : torch.nn.Module
            The module that contains the update step function.
        *args
            Passed to the update step function.
        **kwargs
            Passed to the update step function.

        Returns
        -------
        updates : tuple of torch.Tensor
            The update vectors of optimizable parameters. The shape of each update
            vector is [n_replica, ..., (2 if complex else none)]. `n_replica` is 1 if
            only CPU is used, otherwise it is the number of GPUs. If the update vector
            is complex, it should be returned as real tensor with its real and
            imaginary parts concatenated at the last dimension.
        loss : float
            The batch loss.
        """
        raise NotImplementedError

    def apply_updates(self, *args, **kwargs):
        raise NotImplementedError

    @timer()
    def run(self, n_epochs: Optional[int] = None, *args, **kwargs):
        with torch.no_grad():
            return super().run(n_epochs=n_epochs, *args, **kwargs)


class AnalyticalIterativePtychographyReconstructor(
    AnalyticalIterativeReconstructor, IterativePtychographyReconstructor
):
    parameter_group: "pg.PtychographyParameterGroup"

    def __init__(
        self,
        parameter_group: "pg.PtychographyParameterGroup",
        dataset: Dataset,
        options: Optional["api.base.ReconstructorOptions"] = None,
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
        self.forward_model = None
        self.build_forward_model()

    def build_forward_model(self):
        self.forward_model = fm.PlanarPtychographyForwardModel(
            parameter_group=self.parameter_group, 
            retain_intermediates=True,
            detector_size=tuple(self.dataset.patterns.shape[-2:]),
            wavelength_m=self.dataset.wavelength_m,
            free_space_propagation_distance_m=self.dataset.free_space_propagation_distance_m,
            pad_for_shift=self.options.forward_model_options.pad_for_shift,
            low_memory_mode=self.options.forward_model_options.low_memory_mode,
        )

    def run_post_epoch_hooks(self) -> None:
        with torch.no_grad():
            super().run_post_epoch_hooks()

            object_ = self.parameter_group.object

            # Apply object L1-norm constraint.
            if object_.options.l1_norm_constraint.is_enabled_on_this_epoch(self.current_epoch):
                object_.constrain_l1_norm()
                
            # Apply object L2-norm constraint.
            if object_.options.l2_norm_constraint.is_enabled_on_this_epoch(self.current_epoch):
                object_.constrain_l2_norm()
                
            # Apply total variation constraint.
            if object_.options.total_variation.is_enabled_on_this_epoch(self.current_epoch):
                object_.constrain_total_variation()

    def run_pre_run_hooks(self) -> None:
        self.prepare_data()

    def prepare_data(self, *args, **kwargs):
        self.parameter_group.probe.normalize_eigenmodes()
        logger.info("Probe eigenmodes normalized.")

    @staticmethod
    def replace_propagated_exit_wave_magnitude(
        psi: Tensor, actual_pattern_intensity: Tensor
    ) -> Tensor:
        """
        Replace the propogated exit wave amplitude.

        Parameters
        ----------
        psi : Tensor
            Predicted exit wave propagated to the detector plane.
        actual_pattern_intensity : Tensor
            The measured diffraction pattern at the detector.

        Returns
        -------
        Tensor
            Predicted exit wave with the phase from `psi` and magnitude equal to the square root
            of `actual_pattern_intensity`.
        """

        return (
            psi
            / ((psi.abs() ** 2).sum(1, keepdims=True).sqrt() + 1e-7)
            * torch.sqrt(actual_pattern_intensity + 1e-7)[:, None]
        )
        
    @timer()
    def adjoint_shift_probe_update_direction(self, indices, delta_p, first_mode_only=False):
        """
        Apply the adjoint operator of the subpixel shift to the probe done in forward model.
        
        Parameters
        ----------
        indices : Tensor
            Indices of diffraction patterns in the current batch.
        delta_p : Tensor
            A (batch_size, n_probe_modes, h, w) tensor giving the probe update direction.
        first_mode_only : bool
            If True, only the first mode is shifted.

        Returns
        -------
        Tensor
            A (batch_size, n_probe_modes, h, w) tensor giving the shifted probe update direction.
        """
        pos = self.parameter_group.probe_positions.data[indices]
        orig_shape = delta_p.shape
        fractional_shifts = pos - pos.round()
        
        if first_mode_only:
            delta_p_to_shift = delta_p[..., 0, :, :]
        else:
            delta_p_to_shift = delta_p.reshape(delta_p.shape[0] * delta_p.shape[1], *delta_p.shape[2:])
        
        delta_p_shifted = ip.shift_images(
            delta_p_to_shift, 
            fractional_shifts, 
            method=self.parameter_group.object.options.patch_interpolation_method,
            adjoint=True
        )
        
        if first_mode_only:
            delta_p = torch.cat([delta_p_shifted[..., None, :, :], delta_p[..., 1:, :, :]], dim=-3)
        else:
            delta_p = delta_p_shifted.reshape(orig_shape)
        return delta_p
