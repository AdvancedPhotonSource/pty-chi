from typing import Optional, Tuple, Sequence

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import tqdm

import ptychointerim.ptychotorch.data_structures as ds
from ptychointerim.ptychotorch.utils import to_numpy
import ptychointerim.api as api
import ptychointerim.maps as maps
import ptychointerim.forward_models as fm


class LossTracker:
    def __init__(
        self,
        metric_function: Optional[torch.nn.Module] = None,
        always_compute_loss: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        The loss tracker is used to track the loss during reconstruction.

        :param metric_function: a function that takes y_pred and y_true and returns a loss.
        :param always_compute_loss: determines the behavior of update_batch_loss. If True,
            the loss is computed using the metric function as long as y_pred and y_true
            are given. Otherwise, the tracker logs the provided loss value if it is given,
            only computing the loss when it is not.
        """
        super().__init__(*args, **kwargs)
        self.table = pd.DataFrame(columns=["epoch", "loss"])
        self.table["epoch"] = self.table["epoch"].astype(int)
        self.metric_function = metric_function
        self.epoch_loss = 0.0
        self.accumulated_num_batches = 0
        self.epoch = 0
        self.always_compute_loss = always_compute_loss

    def conclude_epoch(self, epoch: Optional[int] = None) -> None:
        self.epoch_loss = self.epoch_loss / self.accumulated_num_batches
        if epoch is None:
            epoch = self.epoch
            self.epoch += 1
        else:
            self.epoch = epoch + 1
        self.table.loc[len(self.table)] = [epoch, self.epoch_loss]
        self.epoch_loss = 0.0
        self.accumulated_num_batches = 0

    def update_batch_loss(
        self,
        y_pred: Optional[torch.Tensor] = None,
        y_true: Optional[torch.Tensor] = None,
        loss: Optional[float] = None,
    ) -> None:
        data_provided = y_pred is not None and y_true is not None
        loss_provided = loss is not None
        if self.always_compute_loss:
            if not data_provided:
                raise ValueError("Always_compute_loss requires (y_pred, y_true) to be provided.")
        if not (data_provided or loss_provided):
            raise ValueError("One of (y_pred, y_true) and (loss,) must be provided.")

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

    def print(self) -> None:
        print(self.table)

    def print_latest(self) -> None:
        print(
            "Epoch: {}, Loss: {}".format(int(self.table.iloc[-1].epoch), self.table.iloc[-1].loss)
        )

    def to_csv(self, path: str) -> None:
        self.table.to_csv(path, index=False)


class Reconstructor:
    def __init__(
        self,
        parameter_group: "ds.ParameterGroup",
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

    def get_config_dict(self) -> dict:
        d = self.parameter_group.get_config_dict()
        d.update({"reconstructor": self.__class__.__name__})
        return d


class PtychographyReconstructor(Reconstructor):
    parameter_group: "ds.PtychographyParameterGroup"

    def __init__(
        self,
        parameter_group: "ds.PtychographyParameterGroup",
        options: Optional["api.base.ReconstructorOptions"] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(parameter_group, options=options, *args, **kwargs)


class IterativeReconstructor(Reconstructor):
    def __init__(
        self,
        parameter_group: "ds.ParameterGroup",
        dataset: Dataset,
        options: Optional["api.base.ReconstructorOptions"] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Iterative reconstructor base class.

        :param parameter_group: The ParameterGroup containing optimizable and non-optimizable parameters.
        :param dataset: The dataset containing diffraction patterns.
        :param batch_size: The batch size.
        :param n_epochs: The number of epochs.
        :param metric_function: The function that computes the tracked cost. Different from the
            loss_function argument in some reconstructors, this function is only used for cost tracking
            and is not involved in the reconstruction math.
        """
        super().__init__(parameter_group, options=options, *args, **kwargs)
        self.batch_size = options.batch_size
        self.dataset = dataset
        self.n_epochs = options.num_epochs
        self.dataloader = None
        self.metric_function = options.metric_function
        if self.metric_function is not None:
            self.metric_function = maps.get_loss_function_by_enum(options.metric_function)()
        self.current_epoch = 0

    def build(self) -> None:
        super().build()
        self.build_dataloader()
        self.build_loss_tracker()
        self.build_counter()

    def build_dataloader(self):
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            generator=torch.Generator(device=torch.get_default_device()),
            shuffle=True,
        )
        self.dataset.move_attributes_to_device(torch.get_default_device())

    def build_loss_tracker(self):
        self.loss_tracker = LossTracker(metric_function=self.metric_function)

    def build_counter(self):
        self.current_epoch = 0

    def get_config_dict(self) -> dict:
        d = super().get_config_dict()
        d.update({"batch_size": self.batch_size, "n_epochs": self.n_epochs})
        return d

    def run_minibatch(self, input_data: Sequence[Tensor], y_true: Tensor, *args, **kwargs) -> None:
        """
        Process batch, update parameters, calculate loss, and update loss tracker.

        :param input_data: a list of input data. In many cases it is [indices].
        :param y_true: the measured data.
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

    def run(self, n_epochs: Optional[int] = None, *args, **kwargs):
        self.run_pre_run_hooks()
        n_epochs = n_epochs if n_epochs is not None else self.n_epochs
        for _ in tqdm.trange(n_epochs):
            self.run_pre_epoch_hooks()
            for batch_data in self.dataloader:
                input_data = [x.to(torch.get_default_device()) for x in batch_data[:-1]]
                y_true = batch_data[-1].to(torch.get_default_device())

                self.run_pre_update_hooks()
                self.run_minibatch(input_data, y_true)
                self.run_post_update_hooks()
            self.run_post_epoch_hooks()
            self.loss_tracker.conclude_epoch(epoch=self.current_epoch)
            self.loss_tracker.print_latest()

            self.current_epoch += 1


class IterativePtychographyReconstructor(IterativeReconstructor, PtychographyReconstructor):
    parameter_group: "ds.PtychographyParameterGroup"

    def __init__(
        self,
        parameter_group: "ds.PtychographyParameterGroup",
        options: Optional["api.base.ReconstructorOptions"] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(parameter_group, options=options, *args, **kwargs)

    def run_post_epoch_hooks(self) -> None:
        with torch.no_grad():
            probe = self.parameter_group.probe
            object_ = self.parameter_group.object

            # Apply probe power constraint.
            if (
                probe.probe_power > 0.0
                and self.current_epoch >= probe.optimization_plan.start
                and (self.current_epoch - probe.optimization_plan.start)
                % probe.probe_power_constraint_stride
                == 0
            ):
                probe.constrain_probe_power(
                    self.parameter_group.object, self.parameter_group.opr_mode_weights
                )

            # Apply incoherent mode orthogonality constraint.
            if (
                probe.has_multiple_incoherent_modes
                and probe.orthogonalize_incoherent_modes
                and self.current_epoch >= probe.optimization_plan.start
                and (self.current_epoch - probe.optimization_plan.start)
                % probe.orthogonalize_incoherent_modes_stride
                == 0
            ):
                probe.constrain_incoherent_modes_orthogonality()

            # Apply OPR orthogonality constraint.
            if probe.opr_mode_orthogonalization_enabled(self.current_epoch):
                weights = self.parameter_group.opr_mode_weights
                weights_data = probe.constrain_opr_mode_orthogonality(weights)
                weights.set_data(weights_data)

            # Apply smoothness constraint.
            if object_.smoothness_constraint_enabled(self.current_epoch):
                object_.constrain_smoothness()

            # Apply total variation constraint.
            if object_.total_variation_enabled(self.current_epoch):
                object_.constrain_total_variation()
                
            # Remove grid artifacts.
            if object_.remove_grid_artifacts_enabled(self.current_epoch):
                object_.remove_grid_artifacts()


class AnalyticalIterativeReconstructor(IterativeReconstructor):
    def __init__(
        self,
        parameter_group: "ds.ParameterGroup",
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

        This function will be encapsulated in a torch.nn.Module so that it works with
        DataParallel and DistributedDataParallel.

        Do not apply the update vectors to the optimizable parameters in this function.
        When called in a DataParallel or DistributedDataParallel context, the buffers
        of optimizable parameters are copies so the updates will not persist. Instead,
        collect the returned update vectors, reduce them, and apply the updates in
        `apply_updates` which is called outside DataParallel.

        :return: the update vectors and the batch loss. The shape of each update vector
            is [n_replica, ..., (2 if complex else none)]. `n_replica` is 1 if only
            CPU is used, otherwise it is the number of GPUs. If the update vector is
            complex, it should be returned as real tensor with its real and imaginary
            parts concatenated at the last dimension.
        """
        raise NotImplementedError

    def apply_updates(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, n_epochs: Optional[int] = None, *args, **kwargs):
        with torch.no_grad():
            return super().run(n_epochs=n_epochs, *args, **kwargs)


class AnalyticalIterativePtychographyReconstructor(
    AnalyticalIterativeReconstructor, IterativePtychographyReconstructor
):
    parameter_group: "ds.PtychographyParameterGroup"

    def __init__(
        self,
        parameter_group: "ds.PtychographyParameterGroup",
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
        if self.parameter_group.object.options.type == api.enums.ObjectTypes.TWO_D:
            self.forward_model = fm.Ptychography2DForwardModel(
                self.parameter_group, retain_intermediates=True
            )
        elif self.parameter_group.object.options.type == api.enums.ObjectTypes.MULTISLICE:
            self.forward_model = fm.MultislicePtychographyForwardModel(
                self.parameter_group,
                retain_intermediates=True,
                wavelength_m=self.dataset.wavelength_m,
            )

    def run_post_epoch_hooks(self) -> None:
        with torch.no_grad():
            super().run_post_epoch_hooks()

            object_ = self.parameter_group.object

            # Apply object L1-norm constraint.
            if object_.l1_norm_constraint_enabled(self.current_epoch):
                object_.constrain_l1_norm()
