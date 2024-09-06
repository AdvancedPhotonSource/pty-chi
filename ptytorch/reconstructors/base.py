from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ptytorch.data_structures import VariableGroup


class LossTracker:

    def __init__(self, metric_function: Optional[torch.nn.Module] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.table = pd.DataFrame(columns=['epoch', 'loss'])
        self.table['epoch'] = self.table['epoch'].astype(int)
        self.metric_function = metric_function
        self.epoch_loss = 0.0
        self.accumulated_batch_size = 0
        self.epoch = 0

    def conclude_epoch(self, epoch: Optional[int] = None) -> None:
        self.epoch_loss = self.epoch_loss / self.accumulated_batch_size
        if epoch is None:
            epoch = self.epoch
            self.epoch += 1
        else:
            self.epoch = epoch + 1
        self.table.loc[len(self.table)] = [epoch, self.epoch_loss]
        self.epoch_loss = 0.0
        self.accumulated_batch_size = 0
        
    def update_batch_loss(self, 
                          y_pred: Optional[torch.Tensor] = None, 
                          y_true: Optional[torch.Tensor] = None, 
                          loss: Optional[float] = None,
                          batch_size: Optional[int] = None
    ) -> None:
        data_provided = y_pred is not None and y_true is not None
        loss_provided = loss is not None and batch_size is not None
        assert (data_provided or loss_provided) - (data_provided and loss_provided), \
            'One and only one of (y_pred, y_true) and (loss, batch_size) must be provided.'
        if data_provided:
            self.update_batch_loss_with_metric_function(y_pred, y_true)
        else:
            self.update_batch_loss_with_value(loss, batch_size)
        
    def update_batch_loss_with_metric_function(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        assert self.metric_function is not None, \
            "update_batch_loss_with_metric_function requires a metric function."
        batch_loss = self.metric_function(y_pred, y_true)
        batch_size = len(y_pred)
        self.epoch_loss = self.epoch_loss + batch_loss
        self.accumulated_batch_size = self.accumulated_batch_size + batch_size
        
    def update_batch_loss_with_value(self, loss: float, batch_size: int) -> None:
        self.epoch_loss = self.epoch_loss + loss
        self.accumulated_batch_size = self.accumulated_batch_size + batch_size

    def print(self) -> None:
        print(self.table)

    def print_latest(self) -> None:
        print('Epoch: {}, Loss: {}'.format(
            int(self.table.iloc[-1].epoch),
            self.table.iloc[-1].loss)
        )

    def to_csv(self, path: str) -> None:
        self.table.to_csv(path, index=False)


class Reconstructor:

    def __init__(self, variable_group: VariableGroup):
        self.loss_tracker = LossTracker()
        self.variable_group = variable_group

    def build(self) -> None:
        pass

    def get_config_dict(self) -> dict:
        d = self.variable_group.get_config_dict()
        return d


class IterativeReconstructor(Reconstructor):

    def __init__(self,
                 variable_group: VariableGroup,
                 dataset: Dataset,
                 batch_size: int = 1,
                 n_epochs: int = 100,
                 metric_function: Optional[torch.nn.Module] = None,
                 *args, **kwargs
    ) -> None:
        """
        Iterative reconstructor base class.
        
        :param variable_group: The variable group containing optimizable and non-optimizable variables.
        :param dataset: The dataset containing diffraction patterns.
        :param batch_size: The batch size.
        :param n_epochs: The number of epochs.
        :param metric_function: The function that computes the tracked cost. Different from the
            loss_function argument in some reconstructors, this function is only used for cost tracking
            and is not involved in the reconstruction math.
        """
        super().__init__(variable_group, *args, **kwargs)
        self.batch_size = batch_size
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.dataloader = None
        self.metric_function = metric_function

    def build(self) -> None:
        super().build()
        self.build_dataloader()
        self.build_loss_tracker()

    def build_dataloader(self):
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     generator=torch.Generator(device=torch.get_default_device()),
                                     shuffle=True)

    def build_loss_tracker(self):
        self.loss_tracker = LossTracker(metric_function=self.metric_function)

    def get_config_dict(self) -> dict:
        d = super().get_config_dict()
        d.update({'batch_size': self.batch_size, 'n_epochs': self.n_epochs})
        return d


class AnalyticalIterativeReconstructor(IterativeReconstructor):

    def __init__(self,
        variable_group: VariableGroup,
        dataset: Dataset,
        batch_size: int = 1,
        n_epochs: int = 100,
        metric_function: Optional[torch.nn.Module] = None,
        *args, **kwargs
    ) -> None:
        super().__init__(
            variable_group=variable_group,
            dataset=dataset,
            batch_size=batch_size,
            n_epochs=n_epochs,
            metric_function=metric_function,
            *args, **kwargs)
        self.update_step_module: torch.nn.Module = None

    def build(self) -> None:
        super().build()
        self.build_update_step_module()

    def build_update_step_module(self, *args, **kwargs):
        update_step_func = self.compute_updates
        var_group = self.variable_group

        class EncapsulatedUpdateStep(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.variable_module_dict = torch.nn.ModuleDict(var_group.__dict__)

            def forward(self, *args, **kwargs):
                return update_step_func(self, *args, **kwargs)

        self.update_step_module = EncapsulatedUpdateStep()
        if not torch.get_default_device().type == 'cpu':
            # TODO: use CUDA stream instead of DataParallel for non-AD reconstructor. 
            # https://poe.com/s/NZUVScEEGLxBE5ZDmKc0
            self.update_step_module = torch.nn.DataParallel(self.update_step_module)
            self.update_step_module.to(torch.get_default_device())

    @staticmethod
    def compute_updates(update_step_module: torch.nn.Module, *args, **kwargs) -> float:
        """
        Calculate the update vectors of optimizable variables that should be
        applied later. 

        This function will be encapsulated in a torch.nn.Module so that it works with
        DataParallel and DistributedDataParallel.

        Do not apply the update vectors to the optimizable variables in this function. 
        When called in a DataParallel or DistributedDataParallel context, the buffers
        of optimizable variables are copies so the updates will not persist. Instead,
        collect the returned update vectors, reduce them, and apply the updates in
        `apply_updates` which is called outside DataParallel. 

        When DataParallel is in use, the returned update vectors will have an additional
        dimension in the front corresponding to the number of replicas.

        :return: the update vectors.
        """
        raise NotImplementedError

    def apply_updates(self, *args, **kwargs):
        raise NotImplementedError