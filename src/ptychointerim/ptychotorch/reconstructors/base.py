from typing import Optional, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import tqdm

from ptychointerim.ptychotorch.data_structures import VariableGroup, PtychographyVariableGroup
from ptychointerim.ptychotorch.utils import to_numpy


class LossTracker:

    def __init__(self, 
                 metric_function: Optional[torch.nn.Module] = None, 
                 always_compute_loss: bool = False, 
                 *args, **kwargs) -> None:
        """
        The loss tracker is used to track the loss during reconstruction.

        :param metric_function: a function that takes y_pred and y_true and returns a loss.
        :param always_compute_loss: determines the behavior of update_batch_loss. If True,
            the loss is computed using the metric function as long as y_pred and y_true
            are given. Otherwise, the tracker logs the provided loss value if it is given,
            only computing the loss when it is not.
        """
        super().__init__(*args, **kwargs)
        self.table = pd.DataFrame(columns=['epoch', 'loss'])
        self.table['epoch'] = self.table['epoch'].astype(int)
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
        
    def update_batch_loss(self, 
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
        
    def update_batch_loss_with_metric_function(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
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
                
    def check_inputs(self, *args, **kwargs):
        pass

    def build(self) -> None:
        self.check_inputs()


    def get_config_dict(self) -> dict:
        d = self.variable_group.get_config_dict()
        d.update({'reconstructor': self.__class__.__name__})
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
        self.epoch_counter = 0

    def build(self) -> None:
        super().build()
        self.build_dataloader()
        self.build_loss_tracker()
        self.build_counter()

    def build_dataloader(self):
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     generator=torch.Generator(device=torch.get_default_device()),
                                     shuffle=True)
        self.dataset.move_attributes_to_device(torch.get_default_device())

    def build_loss_tracker(self):
        self.loss_tracker = LossTracker(metric_function=self.metric_function)
        
    def build_counter(self):
        self.epoch_counter = 0

    def get_config_dict(self) -> dict:
        d = super().get_config_dict()
        d.update({'batch_size': self.batch_size, 
                  'n_epochs': self.n_epochs})
        return d
                
    def run_minibatch(self, input_data, y_true, *args, **kwargs) -> None:
        """
        Process batch, update variables, calculate loss, and update loss tracker.
        """
        raise NotImplementedError
    
    def run_pre_run_hooks(self) -> None:
        pass
    
    def run_pre_epoch_hooks(self) -> None:
        pass
    
    def run_post_update_hooks(self) -> None:
        pass
                
    def run(self, n_epochs: Optional[int] = None, *args, **kwargs):
        self.run_pre_run_hooks()
        n_epochs = n_epochs if n_epochs is not None else self.n_epochs
        for i_epoch in tqdm.trange(n_epochs):
            self.run_pre_epoch_hooks()
            for batch_data in self.dataloader:
                input_data = [x.to(torch.get_default_device()) for x in batch_data[:-1]]
                y_true = batch_data[-1].to(torch.get_default_device())

                self.run_minibatch(input_data, y_true)
                
                self.run_post_update_hooks()
            self.loss_tracker.conclude_epoch(epoch=i_epoch)
            self.loss_tracker.print_latest()
            
            self.epoch_counter += 1

    
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
        if not torch.get_default_device().type == 'cpu':
            # TODO: use CUDA stream instead of DataParallel for non-AD reconstructor. 
            # https://poe.com/s/NZUVScEEGLxBE5ZDmKc0
            self.update_step_module = torch.nn.DataParallel(self.update_step_module)
            self.update_step_module.to(torch.get_default_device())

    @staticmethod
    def compute_updates(update_step_module: torch.nn.Module, *args, **kwargs) -> Tuple[Tuple[Tensor, ...], float]:
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


class AnalyticalIterativePtychographyReconstructor(AnalyticalIterativeReconstructor):

    def __init__(self, 
                variable_group: PtychographyVariableGroup,
                dataset: Dataset,
                batch_size: int = 1,
                n_epochs: int = 100,
                metric_function: Optional[torch.nn.Module] = None,
                 *args, **kwargs) -> None:
        super().__init__(
            variable_group=variable_group, 
            dataset=dataset, 
            batch_size=batch_size, 
            n_epochs=n_epochs, 
            metric_function=metric_function, 
            *args, **kwargs
        )
