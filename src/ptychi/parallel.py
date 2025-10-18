# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Any, Optional
import os

import torch
import torch.distributed as dist
from torch import Tensor
import numpy as np

from ptychi.api.types import Numeric
from ptychi.utils import to_tensor


class MultiprocessMixin:
    backend = "nccl"
    
    @property
    def rank(self) -> int:
        try:
            return dist.get_rank()
        except ValueError:
            return 0
    
    @property
    def n_ranks(self) -> int:
        try:
            return dist.get_world_size()
        except ValueError:
            return 1
    
    def get_chunk_of_current_rank(self, tensor: Tensor) -> Tensor:
        """
        Get a chunk of the tensor for the current rank.
        
        Parameters
        ----------
        tensor : Tensor
            The tensor to be chunked. The first dimension of the tensor is assumed
            to be the batch dimension and it will be split along this dimension.
            
        Returns
        -------
        Tensor
            A chunk of the tensor for the current rank.
        """
        chunks = torch.chunk(tensor, self.n_ranks, dim=0)
        if len(chunks) != self.n_ranks:
            # torch.chunk might return fewer chunks than asked in some cases.
            chunk_size = tensor.shape[0] // self.n_ranks
            start = self.rank * chunk_size
            end = min(start + chunk_size, tensor.shape[0])
            return tensor[start:end]
        else:
            return chunks[self.rank]

    def sync_buffer(
        self, buffer: Any, 
        indices: Optional[Tensor] = None, 
        source_rank: Optional[int] = None,
        op: dist.ReduceOp = dist.ReduceOp.SUM
    ) -> Any:
        """Synchronize a buffer across ranks.
        
        Parameters
        ----------
        buffer : Any
            The buffer to be synchronized. It can be a scalar, a Python sequence,
            a NumPy array, or a PyTorch tensor.
        indices : Tensor
            If given, only the elements of the buffer at the given indices
            will be synchronized. The rest is kept unchanged. If the buffer is a scalar,
            an exception is raised.
        source_rank : int | None, optional
            If given, the buffer will be broadcasted from the given rank to all ranks.
            Otherwise, the buffer will be synchronized across all ranks through all-reduce.
        op : dist.ReduceOp, optional
            The operation to take on the buffer when it is synchronized across
            ranks through all-reduce. If broadcasting from a designated rank
            is intended, use anything for this argument.
            
        Returns
        -------
        Any
            The synchronized buffer.
        """
        orig_type = type(buffer)
        
        if isinstance(buffer, Numeric) and indices is not None:
            raise ValueError("Indices are not supported for scalar buffers.")
        
        # Convert non-tensor types to tensor.
        if isinstance(buffer, (list, tuple, np.ndarray)):
            buffer = to_tensor(buffer)
        elif isinstance(buffer, Numeric):
            buffer = to_tensor([buffer])
        elif not isinstance(buffer, Tensor):
            raise ValueError(f"Unsupported buffer type: {type(buffer)}")
        
        # Turn scalar tensor into buffer.
        unsqueezed = False
        if buffer.ndim == 0:
            buffer = buffer.unsqueeze(0)
            unsqueezed = True
        
        # For all_reduce, we need consistent shapes across all ranks
        # so we must operate on the full buffer, not sliced views
        if source_rank is not None:
            slicer = slice(None) if indices is None else indices
            dist.broadcast(buffer[slicer], src=source_rank)
        else:
            if indices is not None:
                # Create a temporary buffer for the indexed elements
                temp_buffer = torch.zeros_like(buffer)
                temp_buffer[indices] = buffer[indices]
                dist.all_reduce(temp_buffer, op=op)
                buffer[indices] = temp_buffer[indices]
            else:
                dist.all_reduce(buffer, op=op)
        
        if unsqueezed:
            buffer = buffer.squeeze(0)
        
        if orig_type is list:
            buffer = buffer.tolist()
        elif orig_type is tuple:
            buffer = tuple(buffer.tolist())
        elif orig_type is int:
            buffer = int(buffer.item())
        elif orig_type is float:
            buffer = float(buffer.item())
        elif orig_type is np.ndarray:
            buffer = buffer.cpu().numpy()
        return buffer

    def init_process_group(self, backend: str = "nccl") -> None:
        if dist.is_initialized():
            return
        dist.init_process_group(backend=backend, init_method="env://")

    def detect_launcher(self) -> str | None:
        env = os.environ
        if "GROUP_RANK" in env or "ROLE_RANK" in env or "LOCAL_WORLD_SIZE" in env:
            return "torchrun"
        elif "RANK" in env and "WORLD_SIZE" in env and "LOCAL_RANK" in env:
            return "launch"
        else:
            return None
