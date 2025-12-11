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


def get_rank():
    try:
        return dist.get_rank()
    except ValueError:
        return 0


def get_world_size():
    try:
        return dist.get_world_size()
    except ValueError:
        return 1


class MultiprocessMixin:
    backend = "nccl"
    
    @property
    def rank(self) -> int:
        return get_rank()
    
    @property
    def n_ranks(self) -> int:
        return get_world_size()

    def get_chunk_of_current_rank(
        self,
        tensor: Tensor,
        return_chunk_sizes: bool = False,
    ) -> Tensor:
        """
        Get a chunk of the tensor for the current rank.

        Parameters
        ----------
        tensor : Tensor
            The tensor to be chunked. The first dimension of the tensor is assumed
            to be the batch dimension and it will be split along this dimension.
        return_chunk_sizes : bool
            If True, return a list of integers representing the chunk sizes for each rank.

        Returns
        -------
        Tensor
            A chunk of the tensor for the current rank.
        list[int]
            A list of integers representing the chunk sizes for each rank, Returned
            only if return_chunk_sizes is True.
        """
        chunks = torch.chunk(tensor, self.n_ranks, dim=0)
        if len(chunks) != self.n_ranks:
            # torch.chunk might return fewer chunks than asked in some cases.
            chunk_size = tensor.shape[0] // self.n_ranks
            start = self.rank * chunk_size
            end = min(start + chunk_size, tensor.shape[0])
            chunk = tensor[start:end]
            if return_chunk_sizes:
                chunk_sizes = [
                    min(r * chunk_size + chunk_size, tensor.shape[0]) - r * chunk_size
                    for r in range(self.n_ranks)
                ]
        else:
            chunk = chunks[self.rank]
            if return_chunk_sizes:
                chunk_sizes = [len(c) for c in chunks]
        if return_chunk_sizes:
            return chunk, chunk_sizes
        return chunk

    def sync_buffer(
        self,
        buffer: Any,
        indices: Optional[Tensor] = None,
        source_rank: Optional[int] = None,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
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
        
        slicer = slice(None) if indices is None else indices
        if source_rank is not None:
            dist.broadcast(buffer[slicer], src=source_rank)
        else:
            dist.all_reduce(buffer[slicer], op=op)
        
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
