from typing import Optional

from dataclasses import dataclass


@dataclass
class OptimizationPlan:
    """
    When a `ReconstructParameter` has `optimizable == True`, this class is used to specify
    the start, stop, and stride epochs of the optimization for that parameter.
    """
    start: int = 0
    """
    The starting epoch.
    """
    
    stop: Optional[int] = None
    """
    The starting epoch. If None, optimization will run to the last epoch if the parameter
    is optimizable.
    """
    
    stride: int = 1
    """
    The stride in epochs. Optimization will run every `stride` epochs.
    """
    
    def is_enabled(self, epoch: int) -> bool:
        if self.start is not None and epoch < self.start:
            return False
        if self.stop is not None and epoch >= self.stop:
            return False
        if self.start is None:
            return True
        return (epoch - self.start) % self.stride == 0
    
    def is_in_optimization_interval(self, epoch: int) -> bool:
        if self.start is not None and epoch < self.start:
            return False
        if self.stop is not None and epoch >= self.stop:
            return False
        return True
    