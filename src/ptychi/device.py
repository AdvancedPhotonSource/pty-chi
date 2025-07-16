# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType

import torch


_torch_accelerator_module = torch.cuda


class AcceleratorModuleWrapper:
    """A wrapper class for the accelerator device module of PyTorch.
    """
    
    @classmethod
    def get_module(cls) -> ModuleType:
        return get_torch_accelerator_module()
    
    @classmethod
    def set_module(cls, module: ModuleType):
        set_torch_accelerator_module(module)
        
    @classmethod
    def get_to_device_string(cls) -> str:
        if cls.get_module() == torch.cuda:
            return "cuda"
        elif cls.get_module() == torch.xpu:
            return "xpu"
        else:
            raise ValueError(f"Unsupported accelerator module: {cls.get_module()}")


@dataclass(frozen=True)
class Device:
    backend: str
    ordinal: int
    name: str

    @property
    def torch_device(self) -> str:
        return f"{self.backend.lower()}:{self.ordinal}"


def list_available_devices() -> Sequence[Device]:
    available_devices = list()
    accelerator_module_wrapper = AcceleratorModuleWrapper()
    accelerator_module = accelerator_module_wrapper.get_module()
    
    if accelerator_module.is_available():
        for ordinal in range(accelerator_module.device_count()):
            name = accelerator_module.get_device_name(ordinal)
            device = Device(accelerator_module_wrapper.get_to_device_string(), ordinal, name)
            available_devices.append(device)

    return available_devices


def set_torch_accelerator_module(module: ModuleType):
    """Set the global variable of the torch accelerator module. 
    By default, it is `torch.cuda`.
    
    For Intel GPUs, use `torch.xpu`.
    
    Parameters
    ----------
    module: ModuleType
        The torch accelerator module.
    """
    global _torch_accelerator_module
    _torch_accelerator_module = module


def get_torch_accelerator_module() -> ModuleType:
    """Get the global variable of the torch accelerator module. 
    By default, it is `torch.cuda`.
    
    Returns
    -------
    ModuleType
        The torch accelerator module.
    """
    return _torch_accelerator_module