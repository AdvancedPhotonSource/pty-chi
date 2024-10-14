from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Device:
    backend: str
    ordinal: int
    name: str

    @property
    def torch_device(self) -> str:
        return f"{self.backend.lower()}:{self.ordinal}"

    @classmethod
    def CPU(cls) -> Device:
        return cls("cpu", 0, "CPU:0")


def list_available_devices() -> Sequence[Device]:
    available_devices = list()

    if torch.cpu.is_available():
        for ordinal in range(torch.cpu.device_count()):
            name = f"CPU:{ordinal}"
            device = Device("cpu", ordinal, name)
            available_devices.append(device)

    if torch.cuda.is_available():
        for ordinal in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(ordinal)
            device = Device("cuda", ordinal, name)
            available_devices.append(device)

    if torch.xpu.is_available():
        for ordinal in range(torch.xpu.device_count()):
            name = torch.xpu.get_device_name(ordinal)
            device = Device("xpu", ordinal, name)
            available_devices.append(device)

    return available_devices
