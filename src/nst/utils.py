from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DeviceInfo:
    requested: str
    resolved: str
    mps_available: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(requested: str = "auto") -> DeviceInfo:
    requested = requested.lower()
    mps_available = torch.backends.mps.is_available()

    if requested == "auto":
        resolved = "mps" if mps_available else "cpu"
    elif requested == "mps":
        if not mps_available:
            raise RuntimeError("MPS requested but not available")
        resolved = "mps"
    elif requested == "cpu":
        resolved = "cpu"
    else:
        raise ValueError(f"Unknown device request: {requested}")

    return DeviceInfo(requested=requested, resolved=resolved, mps_available=mps_available)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
