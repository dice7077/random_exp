"""Utility functions."""

import random
from time import time

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device(force_cpu: bool = False) -> torch.device:
    """Get torch device (cuda if available, else cpu)."""
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Timer:
    """Context manager for timing code."""

    def __init__(self, name: str = "Timer") -> None:
        self.name = name
        self.start_time: float = 0

    def __enter__(self):
        self.start_time = time()
        return self

    def __exit__(self, *args):
        elapsed = time() - self.start_time
        print(f"{self.name}: {elapsed:.2f}s")

    @property
    def elapsed(self) -> float:
        """Elapsed time since entering context."""
        return time() - self.start_time
