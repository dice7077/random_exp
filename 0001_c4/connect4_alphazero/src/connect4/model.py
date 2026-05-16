"""Policy-Value Network for Connect4."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with Conv + BN + ReLU."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class PolicyValueNet(nn.Module):
    """Policy-Value network with ResNet backbone."""

    def __init__(
        self,
        num_channels: int = 64,
        num_res_blocks: int = 4,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_obj = torch.device(self.device_str)

        self.stem = nn.Sequential(
            nn.Conv2d(2, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )

        self.trunk = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 6 * 7, 7),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 6 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.to(self.device_obj)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [batch, 2, 6, 7]

        Returns:
            (policy_logits [batch, 7], value [batch, 1])
        """
        features = self.stem(x)
        features = self.trunk(features)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value

    def predict(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Predict policy and value for a single state.

        Args:
            state: [2, 6, 7] numpy array

        Returns:
            (policy logits [7], value scalar)
        """
        with torch.no_grad():
            x = torch.from_numpy(state).unsqueeze(0).to(self.device_obj)
            policy, value = self.forward(x)
            policy_np = policy.squeeze(0).cpu().numpy()
            value_float = value.squeeze().item()
        return policy_np, value_float

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {
                "device": self.device_str,
            },
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, device: Optional[str] = None
    ) -> "PolicyValueNet":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint.get("config", {})
        device_to_use = device or config.get("device", None)
        model = cls(device=device_to_use)
        model.load_state_dict(checkpoint["state_dict"])
        return model
