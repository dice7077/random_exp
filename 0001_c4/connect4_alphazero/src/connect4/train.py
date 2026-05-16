"""Training loop for AlphaZero."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .game import Connect4Game
from .model import PolicyValueNet
from .self_play import ReplayBuffer, SelfPlayGame


class AlphaZeroTrainer:
    """AlphaZero training trainer."""

    def __init__(
        self,
        model: PolicyValueNet,
        replay_buffer: ReplayBuffer,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        n_self_play_games: int = 100,
        n_train_steps: int = 500,
        n_simulations: int = 200,
        checkpoint_dir: str | Path = "checkpoints",
        log_dir: str | Path = "logs",
        use_amp: bool = False,
        c_puct: float = 1.5,
    ) -> None:
        self.model = model
        self.replay_buffer = replay_buffer
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_self_play_games = n_self_play_games
        self.n_train_steps = n_train_steps
        self.n_simulations = n_simulations
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.use_amp = use_amp
        self.c_puct = c_puct

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scaler = GradScaler() if self.use_amp else None
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(
        self,
        n_iterations: int,
        resume_from: Optional[str | Path] = None,
    ) -> None:
        """
        Main training loop.

        Args:
            n_iterations: Number of AlphaZero iterations
            resume_from: Path to checkpoint to resume from
        """
        start_iteration = 0
        if resume_from is not None:
            start_iteration = self.load_checkpoint(resume_from)

        self_play_gen = SelfPlayGame(
            self.model,
            n_simulations=self.n_simulations,
            c_puct=self.c_puct,
        )

        for iteration in range(start_iteration, n_iterations):
            print(f"\n=== Iteration {iteration + 1}/{n_iterations} ===")

            print("Self-play phase...")
            for _ in tqdm(range(self.n_self_play_games), desc="Self-play games"):
                examples = self_play_gen.play_game()
                self.replay_buffer.add(examples)

            print(f"Replay buffer size: {len(self.replay_buffer)}")
            self.writer.add_scalar("buffer_size", len(self.replay_buffer), iteration)

            if len(self.replay_buffer) >= self.batch_size:
                print("Training phase...")
                for step in tqdm(range(self.n_train_steps), desc="Training steps"):
                    loss_dict = self._train_step()
                    global_step = iteration * self.n_train_steps + step
                    self.writer.add_scalar("loss/policy", loss_dict["policy_loss"], global_step)
                    self.writer.add_scalar("loss/value", loss_dict["value_loss"], global_step)
                    self.writer.add_scalar("loss/total", loss_dict["total_loss"], global_step)

            self.save_checkpoint(self.checkpoint_dir / f"iter_{iteration:04d}.pt", iteration)
            self.save_checkpoint(self.checkpoint_dir / "latest.pt", iteration)

        self.writer.close()
        print("\nTraining complete!")

    def _train_step(self) -> dict[str, float]:
        """Single training step."""
        states, pis, zs = self.replay_buffer.sample(self.batch_size)

        states = torch.from_numpy(states).to(self.model.device_obj)
        pis = torch.from_numpy(pis).to(self.model.device_obj)
        zs = torch.from_numpy(zs).to(self.model.device_obj)

        self.optimizer.zero_grad()

        if self.use_amp and self.scaler is not None:
            with autocast():
                policy_logits, values = self.model(states)
                policy_loss = self._compute_policy_loss(policy_logits, pis)
                value_loss = self._compute_value_loss(values, zs)
                total_loss = policy_loss + value_loss

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            policy_logits, values = self.model(states)
            policy_loss = self._compute_policy_loss(policy_logits, pis)
            value_loss = self._compute_value_loss(values, zs)
            total_loss = policy_loss + value_loss

            total_loss.backward()
            self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }

    @staticmethod
    def _compute_policy_loss(logits: torch.Tensor, pi_targets: torch.Tensor) -> torch.Tensor:
        """Soft cross-entropy loss for policy."""
        log_probs = F.log_softmax(logits, dim=1)
        return -(pi_targets * log_probs).sum(dim=1).mean()

    @staticmethod
    def _compute_value_loss(values: torch.Tensor, z_targets: torch.Tensor) -> torch.Tensor:
        """MSE loss for value."""
        return F.mse_loss(values.squeeze(1), z_targets)

    def save_checkpoint(self, path: str | Path, iteration: int) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "iteration": iteration,
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "batch_size": self.batch_size,
                "n_simulations": self.n_simulations,
            },
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str | Path) -> int:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.model.device_obj)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        iteration = checkpoint.get("iteration", 0)
        print(f"Loaded checkpoint from {path} (iteration {iteration})")
        return iteration
