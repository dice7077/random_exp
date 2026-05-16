#!/usr/bin/env python3
"""Training script for AlphaZero."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from connect4.model import PolicyValueNet
from connect4.self_play import ReplayBuffer
from connect4.train import AlphaZeroTrainer
from connect4.utils import get_device, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train Connect4 AlphaZero")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    parser.add_argument(
        "--self-play-games", type=int, default=100, help="Games per self-play phase"
    )
    parser.add_argument("--train-steps", type=int, default=500, help="Training steps per iteration")
    parser.add_argument("--n-simulations", type=int, default=200, help="MCTS simulations")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--num-channels", type=int, default=64, help="ResNet channels")
    parser.add_argument("--num-res-blocks", type=int, default=4, help="ResNet blocks")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument("--log-dir", type=str, default="logs", help="TensorBoard log directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training")
    parser.add_argument(
        "--buffer-max-size", type=int, default=100000, help="Replay buffer size"
    )
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT constant")
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu), auto-detect if not specified"
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = get_device(force_cpu=args.device == "cpu")
    print(f"Using device: {device}")

    model = PolicyValueNet(
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        device=str(device),
    )
    print(f"Model created on {device}")

    replay_buffer = ReplayBuffer(max_size=args.buffer_max_size)

    trainer = AlphaZeroTrainer(
        model=model,
        replay_buffer=replay_buffer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        n_self_play_games=args.self_play_games,
        n_train_steps=args.train_steps,
        n_simulations=args.n_simulations,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        use_amp=args.amp,
        c_puct=args.c_puct,
    )

    trainer.train(n_iterations=args.iterations, resume_from=args.resume)


if __name__ == "__main__":
    main()
