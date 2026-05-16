#!/usr/bin/env python3
"""Evaluate model strength."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from connect4.arena import Arena, model_vs_minimax, model_vs_model, model_vs_random
from connect4.model import PolicyValueNet
from connect4.players import MCTSAlphaZeroPlayer, MinimaxPlayer, RandomPlayer
from connect4.utils import get_device, set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate Connect4 model")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "minimax", "model"],
        help="Opponent type",
    )
    parser.add_argument("--opponent-model", type=str, default=None, help="Opponent model path")
    parser.add_argument("--n-games", type=int, default=100, help="Number of games")
    parser.add_argument("--n-simulations", type=int, default=200, help="MCTS simulations")
    parser.add_argument("--minimax-depth", type=int, default=4, help="Minimax depth")
    parser.add_argument("--verbose", action="store_true", help="Print game results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu), auto-detect if not specified"
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(force_cpu=args.device == "cpu")

    print(f"Loading model from {args.model}...")
    model = PolicyValueNet.load_checkpoint(args.model, device=str(device))
    print(f"Model loaded on {device}")

    if args.opponent == "random":
        print(f"\nEvaluating vs Random ({args.n_games} games)...")
        results = model_vs_random(model, n_games=args.n_games, n_sim=args.n_simulations)

    elif args.opponent == "minimax":
        print(f"\nEvaluating vs Minimax depth={args.minimax_depth} ({args.n_games} games)...")
        results = model_vs_minimax(
            model, depth=args.minimax_depth, n_games=args.n_games, n_sim=args.n_simulations
        )

    elif args.opponent == "model":
        if args.opponent_model is None:
            print("Error: --opponent-model required for model-vs-model evaluation")
            sys.exit(1)
        print(f"\nLoading opponent model from {args.opponent_model}...")
        opponent_model = PolicyValueNet.load_checkpoint(
            args.opponent_model, device=str(device)
        )
        print(f"\nEvaluating vs Model ({args.n_games} games)...")
        results = model_vs_model(
            model, opponent_model, n_games=args.n_games, n_sim=args.n_simulations
        )

    else:
        print("Error: Unknown opponent type")
        sys.exit(1)

    print("\n=== Results ===")
    print(f"Model wins: {results['player1_wins']}")
    print(f"Opponent wins: {results['player2_wins']}")
    print(f"Draws: {results['draws']}")
    total = results['player1_wins'] + results['player2_wins'] + results['draws']
    win_rate = results['player1_wins'] / total * 100 if total > 0 else 0
    print(f"Win rate: {win_rate:.1f}%")


if __name__ == "__main__":
    main()
