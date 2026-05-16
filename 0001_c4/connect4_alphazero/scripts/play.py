#!/usr/bin/env python3
"""Play Connect4 interactively."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from connect4.cli import clear_screen, display_board, display_result
from connect4.game import Connect4Game
from connect4.model import PolicyValueNet
from connect4.players import (
    HumanPlayer,
    MCTSAlphaZeroPlayer,
    MinimaxPlayer,
    NeuralNetPlayer,
    RandomPlayer,
)
from connect4.utils import get_device, set_seed


def main():
    parser = argparse.ArgumentParser(description="Play Connect4")
    parser.add_argument(
        "--mode",
        type=str,
        default="human-ai",
        choices=[
            "human-human",
            "human-ai",
            "ai-ai",
            "human-random",
            "human-minimax",
            "minimax-ai",
        ],
        help="Game mode",
    )
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--n-simulations", type=int, default=400, help="MCTS simulations")
    parser.add_argument("--minimax-depth", type=int, default=4, help="Minimax depth")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay for AI moves (seconds)")
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu), auto-detect if not specified"
    )

    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    device = get_device(force_cpu=args.device == "cpu")
    use_color = not args.no_color

    player1 = None
    player2 = None

    if args.mode == "human-human":
        player1 = HumanPlayer("Player 1")
        player2 = HumanPlayer("Player 2")

    elif args.mode == "human-ai":
        if args.model is None:
            print("Error: --model required for human-ai mode")
            sys.exit(1)
        model = PolicyValueNet.load_checkpoint(args.model, device=str(device))
        player1 = HumanPlayer("Player (X)")
        player2 = MCTSAlphaZeroPlayer(model, n_simulations=args.n_simulations, temperature=0.0)

    elif args.mode == "ai-ai":
        if args.model is None:
            print("Error: --model required for ai-ai mode")
            sys.exit(1)
        model = PolicyValueNet.load_checkpoint(args.model, device=str(device))
        player1 = MCTSAlphaZeroPlayer(model, n_simulations=args.n_simulations, temperature=0.0)
        player2 = MCTSAlphaZeroPlayer(model, n_simulations=args.n_simulations, temperature=0.0)

    elif args.mode == "human-random":
        player1 = HumanPlayer("Player (X)")
        player2 = RandomPlayer("Random (O)")

    elif args.mode == "human-minimax":
        player1 = HumanPlayer("Player (X)")
        player2 = MinimaxPlayer("Minimax (O)", depth=args.minimax_depth)

    elif args.mode == "minimax-ai":
        if args.model is None:
            print("Error: --model required for minimax-ai mode")
            sys.exit(1)
        model = PolicyValueNet.load_checkpoint(args.model, device=str(device))
        player1 = MinimaxPlayer("Minimax (X)", depth=args.minimax_depth)
        player2 = MCTSAlphaZeroPlayer(model, n_simulations=args.n_simulations, temperature=0.0)

    if player1 is None or player2 is None:
        print("Error: Failed to initialize players")
        sys.exit(1)

    game = Connect4Game()
    players = [player1, player2]
    players[0].reset()
    players[1].reset()

    clear_screen()
    display_board(game, use_color=use_color)

    while not game.is_terminal():
        current_idx = 0 if game.current_player == 1 else 1
        player = players[current_idx]

        print(f"\n{player.name}'s turn (column 0-6):")

        if not isinstance(player, HumanPlayer):
            time.sleep(args.delay)

        try:
            action = player.select_action(game)
            game.step(action)
        except ValueError as e:
            print(f"Error: {e}")
            continue
        except KeyboardInterrupt:
            print("\nGame interrupted")
            return

        clear_screen()
        display_board(game, use_color=use_color)

    display_result(game)


if __name__ == "__main__":
    main()
