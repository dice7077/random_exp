"""Arena for evaluating players."""

from typing import Optional

from tqdm import tqdm

from .game import Connect4Game
from .model import PolicyValueNet
from .players import BasePlayer, MCTSAlphaZeroPlayer, MinimaxPlayer, RandomPlayer


class Arena:
    """Arena for playing games between two players."""

    def __init__(self, player1: BasePlayer, player2: BasePlayer) -> None:
        self.player1 = player1
        self.player2 = player2

    def play_games(self, n_games: int, verbose: bool = False) -> dict[str, int]:
        """
        Play n games alternating who goes first.

        Returns:
            {"player1_wins": int, "player2_wins": int, "draws": int}
        """
        results = {"player1_wins": 0, "player2_wins": 0, "draws": 0}

        for i in tqdm(range(n_games), desc="Arena games"):
            if i % 2 == 0:
                outcome = self._play_one_game(self.player1, self.player2)
            else:
                outcome = self._play_one_game(self.player2, self.player1)

            if outcome == 1:
                if i % 2 == 0:
                    results["player1_wins"] += 1
                else:
                    results["player2_wins"] += 1
            elif outcome == -1:
                if i % 2 == 0:
                    results["player2_wins"] += 1
                else:
                    results["player1_wins"] += 1
            else:
                results["draws"] += 1

            if verbose:
                print(f"Game {i+1}: outcome={outcome}")

        return results

    def _play_one_game(self, first_player: BasePlayer, second_player: BasePlayer) -> int:
        """
        Play one game.

        Returns:
            1 if first player wins, -1 if second player wins, 0 for draw
        """
        game = Connect4Game()
        players = [first_player, second_player]
        first_player.reset()
        second_player.reset()

        while not game.is_terminal():
            current_idx = 0 if game.current_player == 1 else 1
            action = players[current_idx].select_action(game)
            game.step(action)

        winner = game.winner()
        if winner == 0:
            return 0
        elif winner == 1:
            return 1
        else:
            return -1


def model_vs_random(
    model: PolicyValueNet, n_games: int = 100, n_sim: int = 200
) -> dict[str, int]:
    """Evaluate model vs random player."""
    mcts_player = MCTSAlphaZeroPlayer(model, n_simulations=n_sim, temperature=0.0)
    random_player = RandomPlayer()
    arena = Arena(mcts_player, random_player)
    return arena.play_games(n_games)


def model_vs_minimax(
    model: PolicyValueNet, depth: int = 4, n_games: int = 40, n_sim: int = 200
) -> dict[str, int]:
    """Evaluate model vs minimax player."""
    mcts_player = MCTSAlphaZeroPlayer(model, n_simulations=n_sim, temperature=0.0)
    minimax_player = MinimaxPlayer(depth=depth)
    arena = Arena(mcts_player, minimax_player)
    return arena.play_games(n_games)


def model_vs_model(
    model1: PolicyValueNet,
    model2: PolicyValueNet,
    n_games: int = 100,
    n_sim: int = 200,
) -> dict[str, int]:
    """Evaluate model1 vs model2."""
    player1 = MCTSAlphaZeroPlayer(model1, n_simulations=n_sim, temperature=0.0)
    player2 = MCTSAlphaZeroPlayer(model2, n_simulations=n_sim, temperature=0.0)
    arena = Arena(player1, player2)
    return arena.play_games(n_games)
