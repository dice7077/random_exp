"""Player implementations for Connect4."""

import random
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .game import Connect4Game
from .mcts import MCTSPlayer
from .model import PolicyValueNet


class BasePlayer(ABC):
    """Abstract base player."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def select_action(self, game: Connect4Game) -> int:
        """Select an action for the current game state."""
        pass

    def reset(self) -> None:
        """Reset player state for a new game."""
        pass


class HumanPlayer(BasePlayer):
    """Player controlled by human input."""

    def __init__(self, name: str = "Human") -> None:
        super().__init__(name)

    def select_action(self, game: Connect4Game) -> int:
        """Get action from user input."""
        legal = game.legal_actions()

        while True:
            try:
                action = int(input(f"{self.name} (player {game.current_player}), choose column (0-6): "))
                if action in legal:
                    return action
                else:
                    print(f"Invalid action. Legal actions: {legal}")
            except (ValueError, IndexError):
                print("Invalid input. Please enter a number 0-6.")


class RandomPlayer(BasePlayer):
    """Player that chooses random legal actions."""

    def __init__(self, name: str = "Random") -> None:
        super().__init__(name)

    def select_action(self, game: Connect4Game) -> int:
        """Choose random legal action."""
        return random.choice(game.legal_actions())


class MinimaxPlayer(BasePlayer):
    """Player using minimax with alpha-beta pruning."""

    def __init__(self, name: str = "Minimax", depth: int = 4) -> None:
        super().__init__(name)
        self.depth = depth

    def select_action(self, game: Connect4Game) -> int:
        """Select best action via minimax."""
        best_action = None
        best_value = float("-inf")

        for action in game.legal_actions():
            game_copy = game.clone()
            game_copy.step(action)
            value = -self._minimax(game_copy, self.depth - 1, float("-inf"), float("inf"), True)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action if best_action is not None else random.choice(game.legal_actions())

    def _minimax(
        self, game: Connect4Game, depth: int, alpha: float, beta: float, maximizing: bool
    ) -> float:
        """Minimax with alpha-beta pruning."""
        if game.is_terminal():
            winner = game.winner()
            if winner == 1:
                return 100
            elif winner == -1:
                return -100
            else:
                return 0

        if depth == 0:
            return self._evaluate(game)

        if maximizing:
            max_eval = float("-inf")
            for action in game.legal_actions():
                game_copy = game.clone()
                game_copy.step(action)
                eval_score = self._minimax(game_copy, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for action in game.legal_actions():
                game_copy = game.clone()
                game_copy.step(action)
                eval_score = self._minimax(game_copy, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def _evaluate(self, game: Connect4Game) -> float:
        """Simple heuristic evaluation (from player 1 perspective)."""
        score = 0.0

        for row in range(game.ROWS):
            for col in range(game.COLS):
                if game.board[row, col] != 0:
                    piece = game.board[row, col]
                    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        count_player = 0
                        count_opponent = 0
                        for d in range(1, game.WIN_LEN):
                            r, c = row + d * dr, col + d * dc
                            if 0 <= r < game.ROWS and 0 <= c < game.COLS:
                                if game.board[r, c] == piece:
                                    if piece == 1:
                                        count_player += 1
                                    else:
                                        count_opponent += 1

                        if piece == 1:
                            if count_player >= 2:
                                score += 10 * count_player
                        else:
                            if count_opponent >= 2:
                                score -= 10 * count_opponent

        center_col = game.COLS // 2
        for row in range(game.ROWS):
            if game.board[row, center_col] == 1:
                score += 3
            elif game.board[row, center_col] == -1:
                score -= 3

        return score


class MCTSAlphaZeroPlayer(BasePlayer):
    """Player using MCTS with neural network guidance."""

    def __init__(
        self,
        model: PolicyValueNet,
        name: str = "AlphaZero",
        n_simulations: int = 200,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        add_noise: bool = False,
    ) -> None:
        super().__init__(name)
        self.mcts_player = MCTSPlayer(
            model=model,
            n_simulations=n_simulations,
            c_puct=c_puct,
            temperature=temperature,
            add_noise=add_noise,
        )

    def select_action(self, game: Connect4Game) -> int:
        """Select action using MCTS."""
        return self.mcts_player.select_action(game)

    def reset(self) -> None:
        """Reset MCTS tree."""
        self.mcts_player.reset()


class NeuralNetPlayer(BasePlayer):
    """Player using greedy policy from neural network (no search)."""

    def __init__(self, model: PolicyValueNet, name: str = "NeuralNet") -> None:
        super().__init__(name)
        self.model = model

    def select_action(self, game: Connect4Game) -> int:
        """Select best action according to model policy."""
        legal = game.legal_actions()
        policy_logits, _ = self.model.predict(game.get_state_tensor())

        mask = np.full(7, -1e9)
        mask[legal] = policy_logits[legal]
        best_action = int(np.argmax(mask))

        return best_action
