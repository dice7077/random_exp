"""Self-play data generation and replay buffer."""

import pickle
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from .game import Connect4Game
from .mcts import MCTS
from .model import PolicyValueNet


class SelfPlayGame:
    """Generate self-play game data."""

    def __init__(
        self,
        model: PolicyValueNet,
        n_simulations: int = 200,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        temp_threshold: int = 10,
    ) -> None:
        self.model = model
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.temp_threshold = temp_threshold

    def play_game(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """
        Play one self-play game and return training examples.

        Returns:
            List of (state, pi, z) tuples
        """
        game = Connect4Game()
        mcts = MCTS(
            self.model,
            c_puct=self.c_puct,
            n_simulations=self.n_simulations,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_eps=self.dirichlet_eps,
        )

        examples = []
        players_history = []
        move_count = 0

        while not game.is_terminal():
            current_player = game.current_player
            players_history.append(current_player)

            add_noise = move_count < self.temp_threshold
            pi = mcts.search(game, add_noise=add_noise)

            state = game.get_state_tensor().copy()
            examples.append((state, pi.copy()))

            if move_count < self.temp_threshold:
                legal = game.legal_actions()
                pi_legal = pi[legal]
                if np.sum(pi_legal) > 0:
                    pi_legal = pi_legal / np.sum(pi_legal)
                else:
                    pi_legal = np.ones(len(legal)) / len(legal)
                action = int(np.random.choice(legal, p=pi_legal))
            else:
                action = int(np.argmax(pi))

            game.step(action)
            mcts.root = mcts.root.children.get(action)
            if mcts.root is not None:
                mcts.root.parent = None

            move_count += 1

        winner = game.winner()

        result = []
        for i, (state, pi) in enumerate(examples):
            player_at_step = players_history[i]
            z = float(winner * player_at_step) if winner != 0 else 0.0
            result.append((state, pi, z))

        return result


class ReplayBuffer:
    """Experience replay buffer for training."""

    def __init__(self, max_size: int = 100_000) -> None:
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)

    def add(self, examples: list[tuple[np.ndarray, np.ndarray, float]]) -> None:
        """Add training examples to buffer."""
        for example in examples:
            self._buffer.append(example)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch from buffer.

        Returns:
            (states [B,2,6,7], pis [B,7], zs [B])
        """
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        states = []
        pis = []
        zs = []

        for idx in indices:
            state, pi, z = self._buffer[idx]
            states.append(state)
            pis.append(pi)
            zs.append(z)

        return (
            np.array(states, dtype=np.float32),
            np.array(pis, dtype=np.float32),
            np.array(zs, dtype=np.float32),
        )

    def save(self, path: str | Path) -> None:
        """Save buffer to disk."""
        with open(path, "wb") as f:
            pickle.dump(list(self._buffer), f, protocol=4)

    def load(self, path: str | Path) -> None:
        """Load buffer from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self._buffer.clear()
            self._buffer.extend(data)

    def __len__(self) -> int:
        """Buffer size."""
        return len(self._buffer)
