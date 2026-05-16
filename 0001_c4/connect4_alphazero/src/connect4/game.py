"""Connect4 game environment."""

import copy
from typing import Optional

import numpy as np


class Connect4Game:
    """6x7 Connect4 game with numpy board representation."""

    ROWS = 6
    COLS = 7
    WIN_LEN = 4

    def __init__(self) -> None:
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = 1
        self.last_action: Optional[int] = None
        self.done = False
        self._winner = 0

    def reset(self) -> "Connect4Game":
        """Reset game state."""
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = 1
        self.last_action = None
        self.done = False
        self._winner = 0
        return self

    def clone(self) -> "Connect4Game":
        """Deep copy of game state."""
        return copy.deepcopy(self)

    def legal_actions(self) -> list[int]:
        """Return list of legal column indices."""
        return [col for col in range(self.COLS) if self.board[0, col] == 0]

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Execute action, return (board, reward, done).

        Args:
            action: Column index 0-6

        Returns:
            (board copy, 0.0 for ongoing or draw, done flag)
            Winner status determined by is_terminal() and winner()
        """
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action: {action}")

        row = self._find_row(action)
        self.board[row, action] = self.current_player
        self.last_action = action

        if self._check_win(row, action):
            self.done = True
            self._winner = self.current_player
        elif self._is_board_full():
            self.done = True
            self._winner = 0

        self.current_player *= -1
        return self.board.copy(), 0.0, self.done

    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self.done

    def winner(self) -> int:
        """Return winner: 1, -1, or 0 (draw/ongoing)."""
        return self._winner

    def get_state_tensor(self) -> np.ndarray:
        """
        Get board as [2, 6, 7] tensor from current player's perspective.

        Channel 0: current player's stones
        Channel 1: opponent's stones
        """
        state = np.zeros((2, self.ROWS, self.COLS), dtype=np.float32)
        state[0] = (self.board == self.current_player).astype(np.float32)
        state[1] = (self.board == -self.current_player).astype(np.float32)
        return state

    def _find_row(self, col: int) -> int:
        """Find lowest empty row in column."""
        for row in range(self.ROWS - 1, -1, -1):
            if self.board[row, col] == 0:
                return row
        raise ValueError(f"Column {col} is full")

    def _check_win(self, row: int, col: int) -> bool:
        """Check if placing at (row, col) creates a win."""
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            for d in [-1, 1]:
                r, c = row + d * dr, col + d * dc
                while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.board[r, c] == player:
                    count += 1
                    r += d * dr
                    c += d * dc
            if count >= self.WIN_LEN:
                return True
        return False

    def _is_board_full(self) -> bool:
        """Check if all columns are full."""
        return all(self.board[0, col] != 0 for col in range(self.COLS))
