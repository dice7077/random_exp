"""CLI utilities for Connect4."""

import os

from colorama import Fore, Style, init

from .game import Connect4Game

init(autoreset=True)


def display_board(game: Connect4Game, use_color: bool = True) -> None:
    """Display the Connect4 board."""
    print("\n  0   1   2   3   4   5   6")
    print("+---+---+---+---+---+---+---+")

    for row in range(game.ROWS):
        row_str = "|"
        for col in range(game.COLS):
            cell = game.board[row, col]
            if cell == 1:
                if use_color:
                    row_str += f" {Fore.RED}X{Style.RESET_ALL} |"
                else:
                    row_str += " X |"
            elif cell == -1:
                if use_color:
                    row_str += f" {Fore.YELLOW}O{Style.RESET_ALL} |"
                else:
                    row_str += " O |"
            else:
                row_str += " . |"

        print(row_str)
        print("+---+---+---+---+---+---+---+")


def display_result(game: Connect4Game) -> None:
    """Display game result."""
    winner = game.winner()

    if winner == 0:
        print("\nGame ended in a draw!")
    elif winner == 1:
        print("\nPlayer 1 (RED/X) wins!")
    else:
        print("\nPlayer -1 (YELLOW/O) wins!")


def clear_screen() -> None:
    """Clear terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


# Monkey-patch to track last action's row for display
def _find_row_for_last_action(self) -> int:
    """Find the row of the last action."""
    if self.last_action is None:
        return -1
    for row in range(self.ROWS):
        if self.board[row, self.last_action] != 0:
            return row
    return -1


Connect4Game._find_row_for_last_action = _find_row_for_last_action
