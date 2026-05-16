"""Connect4 AlphaZero package."""

__version__ = "0.1.0"

from .game import Connect4Game
from .model import PolicyValueNet
from .mcts import MCTS, MCTSPlayer
from .players import (
    BasePlayer,
    HumanPlayer,
    MCTSAlphaZeroPlayer,
    MinimaxPlayer,
    NeuralNetPlayer,
    RandomPlayer,
)

__all__ = [
    "Connect4Game",
    "PolicyValueNet",
    "MCTS",
    "MCTSPlayer",
    "BasePlayer",
    "HumanPlayer",
    "RandomPlayer",
    "MinimaxPlayer",
    "MCTSAlphaZeroPlayer",
    "NeuralNetPlayer",
]
