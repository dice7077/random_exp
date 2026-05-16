"""Monte Carlo Tree Search with PUCT."""

import math
from typing import Optional

import numpy as np

from .game import Connect4Game
from .model import PolicyValueNet


class MCTSNode:
    """Node in MCTS tree."""

    def __init__(self, parent: Optional["MCTSNode"] = None, action: Optional[int] = None) -> None:
        self.parent = parent
        self.action = action
        self.children: dict[int, MCTSNode] = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = 0.0

    def puct(self, parent_n: int, c_puct: float) -> float:
        """Compute PUCT value."""
        if self.N == 0:
            return c_puct * self.P * math.sqrt(parent_n)
        return self.Q + c_puct * self.P * math.sqrt(parent_n) / (1 + self.N)


class MCTS:
    """MCTS with PUCT and neural network guidance."""

    def __init__(
        self,
        model: PolicyValueNet,
        c_puct: float = 1.5,
        n_simulations: int = 200,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
    ) -> None:
        self.model = model
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.root: Optional[MCTSNode] = None

    def search(self, game: Connect4Game, add_noise: bool = False) -> np.ndarray:
        """
        Run MCTS search and return visit-count policy.

        Args:
            game: Current game state
            add_noise: Whether to add Dirichlet noise at root

        Returns:
            Policy array [7] normalized over legal actions
        """
        if self.root is None:
            self.root = MCTSNode()

        for _ in range(self.n_simulations):
            game_sim = game.clone()
            leaf, value = self._select_and_expand(self.root, game_sim)
            self._backup(leaf, value)

        if add_noise:
            self._add_dirichlet_noise()

        pi = self._get_policy(game)
        return pi

    def _select_and_expand(
        self, node: MCTSNode, game: Connect4Game
    ) -> tuple[MCTSNode, float]:
        """
        Select nodes via PUCT until leaf, then expand.

        Returns:
            (leaf node, value from expansion)
        """
        while not game.is_terminal():
            legal = game.legal_actions()

            if not node.children:
                return self._expand(node, game)

            best_action = max(
                legal,
                key=lambda a: node.children[a].puct(node.N, self.c_puct)
                if a in node.children
                else 0,
            )

            if best_action not in node.children:
                node.children[best_action] = MCTSNode(parent=node, action=best_action)

            node = node.children[best_action]
            game.step(best_action)

        return self._handle_terminal(node, game)

    def _expand(self, node: MCTSNode, game: Connect4Game) -> tuple[MCTSNode, float]:
        """Expand node using model prediction."""
        legal = game.legal_actions()

        policy_logits, value = self.model.predict(game.get_state_tensor())

        mask = np.full(7, -1e9)
        mask[legal] = policy_logits[legal]
        priors = self._softmax(mask)

        for action in legal:
            node.children[action] = MCTSNode(parent=node, action=action)
            node.children[action].P = priors[action]

        return node, value

    def _handle_terminal(self, node: MCTSNode, game: Connect4Game) -> tuple[MCTSNode, float]:
        """Handle terminal state: return value from current player perspective."""
        winner = game.winner()

        if winner == 0:
            value = 0.0
        elif winner == game.current_player:
            value = 1.0
        else:
            value = -1.0

        return node, value

    def _backup(self, node: MCTSNode, value: float) -> None:
        """
        Backup value up tree, flipping sign at each level.

        Value is from perspective of player-to-move at each node.
        """
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value
            node = node.parent

    def _add_dirichlet_noise(self) -> None:
        """Add Dirichlet noise to root prior for exploration."""
        if self.root is None or not self.root.children:
            return

        children_list = list(self.root.children.values())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(children_list))

        for child, n in zip(children_list, noise):
            child.P = (1 - self.dirichlet_eps) * child.P + self.dirichlet_eps * n

    def _get_policy(self, game: Connect4Game) -> np.ndarray:
        """Get visit-count based policy from root."""
        pi = np.zeros(7, dtype=np.float32)

        if self.root is None or not self.root.children:
            return pi

        total_visits = self.root.N
        for action, child in self.root.children.items():
            pi[action] = child.N / total_visits if total_visits > 0 else 0

        return pi

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax with numerical stability."""
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)


class MCTSPlayer:
    """Player using MCTS with neural network."""

    def __init__(
        self,
        model: PolicyValueNet,
        n_simulations: int = 200,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        add_noise: bool = False,
    ) -> None:
        self.model = model
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.add_noise = add_noise
        self.mcts: Optional[MCTS] = None

    def select_action(self, game: Connect4Game) -> int:
        """Select action using MCTS."""
        if self.mcts is None:
            self.mcts = MCTS(
                self.model,
                c_puct=self.c_puct,
                n_simulations=self.n_simulations,
            )

        pi = self.mcts.search(game, add_noise=self.add_noise)
        legal = game.legal_actions()

        if self.temperature <= 0:
            return int(legal[np.argmax(pi[legal])])
        else:
            pi_legal = pi[legal]
            if np.sum(pi_legal) > 0:
                pi_legal = pi_legal / np.sum(pi_legal)
            else:
                pi_legal = np.ones(len(legal)) / len(legal)

            return int(np.random.choice(legal, p=pi_legal))

    def reset(self) -> None:
        """Reset MCTS tree for new game."""
        self.mcts = None
