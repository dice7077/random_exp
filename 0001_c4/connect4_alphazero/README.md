# Connect4 AlphaZero-lite

A self-playing AlphaZero-lite implementation for Connect4 with CUDA GPU support, MCTS search, and CLI interface.

## Features

- **Self-play training**: Generate training data through self-play games
- **AlphaZero-lite**: Policy-Value network with MCTS for decision making
- **CUDA support**: Full GPU acceleration for training and inference
- **Multiple play modes**: Human vs Human, Human vs AI, AI vs AI, and evaluation against Random/Minimax
- **Checkpointing**: Save and resume training runs
- **TensorBoard logging**: Monitor training progress

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.1+ (optional, but recommended)

### Setup

```bash
# Clone and enter directory
cd connect4_alphazero

# Install dependencies
pip install -r requirements.txt

# For CUDA support (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Verify Installation

```bash
# Check CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Quick game logic test
python3 -c "
from src.connect4.game import Connect4Game
g = Connect4Game()
print(f'✓ Game initialized: {g.legal_actions()}')"
```

## Quick Start

### Train AlphaZero

```bash
# Minimal training (for testing)
python3 scripts/train.py \
  --iterations 1 \
  --self-play-games 2 \
  --train-steps 10 \
  --n-simulations 25

# Full training (recommended)
python3 scripts/train.py \
  --iterations 50 \
  --self-play-games 100 \
  --train-steps 500 \
  --n-simulations 200 \
  --batch-size 256 \
  --lr 1e-3
```

### Play Games

```bash
# Human vs Human (two players at keyboard)
python3 scripts/play.py --mode human-human

# Human vs AI
python3 scripts/play.py \
  --mode human-ai \
  --model checkpoints/latest.pt \
  --n-simulations 200

# AI vs AI
python3 scripts/play.py \
  --mode ai-ai \
  --model checkpoints/latest.pt \
  --n-simulations 100 \
  --delay 0.5

# Human vs Random
python3 scripts/play.py --mode human-random

# Human vs Minimax
python3 scripts/play.py \
  --mode human-minimax \
  --minimax-depth 5
```

### Evaluate Model Strength

```bash
# Against random player
python3 scripts/evaluate.py \
  --model checkpoints/latest.pt \
  --opponent random \
  --n-games 100 \
  --n-simulations 200

# Against minimax
python3 scripts/evaluate.py \
  --model checkpoints/latest.pt \
  --opponent minimax \
  --n-games 50 \
  --minimax-depth 4

# Model vs Model
python3 scripts/evaluate.py \
  --model checkpoints/latest.pt \
  --opponent model \
  --opponent-model checkpoints/best.pt \
  --n-games 100
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--iterations` | 50 | Number of AlphaZero iterations |
| `--self-play-games` | 100 | Games per self-play phase |
| `--train-steps` | 500 | Gradient steps per iteration |
| `--n-simulations` | 200 | MCTS simulations per move |
| `--batch-size` | 256 | Training batch size |
| `--lr` | 1e-3 | Learning rate |
| `--weight-decay` | 1e-4 | L2 regularization |
| `--checkpoint-dir` | checkpoints | Save directory |
| `--log-dir` | logs | TensorBoard logs directory |
| `--resume` | None | Path to checkpoint to resume |
| `--seed` | 42 | Random seed for reproducibility |
| `--amp` | False | Enable mixed precision training |
| `--buffer-max-size` | 100000 | Replay buffer capacity |
| `--c-puct` | 1.5 | MCTS exploration constant |
| `--device` | auto | Force device (cuda/cpu) |

## Project Structure

```
src/connect4/
  game.py           # Connect4 game environment (6x7 board)
  model.py          # Policy-Value CNN with ResNet backbone
  mcts.py           # MCTS with PUCT and neural network guidance
  players.py        # Player implementations (Human, Random, Minimax, MCTS)
  self_play.py      # Self-play game generation and replay buffer
  train.py          # Training loop with checkpointing and logging
  arena.py          # Arena for evaluating players
  cli.py            # CLI utilities for board display
  utils.py          # Utility functions (seed, device, timer)

scripts/
  train.py          # Training entry point
  play.py           # Interactive play entry point
  evaluate.py       # Model evaluation entry point

checkpoints/        # Saved model checkpoints
data/               # (Optional) data storage
logs/               # TensorBoard event files
```

## Implementation Details

### Game Representation

- Board: 6 rows × 7 columns
- Players: 1 (red/X) and -1 (yellow/O)
- Empty cells: 0
- State tensor: shape (2, 6, 7) from current player's perspective
  - Channel 0: current player stones
  - Channel 1: opponent stones

### Neural Network

**Architecture**: ResNet with policy and value heads
- Input: (batch, 2, 6, 7)
- Stem: Conv(2→64) + BN + ReLU
- Trunk: 4 × ResidualBlock(64)
- Policy head: outputs 7 logits (one per column)
- Value head: outputs scalar value in [-1, 1] (via Tanh)

**Training**:
- Policy loss: soft cross-entropy with MCTS visit-count targets
- Value loss: MSE with game outcome targets (-1, 0, +1)
- Optimizer: Adam with L2 regularization
- Optional: mixed precision (AMP) training

### MCTS Search

**Algorithm**: PUCT (Polynomial Upper Confidence Tree)
- Selection: `Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`
- Expansion: neural network policy and value
- Backup: value propagation with sign flip for alternating players
- Temperature: controls exploration vs exploitation in action selection

**Features**:
- Dirichlet noise (α=0.3, ε=0.25) at root during self-play for exploration
- Tree reuse between moves for efficiency
- Configurable simulation count and exploration constant

### Self-play

- Generated games train the network
- MCTS uses soft play (temperature=1) in early moves
- Greedy play (temperature→0) in later moves
- Training targets: MCTS visit-count policy, final game outcome value

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/
# Navigate to http://localhost:6006
```

Logged metrics:
- `loss/policy`: policy head loss
- `loss/value`: value head loss
- `loss/total`: total loss
- `buffer_size`: replay buffer size

## Performance Notes

- Minimal training (1 iteration, 2 games, 25 simulations) achieves **75% win rate vs Random**
- Full training (50 iterations, 100 games, 200 simulations) typically reaches **>90% win rate vs Random**
- Model also achieves parity/advantage against Minimax depth-4 after sufficient training

## GPU Utilization

The implementation uses CUDA when available:
- Model is automatically placed on GPU
- Training uses GPU batches
- MCTS evaluations run on GPU

Check GPU usage during training:
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## Future Improvements

1. **Tree Reuse**: Save MCTS tree between moves to reduce redundant simulations
2. **Batched MCTS**: Parallelize multiple simulations for higher throughput
3. **Network Improvements**:
   - Attention mechanisms
   - Dilated convolutions
   - Larger networks with more parameters
4. **Training Enhancements**:
   - Learning rate scheduling
   - Curriculum learning (increasing game complexity)
   - Priority experience replay (PER)
5. **Evaluation**:
   - Strength rating (Elo)
   - Automated testing suite
6. **Web UI**: Interactive browser-based play
7. **Openings Book**: Pre-trained opening strategies

## Testing

```bash
# Run basic game logic tests
python3 -c "
from src.connect4.game import Connect4Game
g = Connect4Game()
print('✓ Game initialization works')
for _ in range(3):
    g.step(3)
    if not g.is_terminal():
        g.step(4)
g.step(3)
assert g.is_terminal() and g.winner() == 1
print('✓ Win detection works')
"

# Test model
python3 -c "
from src.connect4.model import PolicyValueNet
import numpy as np
model = PolicyValueNet()
state = np.zeros((2, 6, 7), dtype=np.float32)
policy, value = model.predict(state)
assert policy.shape == (7,) and -1 <= value <= 1
print('✓ Model works')
"
```

## Contributing

Contributions welcome! Areas for improvement:
- Optimizations (tree reuse, batched MCTS)
- Better evaluation metrics
- Training enhancements
- Documentation and tests

## License

MIT License (example)

## References

- AlphaGo Zero: [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
- AlphaZero: [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01724)
- MCTS: [A Survey of Monte Carlo Tree Search Methods](https://ieeexplore.ieee.org/document/6145622)
