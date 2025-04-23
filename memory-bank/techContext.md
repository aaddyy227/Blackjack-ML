# Technical Context: Blackjack Strategy Advisor (DQN Version)

## Technology Stack

### Core Technologies
- **Python 3.8+**: The primary programming language.
- **PyTorch**: Deep learning framework used for DQN implementation and GPU acceleration.
- **Gymnasium (formerly OpenAI Gym)**: Provides the Blackjack environment used for training.
- **NumPy**: Used for numerical operations, experience replay buffer management.
- **Tkinter**: Built-in Python GUI toolkit used for the advisor interface.

### Project Structure
```
blackjack/
├── train_dqn_policy.py   # Script to train the DQN model (PyTorch)
├── strategy_advisor.py   # GUI application (loads PyTorch model)
├── blackjack_dqn_policy.pth # (Generated) Saved DQN model weights
├── generate_policy.py    # (Outdated) Old script for tabular MC method
├── blackjack_policy*.pkl # (Outdated) Old policy files
├── requirements.txt      # Project dependencies (includes torch)
├── README.md             # Project documentation (updated for DQN)
└── memory-bank/          # Memory bank files
```

## Development Setup

### Environment
- Python 3.8+
- PyTorch installation (CPU or CUDA version)
- Standard libraries + dependencies listed in requirements.txt

### Dependencies
- **torch>=1.10.0**: Core PyTorch library for tensor operations, neural networks, autograd.
- **numpy>=1.20.0**: Numerical computations.
- **gymnasium>=0.28.1**: Blackjack environment.
- **tkinter**: GUI toolkit.
- **matplotlib** (Optional): For plotting training progress.

### Installation
1. Install PyTorch: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2. Install other dependencies:
```bash
pip install -r requirements.txt
pip install matplotlib # Optional
```

## Technical Implementation Details

### Policy Training (`train_dqn_policy.py`)
- Uses **Deep Q-Networks (DQN)** algorithm.
- **Q-Network**: Simple feed-forward neural network (3 input -> 128 hidden -> 128 hidden -> 2 output) implemented in PyTorch.
- **Experience Replay**: Stores transitions in a `ReplayBuffer` (deque) and samples mini-batches for training.
- **Target Network**: Uses a separate target network, updated softly (using `TAU`) from the policy network for stable learning.
- **Optimizer**: AdamW.
- **Loss Function**: Smooth L1 Loss (Huber Loss).
- **Exploration**: Epsilon-greedy strategy with exponential decay.
- **Device Handling**: Automatically detects and uses CUDA GPU if available, otherwise CPU.
- State representation (input to NN): `[player_sum, dealer_card_value, usable_ace_flag]` (3 features).
- Action space: 0 (Stand) or 1 (Hit).
- Training episodes: Default **50,000,000** (adjustable, requires significant time).

### Advisor GUI (`strategy_advisor.py`)
- Tkinter interface allowing visual card selection.
- **Loads PyTorch model**: Loads the saved `blackjack_dqn_policy_50m.pth` state dictionary into the DQN network structure.
- Sets model to evaluation mode (`model.eval()`).
- Calculates hand value and usable ace status from selected cards.
- Creates state feature tensor for the model.
- **Performs inference**: Uses the loaded model (`policy_net`) to predict Q-values for the current state.
- Recommends action based on the `argmax` of the predicted Q-values.
- Provides color-coded advice (green for Hit, red for Stand).

## Technical Constraints
- DQN performance is sensitive to hyperparameter tuning.
- Training 50 million episodes is extremely time-consuming, even with GPU.
- Requires PyTorch installation, potentially with CUDA setup for GPU usage.
- Still only handles basic Hit/Stand decisions. 