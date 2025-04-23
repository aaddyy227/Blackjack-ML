# Active Context: Blackjack Strategy Advisor (DQN Version)

## Current Focus
Refactored the project to use Deep Q-Networks (DQN) with PyTorch for training the Blackjack strategy, enabling potential GPU acceleration. Updated the GUI advisor to load and use the trained DQN model.

## Implementation Status

### Components Created/Modified
1. **DQN Training Script (`train_dqn_policy.py`)**
   - Implements the DQN algorithm using PyTorch.
   - Defines a Q-Network (nn.Module), Replay Buffer, and DQNAgent class.
   - Includes training loop with experience replay, target network updates, and epsilon-greedy exploration.
   - Detects and utilizes CUDA GPU if available.
   - Trains for a configurable number of episodes (default 50k).
   - Saves the trained policy network weights to `blackjack_dqn_policy.pth`.

2. **Strategy Advisor GUI (`strategy_advisor.py`)**
   - Modified to load the PyTorch model (`.pth` file) instead of the pickle Q-table.
   - Instantiates the DQN network structure.
   - Loads the saved state dictionary into the model.
   - Performs model inference (`policy_net(state_tensor)`) to get Q-values.
   - Recommends action based on the highest Q-value.
   - Retains the visual card selection interface.

3. **Supporting Files**
   - Updated `requirements.txt` to include `torch`.
   - Updated `README.md`, `techContext.md`, `systemPatterns.md`, `progress.md` to reflect the DQN approach.

4. **Outdated Files**
   - `generate_policy.py` (tabular MC) is now superseded by `train_dqn_policy.py`.
   - `.pkl` policy files are superseded by `.pth` model files.

## Current Design Decisions

1. **Deep Reinforcement Learning (DQN)**: Chosen for its ability to potentially train faster (especially with GPU) and handle more complex state spaces if needed in the future, compared to tabular methods.
2. **PyTorch Framework**: Selected for its Pythonic interface and robust CUDA support.
3. **Visual Card Selection GUI**: Maintained the user-friendly card selection interface.
4. **Fixed Number of Episodes**: Training runs for a predefined number of episodes (50k default). More advanced stopping criteria (e.g., based on convergence) were not implemented for simplicity.

## Next Steps

1. **Install Dependencies**: Ensure PyTorch and other requirements are installed (`pip install torch -r requirements.txt`).
2. **Train DQN Model**: Run `python train_dqn_policy.py`. Monitor output for CUDA detection and training progress. This will take time depending on hardware.
3. **Run Advisor**: Once training is complete and `blackjack_dqn_policy.pth` exists, run `python strategy_advisor.py`.
4. **Testing & Tuning**: Test the advisor's recommendations. If performance is suboptimal, consider adjusting hyperparameters in `train_dqn_policy.py` (e.g., `NUM_EPISODES`, `LR`, `EPS_DECAY`) and retraining. 