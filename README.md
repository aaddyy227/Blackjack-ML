# Blackjack Strategy Advisor (DQN Version)

A Python application that provides optimal move recommendations for Blackjack based on a Deep Q-Network (DQN) trained with PyTorch.

## Overview

This project implements a Blackjack strategy advisor using Deep Reinforcement Learning (specifically DQN). It trains a neural network to approximate the optimal action-value function (Q-function) through simulated Blackjack games. The trained network is then used to recommend the best move (Hit or Stand) given:

1. Your current hand total
2. The dealer's visible card
3. Whether you have a usable Ace (counting as 11)

This version uses PyTorch and can leverage a CUDA-enabled GPU for faster training compared to tabular methods.

## Requirements

- Python 3.8+
- PyTorch (>= 1.10.0)
- NVIDIA GPU with CUDA 12.1+ compatible drivers (for GPU acceleration)
- Required packages:
  - numpy>=1.20.0
  - gymnasium>=0.28.1
  - tkinter (usually comes with Python)
  - matplotlib (optional, for plotting training results)

## Installation

1. Clone or download this repository.
2. **Install PyTorch**: 
   - **For CUDA 12.1 (Recommended for NVIDIA GPU):** 
     Check the official command at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). It will likely resemble:
     ```bash
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
   - **For CPU only:**
     ```bash
     pip3 install torch torchvision torchaudio
     ```
3. Install the remaining required packages:
   ```bash
   pip install -r requirements.txt
   # Optional for plotting:
   # pip install matplotlib 
   ```

## Usage

### Step 1: Train the DQN Model (First Time Only)

Before using the advisor, you need to train the DQN model and generate the policy file. This only needs to be done once, or when you want to retrain.

```bash
python train_dqn_policy.py
```

This script will:
- Detect if a CUDA GPU is available (requires correct PyTorch installation and drivers) and use it, otherwise fall back to CPU.
- Train the DQN agent for a set number of episodes (default 50,000,000, adjustable in the script).
- Print progress including average rewards and loss.
- Save the trained model weights to `blackjack_dqn_policy_50m.pth`.
- Optionally generate a plot `dqn_training_plot_50m.png` if matplotlib is installed.

**Warning:** Training for 50 million episodes will take a very long time, potentially many hours or days, even on a GPU.

### Step 2: Run the Strategy Advisor

Once the model file (`blackjack_dqn_policy_50m.pth`) has been generated, you can run the advisor:

```bash
python strategy_advisor.py
```

### Using the Advisor GUI

1. Click buttons under "Add Card:" to add cards to your hand.
2. Click a button under "Dealer Shows:" to set the dealer's visible card.
3. The player hand value (and usable Ace status) updates automatically.
4. Click "Get Advice" or press Enter.
5. The recommended action (HIT or STAND), based on the DQN model's prediction, will be displayed.
6. Use "Clear Hand" to reset the player and dealer cards.

## Pushing to GitHub

To push this project to your GitHub repository (`https://github.com/aaddyy227/promakoo.git`):

1. **Initialize Git (if not already):**
   ```bash
   git init
   ```
2. **Stage all files:**
   ```bash
   git add .
   ```
3. **Commit the changes:**
   ```bash
   git commit -m "feat: Add Blackjack DQN Advisor project"
   ```
4. **Add the remote repository:**
   ```bash
   # Check if origin already exists: git remote -v
   # If it exists and is wrong, remove it: git remote remove origin
   git remote add origin https://github.com/aaddyy227/promakoo.git
   # Or if origin exists and points elsewhere, set the URL:
   # git remote set-url origin https://github.com/aaddyy227/promakoo.git 
   ```
5. **Push to GitHub:**
   *Ensure your Git is configured with your GitHub credentials.*
   ```bash
   # Push to the 'main' branch (use 'master' if that's your default)
   git push -u origin main 
   ```

## How It Works

The application uses Deep Q-Networks (DQN), a Deep Reinforcement Learning algorithm:

1. **Neural Network (Q-Network)**: A feed-forward neural network takes the game state (player sum, dealer card, usable ace) as input and outputs the estimated Q-values for each possible action (Stand, Hit).
2. **Experience Replay**: During training, transitions (state, action, reward, next_state, done) are stored in a replay buffer. The network learns by sampling mini-batches from this buffer, improving stability and efficiency.
3. **Target Network**: A separate target network (a delayed copy of the main Q-network) is used to calculate target Q-values, further stabilizing training.
4. **Epsilon-Greedy Exploration**: The agent balances exploring random actions and exploiting the learned policy using a decaying epsilon value.
5. **Optimization**: The Q-Network is trained using an optimizer (AdamW) and a loss function (Smooth L1 Loss / Huber Loss) to minimize the difference between predicted Q-values and target Q-values.

The advisor loads the weights of the trained Q-Network and uses it to predict the best action for the current state entered via the GUI.

## Troubleshooting

### Common Issues

1. **PyTorch / CUDA Errors**:
   - Ensure PyTorch is installed correctly for your system (CPU or specific CUDA version, e.g., 12.1).
   - If using GPU, make sure your NVIDIA drivers and CUDA toolkit version are compatible with your PyTorch build.
   - Check output of `train_dqn_policy.py` to see if CUDA is detected.

2. **"Model file not found"**:
   - You need to run `python train_dqn_policy.py` *successfully* before running the advisor.
   - Ensure the script completes and saves `blackjack_dqn_policy_50m.pth`.

3. **Incorrect Advice / Poor Performance**:
   - DQN training can be sensitive to hyperparameters (learning rate, buffer size, epsilon decay, network size). The defaults provided are reasonable starting points but may need tuning.
   - Try training for more episodes if performance seems poor (adjust `NUM_EPISODES` in `train_dqn_policy.py`).
   - Ensure the state representation matches between training and the advisor.

4. **Unicode Card Symbols Not Displaying**: 
   - The GUI attempts to use fonts like "Segoe UI Symbol" (Windows) or "DejaVu Sans" (Linux). If cards don't display correctly, you may need to install one of these fonts or modify the font settings in `strategy_advisor.py`.

5. **Tkinter errors**: 
   - Make sure Tkinter is installed with your Python distribution (usually default).
   - On Linux, you might need to install it: `sudo apt-get install python3-tk`

6. **Git Errors**: 
   - `fatal: remote origin already exists.`: Use `git remote remove origin` before `git remote add ...` or use `git remote set-url origin ...` instead.
   - `Authentication failed`: Ensure your Git is configured correctly to authenticate with GitHub (e.g., using SSH keys or a Personal Access Token).
   - `src refspec main does not match any`: Your default branch might be `master` instead of `main`. Use `git push -u origin master`.

## Notes

- This advisor only handles basic Hit/Stand decisions.
- The training process uses the Gymnasium Blackjack-v1 environment. 