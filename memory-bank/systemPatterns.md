# System Patterns: Blackjack Strategy Advisor (DQN Version)

## Architecture Overview

The architecture now uses Deep Reinforcement Learning (DQN) with PyTorch, separating training and usage phases.

1. **Training Phase**: DQN model training using simulation and experience replay.
2. **Usage Phase**: Strategy prediction using the trained neural network via GUI.

```mermaid
graph TD
    subgraph "Training Phase (PyTorch)"
        A[Gym Blackjack Environment] --> B(Experience Replay Buffer)
        B --> C{DQN Agent}
        C -- Samples Batch --> B
        C -- Updates Policy Net --> D[Q-Network (Policy Net)]
        D -- Updates Target Net --> E[Q-Network (Target Net)]
        C -- Uses Policy Net --> A
        D -- Saves Weights --> F[Model File (.pth)]
    end
    
    subgraph "Usage Phase (GUI)"
        G[Load Model Weights] --> H[DQN Model (Evaluation Mode)]
        I[User Input via GUI] --> J[State Calculation]
        J --> K[State Tensor]
        K -- Input --> H
        H -- Q-Values --> L[Action Selection (argmax)]
        L --> M[Recommendation Display]
    end
    
    F --> G
```

## Component Relationships

### Training Component (`train_dqn_policy.py`)
- **DQN (nn.Module)**: PyTorch neural network defining the Q-value approximator.
- **ReplayBuffer**: Stores environment interaction tuples (state, action, reward, next_state, done).
- **DQNAgent**: Orchestrates training, including action selection (epsilon-greedy), sampling from the buffer, optimizing the policy network, and updating the target network.
- Uses **Gymnasium Environment** for simulation.
- Produces **blackjack_dqn_policy.pth**: Saved state dictionary of the trained policy network.

### Advisor Component (`strategy_advisor.py`)
- **DQN (nn.Module)**: Same network structure as used in training, loaded for inference.
- **GUI (Tkinter)**: Provides interface for card selection and displays results.
- **Card/Hand Logic**: Calculates state features (player sum, dealer card, usable ace) from selected cards.
- Loads the saved model weights (`.pth` file) into the network.
- Performs forward pass on the network to get Q-values for the current state and determines the best action.

## Design Patterns

### Policy Training Phase
- **Deep Reinforcement Learning (DRL)**: Uses a neural network for function approximation.
- **Deep Q-Networks (DQN)**: Specific algorithm utilizing Experience Replay and Target Networks.
- **Experience Replay**: Decouples interaction from learning, improving sample efficiency and stability.
- **Target Network**: Provides stable targets for Q-value updates.
- **Epsilon-Greedy Exploration**: Standard exploration strategy.
- **Observer Pattern** (Implicit): Training loop observes environment and agent interactions.

### Strategy Advisor Phase
- **Model-View-Controller (MVC)** (Loosely): 
  - Model: The loaded PyTorch DQN model and its prediction logic.
  - View: Tkinter GUI components.
  - Controller: Event handlers (button clicks), state calculation, and prediction triggering logic.
- **Command Pattern**: Button clicks trigger specific actions (add card, get advice, clear).

## Data Flow

1. **Training Data Flow**:
   - Agent interacts with Gym environment, generating transitions.
   - Transitions are stored in the Replay Buffer.
   - Batches are sampled from the buffer.
   - Policy network predicts Q-values for current states; Target network predicts Q-values for next states.
   - Loss is calculated and backpropagated to update the Policy network.
   - Target network is soft-updated periodically.
   - Final Policy network weights are saved.

2. **Advisor Data Flow**:
   - User selects cards via GUI.
   - State features are calculated.
   - State features are converted to a PyTorch tensor.
   - Tensor is fed into the loaded DQN model (inference mode).
   - Model outputs Q-values for Stand and Hit.
   - Action with the highest Q-value is selected.
   - Recommendation (Stand/Hit) is displayed in the GUI.

## Memory Management

- **Training**: Replay buffer holds a fixed number of transitions (can be large). Neural network weights and gradients are managed by PyTorch (potentially on GPU memory if available).
- **Advisor**: Loads the model weights into memory (CPU or GPU RAM). Minimal memory usage otherwise. 