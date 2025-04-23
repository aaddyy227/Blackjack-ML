import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import time
import os

# --- Configuration ---
ENV_NAME = 'Blackjack-v1'
MODEL_FILENAME = "blackjack_dqn_policy_50m.pth"
NUM_EPISODES = 50_000_000  # <<< Increased to 50 Million episodes >>>
BATCH_SIZE = 128        # Number of experiences to sample from buffer
GAMMA = 0.99          # Discount factor
EPS_START = 1.0       # Starting epsilon for exploration
EPS_END = 0.05        # Minimum epsilon
EPS_DECAY = 2_000_000 # <<< Increased decay steps for longer training >>>
TAU = 0.005           # Soft update parameter for target network
LR = 1e-4             # Learning rate for the optimizer
BUFFER_SIZE = 500_000 # <<< Increased buffer size for longer training >>>

# --- Define the Q-Network ---
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # Simple feed-forward network
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        # Ensure input is a FloatTensor
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        # Normalize state features? (Optional but can help)
        # E.g., player sum / 32, dealer card / 11, usable ace is 0 or 1
        # Let's try without explicit normalization first for simplicity.
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# --- Replay Buffer ---
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.n_actions = env.action_space.n
        # Observation space is Tuple(Discrete(32), Discrete(11), Discrete(2))
        # We represent it as a 3-element vector
        self.n_observations = 3 
        
        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is not trained directly
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        # Exponential decay for epsilon
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return None # Return None for loss when not optimizing
            
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                      device=self.device, dtype=torch.bool)
        
        # Filter out None values before creating tensor
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        if len(non_final_next_states_list) == 0:
            # Handle case where all next states are None (all episodes in batch ended)
            next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        else:
             non_final_next_states = torch.tensor(np.array(non_final_next_states_list), 
                                               dtype=torch.float32, device=self.device)

        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device) # Added done batch

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        if len(non_final_next_states_list) > 0:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
                
        # Compute the expected Q values: reward + gamma * max_a Q(s', a)        
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss() # Huber loss
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item() # Return loss for tracking

    def update_target_network(self):
        # Soft update of the target network's weights
        # θ' = τθ + (1 - τ)θ'
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)
        
    def save_model(self, filename=MODEL_FILENAME):
        try:
            torch.save(self.policy_net.state_dict(), filename)
            print(f"Model saved successfully to {filename}")
        except Exception as e:
            print(f"Error saving model: {e}")

# --- Training Loop ---
if __name__ == "__main__":
    print(f"Starting DQN training for {NUM_EPISODES:,} episodes...") # Formatted number
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # elif torch.backends.mps.is_available(): # For MacOS Metal
    #     device = torch.device("mps")
    #     print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    env = gym.make(ENV_NAME) # Consider gym.make(ENV_NAME, sab=True) for slightly different rules if desired
    agent = DQNAgent(env, device)

    episode_rewards = []
    episode_losses = []
    start_time = time.time()
    # Use a larger reporting interval due to huge episode count
    report_interval = 1000 
    print_interval = 10000 # For newline

    for i_episode in range(NUM_EPISODES):
        state, _ = env.reset()
        # state = tuple(state) # Ensure hashable if using dicts, but tensor conversion handles it
        
        done = False
        total_reward = 0
        total_loss = 0
        steps = 0
        
        while not done:
            action_tensor = agent.select_action(state)
            action = action_tensor.item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            # next_state = tuple(next_state)
            done = terminated or truncated
            total_reward += reward
            
            # Store the transition in memory
            # Ensure next_state is None if done to handle terminal states correctly
            agent.memory.push(state, action_tensor, next_state if not done else None, reward, done)
            
            # Move to the next state
            state = next_state
            
            # Perform one step of the optimization (on the policy network)
            loss = agent.optimize_model()
            if loss is not None:
                 total_loss += loss
                 steps += 1

            # Soft update of the target network's weights
            agent.update_target_network()
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        avg_loss = total_loss / steps if steps > 0 else 0
        episode_losses.append(avg_loss)
        
        # Logging progress with updated intervals
        if (i_episode + 1) % report_interval == 0:
            avg_reward = np.mean(episode_rewards[-report_interval:]) # Avg over report interval
            avg_loss_last_interval = np.mean(episode_losses[-report_interval:])
            elapsed_time = time.time() - start_time
            time_per_episode = elapsed_time / (i_episode + 1)
            est_total_time = time_per_episode * NUM_EPISODES
            est_remaining_time = est_total_time - elapsed_time
            
            print(f"\rEpisode {i_episode+1:,}/{NUM_EPISODES:,} | Avg Reward ({report_interval} ep): {avg_reward:.3f} | Avg Loss: {avg_loss_last_interval:.5f} | Steps: {agent.steps_done:,} | Time: {elapsed_time:.1f}s | ETA: {est_remaining_time/3600:.1f} hrs", end="")
        
        if (i_episode + 1) % print_interval == 0:
             print() # Newline less frequently

    env.close()
    total_training_time = time.time() - start_time
    print(f"\n\nTraining complete in {total_training_time:.2f} seconds ({total_training_time/3600:.2f} hours).")
    
    # Save the trained policy network
    agent.save_model()

    # --- Optional: Plot results ---
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        rewards_moving_avg = np.convolve(episode_rewards, np.ones(report_interval)//report_interval, mode='valid') # Smoothed over report interval
        plt.plot(rewards_moving_avg)
        plt.title(f'Episode Rewards (Moving Average {report_interval})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        losses_moving_avg = np.convolve(episode_losses, np.ones(report_interval)//report_interval, mode='valid')
        plt.plot(losses_moving_avg)
        plt.title(f'Training Loss (Moving Average {report_interval})')
        plt.xlabel('Episode')
        plt.ylabel('Average Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("dqn_training_plot_50m.png") # New plot filename
        print("Training plot saved to dqn_training_plot_50m.png")
        # plt.show() # Don't show if running non-interactively
    except ImportError:
        print("Matplotlib not found. Skipping plot generation. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error generating plot: {e}") 