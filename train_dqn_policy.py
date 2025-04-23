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
CHECKPOINT_FILE = "dqn_checkpoint.pth"
MANUAL_CHECKPOINT_TRIGGER = "save_checkpoint.trigger"
NUM_EPISODES = 50_000_000
BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 1_500_000
TAU = 0.005
LR = 1e-4
BUFFER_SIZE = 500_000
CHECKPOINT_INTERVAL = 1_000_000

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
        # We reent it as a 3-element vector
        self.n_observations = 3 
        
        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is not trained directly
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.steps_done = 0
        self.scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

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
                with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                    q_values = self.policy_net(state_tensor)
                    return q_values.max(1)[1].view(1, 1)
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
        with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        if len(non_final_next_states_list) > 0:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].float()
                
        # Compute the expected Q values: reward + gamma * max_a Q(s', a)        
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss() # Huber loss
        loss = criterion(state_action_values.float(), expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
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
            print(f"\nFinal model saved successfully to {filename}")
        except Exception as e:
            print(f"\nError saving final model: {e}")

    def save_checkpoint(self, filename, episode):
        checkpoint = {
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'scaler_state_dict': self.scaler.state_dict()
        }
        try:
            torch.save(checkpoint, filename)
            print(f"\nCheckpoint saved to {filename} at episode {episode:,}")
        except Exception as e:
            print(f"\nError saving checkpoint: {e}")

    def load_checkpoint(self, filename):
        if not os.path.exists(filename):
            print(f"Checkpoint file {filename} not found. Starting from scratch.")
            return 0

        try:
            checkpoint = torch.load(filename, map_location='cpu')
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode']
            self.policy_net.to(self.device)
            self.target_net.to(self.device)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            self.steps_done = checkpoint.get('steps_done', start_episode * (BUFFER_SIZE / NUM_EPISODES))

            if self.scaler.is_enabled() and 'scaler_state_dict' in checkpoint:
                 self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                 print("Loaded GradScaler state.")
            elif self.scaler.is_enabled():
                 print("GradScaler state not found in checkpoint, initializing new scaler.")

            print(f"Checkpoint loaded from {filename}. Resuming from episode {start_episode + 1:,}")
            return start_episode
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            return 0

# --- Training Loop ---
if __name__ == "__main__":
    print(f"Starting DQN training for {NUM_EPISODES:,} episodes...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    env = gym.make(ENV_NAME)
    agent = DQNAgent(env, device)

    start_episode = agent.load_checkpoint(CHECKPOINT_FILE)
    agent.steps_done = start_episode * (BUFFER_SIZE / NUM_EPISODES)

    episode_rewards = []
    episode_losses = []
    start_time = time.time()
    report_interval = 1000 
    print_interval = 10000

    for i_episode in range(start_episode, NUM_EPISODES):
        state, _ = env.reset()
        
        done = False
        total_reward = 0
        total_loss = 0
        steps = 0
        
        while not done:
            action_tensor = agent.select_action(state)
            action = action_tensor.item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            agent.memory.push(state, action_tensor, next_state if not done else None, reward, done)
            
            state = next_state
            
            loss = agent.optimize_model()
            if loss is not None:
                 total_loss += loss
                 steps += 1

            agent.update_target_network()
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        avg_loss = total_loss / steps if steps > 0 else 0
        episode_losses.append(avg_loss)
        
        if (i_episode + 1) % CHECKPOINT_INTERVAL == 0:
            agent.save_checkpoint(CHECKPOINT_FILE, i_episode + 1)
            
        if os.path.exists(MANUAL_CHECKPOINT_TRIGGER):
            manual_filename = f"manual_checkpoint_ep{i_episode + 1}.pth"
            agent.save_checkpoint(manual_filename, i_episode + 1)
            try:
                os.remove(MANUAL_CHECKPOINT_TRIGGER)
                print(f"\nRemoved trigger file: {MANUAL_CHECKPOINT_TRIGGER}")
            except OSError as e:
                print(f"\nError removing trigger file {MANUAL_CHECKPOINT_TRIGGER}: {e}")

        if (i_episode + 1) % report_interval == 0:
            avg_reward = np.mean(episode_rewards[-report_interval:])
            avg_loss_last_interval = np.mean(episode_losses[-report_interval:])
            elapsed_time = time.time() - start_time
            time_per_episode = elapsed_time / (i_episode + 1 - start_episode)
            est_total_time = time_per_episode * (NUM_EPISODES - start_episode)
            est_remaining_time = est_total_time - elapsed_time
            
            print(f"\rEpisode {i_episode+1:,}/{NUM_EPISODES:,} | Avg Reward ({report_interval} ep): {avg_reward:.3f} | Avg Loss: {avg_loss_last_interval:.5f} | Steps (Total Est): {agent.steps_done + (i_episode+1 - start_episode)*steps:.0f} | Time: {elapsed_time:.1f}s | ETA: {est_remaining_time/3600:.1f} hrs", end="")
        
        if (i_episode + 1) % print_interval == 0:
             print()

    agent.save_model(MODEL_FILENAME)
    print(f"\nTraining finished after {NUM_EPISODES:,} episodes.")

    env.close()
    total_training_time = time.time() - start_time
    print(f"\n\nTraining complete in {total_training_time:.2f} seconds ({total_training_time/3600:.2f} hours).")
    
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        rewards_moving_avg = np.convolve(episode_rewards, np.ones(report_interval)//report_interval, mode='valid')
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
        plt.savefig("dqn_training_plot_50m.png")
        print("Training plot saved to dqn_training_plot_50m.png")
    except ImportError:
        print("Matplotlib not found. Skipping plot generation. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error generating plot: {e}") 