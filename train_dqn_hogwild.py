import torch
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
import os
import multiprocessing as mp
from collections import deque, namedtuple

# --- Configuration ---
ENV_NAME = 'Blackjack-v1'
TOTAL_EPISODES = 500_000          # Total episodes across all GPUs
BATCH_SIZE = 512                  # Per-GPU batch size
GAMMA = 0.99                      # Discount factor
EPS_START = 1.0                   # Starting epsilon for exploration
EPS_END = 0.05                    # Minimum epsilon
EPS_DECAY = 1_500_000             # Epsilon decay (global steps)
TAU = 0.005                       # Soft update coefficient
LR = 1e-4                         # Learning rate
BUFFER_SIZE = 500_000             # Replay buffer capacity
CHECKPOINT_INTERVAL = 100_000     # Local episodes between automatic checkpoints
MANUAL_TRIGGER = 'save.trigger'   # Trigger file name for manual checkpoint

Transition = namedtuple('Transition', ('state','action','next_state','reward','done'))
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        size = min(batch_size, len(self.memory))
        return random.sample(self.memory, size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, out_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Worker function: trains on one GPU independently
def worker(rank, world_size):
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    # Episodes per GPU
    episodes_per_worker = TOTAL_EPISODES // world_size
    # Set up
    env = gym.make(ENV_NAME)
    buffer = ReplayBuffer(BUFFER_SIZE)
    n_obs = 3
    n_act = env.action_space.n
    policy = DQN(n_obs, n_act).to(device)
    target = DQN(n_obs, n_act).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = optim.AdamW(policy.parameters(), lr=LR)
    steps_done = 0

    def select_action(state):
        nonlocal steps_done
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if random.random() > eps:
            with torch.no_grad():
                return policy(state).max(1)[1].view(1,1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    def optimize():
        if len(buffer) < BATCH_SIZE:
            return
        trans = buffer.sample(BATCH_SIZE)
        batch = Transition(*zip(*trans))
        state_b = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
        action_b = torch.cat(batch.action)
        reward_b = torch.tensor(batch.reward, dtype=torch.float32, device=device)
        mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
        nxt_states = torch.tensor([s for s in batch.next_state if s is not None], dtype=torch.float32, device=device)

        q_vals = policy(state_b).gather(1, action_b)
        next_q = torch.zeros(BATCH_SIZE, device=device)
        if nxt_states.size(0) > 0:
            with torch.no_grad():
                next_q[mask] = target(nxt_states).max(1)[0]
        expected = reward_b + GAMMA * next_q

        loss = F.smooth_l1_loss(q_vals.squeeze(), expected)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(policy.parameters(), 100)
        optimizer.step()
        return loss.item()

    # Prepare checkpoint dir
    os.makedirs('checkpoints', exist_ok=True)

    for ep in range(1, episodes_per_worker + 1):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        total_reward = 0
        while not done:
            action = select_action(state)
            nxt, r, ter, tru, _ = env.step(action.item())
            done_flag = ter or tru
            buffer.push(state.cpu().numpy().squeeze(), action, nxt if not done_flag else None, r, done_flag)
            state = torch.tensor(nxt, dtype=torch.float32, device=device).unsqueeze(0)
            optimize()
            total_reward += r
            done = done_flag
        # Soft update target network after each episode
        for param, target_param in zip(policy.parameters(), target.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0-TAU) * target_param.data)

        # Automatic checkpoint
        if ep % CHECKPOINT_INTERVAL == 0:
            path = f'checkpoints/worker{rank}_ep{ep}.pth'
            torch.save(policy.state_dict(), path)
            if rank == 0:
                print(f"Worker {rank}: saved checkpoint at episode {ep}")

        # Manual trigger
        if os.path.exists(MANUAL_TRIGGER):
            path = f'checkpoints/worker{rank}_manual_ep{ep}.pth'
            torch.save(policy.state_dict(), path)
            os.remove(MANUAL_TRIGGER)
            print(f"Worker {rank}: manual checkpoint at episode {ep}")

        # Logging
        if rank == 0 and ep % (CHECKPOINT_INTERVAL//2) == 0:
            print(f"[Worker {rank}] Episode {ep}/{episodes_per_worker}, reward {total_reward:.2f} - steps {steps_done}")

    # Final model save
    final_path = f'checkpoints/worker{rank}_final.pth'
    torch.save(policy.state_dict(), final_path)
    print(f"Worker {rank} done. Final model saved.")

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f"Launching {world_size} parallel workers (Hogwild style)...")
    mp.spawn(worker=worker, args=(world_size,), nprocs=world_size, join=True)
    print("All workers have completed training.") 