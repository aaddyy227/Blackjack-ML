import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
import os
import socket
from collections import deque, namedtuple

# --- Configuration ---
ENV_NAME = 'Blackjack-v1'
NUM_EPISODES = 500_000            # Total episodes to train
BATCH_SIZE = 512                 # Batch size per GPU
GAMMA = 0.99                      # Discount factor
EPS_START = 1.0                  # Starting epsilon for exploration
EPS_END = 0.05                   # Ending epsilon
EPS_DECAY = 2_000_000            # Epsilon decay rate (per action)
TAU = 0.005                       # Soft update parameter for target network
LR = 1e-4                        # Learning rate
BUFFER_SIZE = 500_000            # Replay buffer capacity
SYNC_EPISODES = 10_000           # Episodes between weight averaging
REPORT_INTERVAL = 10_000         # Episodes between logging
MODEL_FILENAME = 'async_dqn_policy.pth'
CHECKPOINT_FILE = 'async_dqn_checkpoint.pth'
MANUAL_TRIGGER = 'async_save.trigger'

# --- Replay Buffer ---
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
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

# --- Q-Network ---
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

# --- Utility to find free port ---
def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    port = s.getsockname()[1]
    s.close()
    return port

# --- Worker Function ---
def main_worker(rank, world_size, port):
    # Setup process group for NCCL
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Create environment and buffer
    env = gym.make(ENV_NAME)
    buffer = ReplayBuffer(BUFFER_SIZE)

    # Initialize networks
    n_obs = 3
    n_act = env.action_space.n
    policy = DQN(n_obs, n_act).to(device)
    target = DQN(n_obs, n_act).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = optim.AdamW(policy.parameters(), lr=LR)

    steps_done = 0
    episode_count = 0

    # Epsilon-greedy action selection
    def select_action(state_tensor):
        nonlocal steps_done
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if random.random() > eps:
            with torch.no_grad():
                return policy(state_tensor).max(1)[1].view(1,1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    # Single optimization step
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
                next_q[mask] = target(nxt_states).max(1)[0].detach()
        expected = reward_b + GAMMA * next_q
        loss = F.smooth_l1_loss(q_vals.squeeze(), expected)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(policy.parameters(), 100)
        optimizer.step()

    # Optionally load existing checkpoint
    if os.path.exists(CHECKPOINT_FILE) and rank == 0:
        ckpt = torch.load(CHECKPOINT_FILE, map_location='cpu')
        policy.load_state_dict(ckpt['policy'])
        target.load_state_dict(ckpt['target'])
    dist.barrier()

    # Training loop
    while episode_count < NUM_EPISODES:
        state, _ = env.reset()
        st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False

        while not done:
            action = select_action(st)
            nxt, r, ter, tru, _ = env.step(action.item())
            done_flag = ter or tru
            buffer.push(state, action, nxt if not done_flag else None, r, done_flag)
            st = torch.tensor(nxt, dtype=torch.float32, device=device).unsqueeze(0)
            optimize()
            state = nxt
            done = done_flag

        episode_count += 1

        # Periodic weight averaging across GPUs
        if episode_count % SYNC_EPISODES == 0:
            for p in policy.state_dict().values():
                dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
                p.data /= world_size
            target.load_state_dict(policy.state_dict())
            dist.barrier()
            if rank == 0:
                torch.save({'policy':policy.state_dict(), 'target':target.state_dict()}, CHECKPOINT_FILE)

        # Manual checkpoint trigger
        if rank == 0 and os.path.exists(MANUAL_TRIGGER):
            torch.save({'policy':policy.state_dict(), 'target':target.state_dict()}, f'manual_ep{episode_count}.pth')
            os.remove(MANUAL_TRIGGER)

        # Logging (rank 0)
        if rank == 0 and episode_count % REPORT_INTERVAL == 0:
            print(f"[EP {episode_count}/{NUM_EPISODES}] - Steps done: {steps_done}")

    # Final save by rank 0
    if rank == 0:
        torch.save({'policy':policy.state_dict(), 'target':target.state_dict()}, MODEL_FILENAME)
        print("Training complete.")

    dist.destroy_process_group()

# --- Entry Point ---
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    port = find_free_port()
    print(f"Found {world_size} GPUs; spawning workers on port {port}...")
    mp.spawn(main_worker, args=(world_size, port), nprocs=world_size, join=True)
    print("All workers finished.") 