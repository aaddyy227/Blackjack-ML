import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
import os
from collections import deque, namedtuple

# --- Configuration ---
ENV_NAME = 'Blackjack-v1'
NUM_EPISODES = 500_000          # Total episodes to train
PER_GPU_BATCH = 256             # Batch size per GPU
GAMMA = 0.99                    # Discount factor
EPS_START = 1.0                 # Starting epsilon
EPS_END = 0.05                  # Minimum epsilon
EPS_DECAY = 2_000_000           # Epsilon decay
TAU = 0.005                     # Soft update coeff
LR = 1e-4                       # Learning rate
BUFFER_SIZE = 500_000           # Replay buffer capacity
CHECKPOINT_INTERVAL = 1_000_000 # Episodes between automatic checkpoints
MANUAL_TRIGGER = 'save_trigger.trigger'
MODEL_FILENAME = 'dp_dqn_policy.pth'
CHECKPOINT_FILE = 'dp_dqn_checkpoint.pth'

# --- Replay buffer ---
Transition = namedtuple('Transition', ('state','action','next_state','reward','done'))
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        size = min(batch_size, len(self.memory))
        return random.sample(self.memory, size)
    def __len__(self): return len(self.memory)

# --- Q-network ---
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

if __name__ == '__main__':
    # Setup device and DataParallel
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPUs are required')
    ngpus = torch.cuda.device_count()
    device = torch.device('cuda')
    global_batch = PER_GPU_BATCH * ngpus

    # Initialize model
    net = DQN(3, gym.make(ENV_NAME).action_space.n).to(device)
    net = nn.DataParallel(net)
    target = DQN(3, gym.make(ENV_NAME).action_space.n).to(device)
    target = nn.DataParallel(target)
    target.load_state_dict(net.state_dict())

    optimizer = optim.AdamW(net.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(enabled=True)
    buffer = ReplayBuffer(BUFFER_SIZE)

    steps_done = 0

    # Optional resume from checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        ckpt = torch.load(CHECKPOINT_FILE, map_location='cpu')
        net.load_state_dict(ckpt['net'])
        target.load_state_dict(ckpt['target'])
        optimizer.load_state_dict(ckpt['opt'])
        steps_done = ckpt.get('steps_done', 0)
        print(f"Resumed from checkpoint at step {steps_done}")

    start_time = time.time()

    for episode in range(1, NUM_EPISODES+1):
        state, _ = gym.make(ENV_NAME).reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy
            eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            if random.random() > eps:
                with torch.no_grad():
                    action = net(state).max(1)[1].view(1,1)
            else:
                action = torch.tensor([[gym.make(ENV_NAME).action_space.sample()]], device=device)

            nxt, r, ter, tru, _ = gym.make(ENV_NAME).step(action.item())
            done_flag = ter or tru
            buffer.push(state.cpu().numpy().squeeze(), action, nxt if not done_flag else None, r, done_flag)
            state = torch.tensor(nxt, dtype=torch.float32, device=device).unsqueeze(0)
            total_reward += r

            # Optimize
            if len(buffer) >= global_batch:
                batch = buffer.sample(global_batch)
                batch = Transition(*zip(*batch))
                sb = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
                ab = torch.cat(batch.action)
                rb = torch.tensor(batch.reward, dtype=torch.float32, device=device)
                mask = torch.tensor([s is not None for s in batch.next_state], device=device)
                ns = torch.tensor([s for s in batch.next_state if s is not None], dtype=torch.float32, device=device)

                optimizer.zero_grad()
                with torch.amp.autocast():
                    qv = net(sb).gather(1, ab)
                    nq = torch.zeros(global_batch, device=device)
                    if ns.size(0) > 0:
                        with torch.no_grad():
                            nq[mask] = target(ns).max(1)[0]
                    exp_q = rb + GAMMA * nq
                    loss = F.smooth_l1_loss(qv.squeeze(), exp_q)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_value_(net.parameters(), 100)
                scaler.step(optimizer)
                scaler.update()

        # Soft update target
        for p, tp in zip(net.module.parameters(), target.module.parameters()):
            tp.data.copy_(TAU * p.data + (1-TAU) * tp.data)

        # Automatic checkpoint
        if episode % CHECKPOINT_INTERVAL == 0:
            torch.save({'net':net.state_dict(), 'target':target.state_dict(), 'opt':optimizer.state_dict(), 'steps_done':steps_done}, CHECKPOINT_FILE)
            print(f"Saved checkpoint at episode {episode}")

        # Manual trigger
        if os.path.exists(MANUAL_TRIGGER):
            torch.save({'net':net.state_dict(), 'target':target.state_dict(), 'opt':optimizer.state_dict(), 'steps_done':steps_done}, f'manual_ep{episode}.pth')
            os.remove(MANUAL_TRIGGER)
            print(f"Manual checkpoint at episode {episode}")

        # Logging
        if episode % REPORT_INTERVAL == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode}/{NUM_EPISODES} | Reward {total_reward:.2f} | Steps {steps_done} | Time {elapsed:.1f}s")

    # Final save
    torch.save(net.state_dict(), MODEL_FILENAME)
    print("Training complete.") 