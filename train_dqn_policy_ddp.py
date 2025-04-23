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
import argparse # For potential future command-line args
import socket # Import socket

# --- DDP Imports ---
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# --- Configuration ---
# Moved some configs inside main_worker or passed as args
# These remain global or default arguments
ENV_NAME = 'Blackjack-v1'
DEFAULT_MODEL_FILENAME = "blackjack_dqn_policy_ddp_50m.pth"
DEFAULT_CHECKPOINT_FILE = "dqn_checkpoint_ddp.pth"
MANUAL_CHECKPOINT_TRIGGER = "save_checkpoint_ddp.trigger" # Separate trigger
NUM_EPISODES = 500_000 # Per process, effectively? Or total? Let's aim for total.
BATCH_SIZE = 512 # Per GPU batch size
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
# << Adjusted EPS_DECAY for multi-GPU experience rate >>
# Original was 2_000_000. Divide by estimated GPU count (needs world_size later)
# Let's define it lower globally, or adjust dynamically in main_worker
BASE_EPS_DECAY = 2_000_000 
TAU = 0.005
LR = 1e-4 # Learning Rate
BUFFER_SIZE = 500_000 # Per process buffer size
CHECKPOINT_INTERVAL = 10_000 # Total episodes checkpoint interval
REPORT_INTERVAL = 1000 # Logging interval (rank 0 only)
PRINT_INTERVAL = 10000 # Newline interval (rank 0 only)
# << Gradient Accumulation >>
GRADIENT_ACCUMULATION_STEPS = 8

# --- Helper function to find free port ---
def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("localhost", 0)) # Bind to port 0 lets OS choose a free port
    port = sock.getsockname()[1]
    sock.close()
    return port

# --- DDP Setup/Cleanup ---
def setup(rank, world_size, port): # Accept port argument
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # Use NCCL backend for Linux/NVIDIA
    backend = 'nccl'
    if not torch.distributed.is_nccl_available():
        print(f"Warning: NCCL backend not available, falling back to gloo (Performance may suffer)")
        backend = 'gloo'

    print(f"Rank {rank} initializing process group with backend: {backend}")
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == 'nccl': # Only pin device for NCCL
        torch.cuda.set_device(rank) # Pin the process to a specific GPU

def cleanup():
    dist.destroy_process_group()

# --- Define the Q-Network (Identical to previous) ---
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
             # Move tensor creation logic to where state is used
             # x = torch.FloatTensor(x) # Handled in agent/training loop
             pass # Input should already be a tensor here
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# --- Replay Buffer (Identical) ---
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # Ensure we don't sample more than available
        actual_batch_size = min(batch_size, len(self.memory))
        if actual_batch_size == 0:
            return [] # Return empty list if buffer is empty
        return random.sample(self.memory, actual_batch_size)


    def __len__(self):
        return len(self.memory)

# --- DQN Agent (Modified for Gradient Accumulation) ---
class DQNAgent:
    # Removed env dependency from init, pass device=rank
    def __init__(self, n_observations, n_actions, rank, buffer_capacity=BUFFER_SIZE, lr=LR):
        self.rank = rank # GPU rank / device ID
        self.n_actions = n_actions
        self.n_observations = n_observations
        
        # Networks initialized on the specific GPU for this process
        self.policy_net = DQN(self.n_observations, self.n_actions).to(rank)
        # Wrap policy_net with DDP *after* initialization and potential checkpoint loading
        self.target_net = DQN(self.n_observations, self.n_actions).to(rank)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayBuffer(buffer_capacity)
        self.steps_done = 0 # Track steps per process for epsilon decay
        # << Use new torch.amp API >>
        self.scaler = torch.amp.GradScaler('cuda')
        # << Counter for gradient accumulation >>
        self.accum_step_counter = 0

    # Wrap policy net with DDP after potential loading
    def wrap_policy_net_ddp(self):
         # Find_unused_parameters might be needed if parts of the network aren't used in forward pass
         self.policy_net = DDP(self.policy_net, device_ids=[self.rank], find_unused_parameters=False)


    def select_action(self, state, env_action_space, eps_threshold): # Accept eps_threshold
        sample = random.random()
        # Epsilon decay logic is now handled *outside* this function
        # self.steps_done is still incremented outside for tracking
        
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.rank).unsqueeze(0)
        
        if sample > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor) 
                return q_values.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[env_action_space.sample()]], device=self.rank, dtype=torch.long)

    def optimize_model(self, batch_size=BATCH_SIZE, gamma=GAMMA, grad_accum_steps=GRADIENT_ACCUMULATION_STEPS):
        if len(self.memory) < batch_size:
            return None, False # Return loss, and bool indicating if optimizer stepped
            
        transitions = self.memory.sample(batch_size)
        if not transitions:
             return None, False
             
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                      device=self.rank, dtype=torch.bool)
        
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.rank)
        action_batch = torch.cat(batch.action) 
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.rank)
        
        optimizer_stepped = False
        with torch.amp.autocast(device_type='cuda'):
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)
            next_state_values = torch.zeros(len(transitions), device=self.rank)
            if len(non_final_next_states_list) > 0:
                 non_final_next_states = torch.tensor(np.array(non_final_next_states_list), 
                                                   dtype=torch.float32, device=self.rank)
                 with torch.no_grad():
                      next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].float()
            
            expected_state_action_values = (next_state_values * gamma) + reward_batch
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values.float(), expected_state_action_values.unsqueeze(1))
            
            # << Scale loss for accumulation >>
            loss = loss / grad_accum_steps

        # Accumulate scaled gradients
        self.scaler.scale(loss).backward() 
        self.accum_step_counter += 1

        # Perform optimizer step only after accumulating gradients for N steps
        if self.accum_step_counter % grad_accum_steps == 0:
            # Unscale gradients before clipping (optional but can be safer)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) 
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad() # Zero gradients ONLY after optimizer step
            optimizer_stepped = True
            # Reset counter after step (optional, modulus handles it)
            # self.accum_step_counter = 0 
        
        return loss.item() * grad_accum_steps, optimizer_stepped # Return unscaled loss, and if step occurred

    def update_target_network(self, tau=TAU):
        target_net_state_dict = self.target_net.state_dict()
        # Access the underlying model's state dict when wrapped with DDP
        policy_net_state_dict = self.policy_net.module.state_dict() 
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
        self.target_net.load_state_dict(target_net_state_dict)
        
    # Checkpoint saving/loading modified for DDP + AMP
    def save_checkpoint(self, filename, episode):
        if self.rank != 0:
            return
            
        checkpoint = {
            'episode': episode,
            'policy_net_state_dict': self.policy_net.module.state_dict(), 
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            # Save GradScaler state
            'scaler_state_dict': self.scaler.state_dict() 
        }
        try:
            torch.save(checkpoint, filename)
            print(f"\nCheckpoint saved to {filename} at episode {episode:,} (Rank 0)")
        except Exception as e:
            print(f"\nRank 0 Error saving checkpoint: {e}")

    def load_checkpoint(self, filename):
        if not os.path.exists(filename):
             if self.rank == 0: # Print only once
                 print(f"Checkpoint file {filename} not found. Starting from scratch.")
             return 0, 0 # Start episode 0, steps_done 0
        
        try:
            # Load checkpoint onto the CPU first to avoid GPU mismatches
            checkpoint = torch.load(filename, map_location='cpu')
            
            # Load state dict into the underlying model before wrapping with DDP
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            
            # Move models to the correct device for this rank
            self.policy_net.to(self.rank)
            self.target_net.to(self.rank)
            
            # Load optimizer state AFTER moving models to the correct device
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Manually move optimizer states to the correct device 
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.rank)

            start_episode = checkpoint.get('episode', 0)
            steps_done = checkpoint.get('steps_done', 0)
            
            # Load GradScaler state
            if 'scaler_state_dict' in checkpoint:
                 self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                 if self.rank == 0:
                      print("Loaded GradScaler state.")
            else:
                 if self.rank == 0:
                      print("GradScaler state not found in checkpoint, initializing new scaler.")
            
            if self.rank == 0:
                print(f"Checkpoint loaded from {filename}. Resuming from episode {start_episode + 1:,}")
            # Barrier to ensure all processes loaded before proceeding
            dist.barrier() 
            return start_episode, steps_done
            
        except Exception as e:
            if self.rank == 0:
                 print(f"Error loading checkpoint: {e}. Starting from scratch.")
            dist.barrier() # Ensure all processes hit the error path together
            return 0, 0


# --- Main Worker Function for DDP ---
def main_worker(rank, world_size, port, model_filename, checkpoint_file): # Accept port argument
    setup(rank, world_size, port) # Pass port to setup
    if rank == 0:
        print(f"=== Starting DDP Training with {world_size} GPUs on Port {port} ===") # Log the port used
        print(f"Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")

    # Each process gets its own environment instance
    # Seed environment differently per process? Might help exploration.
    env = gym.make(ENV_NAME)
    # seed = int(time.time()) + rank # Simple seeding
    # env.reset(seed=seed)
    # env.action_space.seed(seed)

    n_observations = 3 # Specific to Blackjack state representation
    n_actions = env.action_space.n
    
    # << Adjust Epsilon Decay based on world size >>
    effective_eps_decay = BASE_EPS_DECAY / world_size
    if rank == 0:
        print(f"Base EPS_DECAY: {BASE_EPS_DECAY}, Effective EPS_DECAY: {effective_eps_decay:.0f}")

    agent = DQNAgent(n_observations, n_actions, rank)

    # Load checkpoint *before* wrapping policy_net with DDP
    start_episode, start_steps_done = agent.load_checkpoint(checkpoint_file)
    agent.steps_done = start_steps_done # Resume steps_done for epsilon decay

    # Now wrap the policy network for DDP
    agent.wrap_policy_net_ddp()

    episode_rewards = [] # Track rewards per process
    episode_losses = [] # Track losses per process
    total_optimizer_steps = 0 # Track actual optimizer steps

    if rank == 0: # Only rank 0 tracks global time and episode counts for logging
        start_time = time.time()
        global_episode_count = start_episode # Track total episodes completed across processes

    # Calculate episodes per process (approximate)
    # Aim for total NUM_EPISODES across all processes
    episodes_to_run_total = NUM_EPISODES - start_episode
    # Each process runs a fraction of the remaining episodes
    # This isn't perfectly synchronized, but gives a target
    episodes_per_process = episodes_to_run_total // world_size 
    if rank < (episodes_to_run_total % world_size): # Distribute remainder
         episodes_per_process += 1
         
    local_start_episode = 0 # Each process counts its own episodes from 0

    # --- Training Loop (per process) ---
    # Loop for a number of episodes determined for this process
    for i_local_episode in range(episodes_per_process): 
        current_global_episode = start_episode + (rank * (episodes_to_run_total // world_size)) + min(rank, episodes_to_run_total % world_size) + i_local_episode +1 # Estimate global episode
        
        state, _ = env.reset()
        done = False
        total_reward = 0
        total_loss_in_episode = 0
        batches_in_episode = 0
        optimizer_steps_in_episode = 0
        
        while not done:
            # << Pass effective_eps_decay to select_action >>
            # Needs modification in DQNAgent.select_action to accept this
            # OR: Calculate eps_threshold here directly using agent.steps_done and effective_eps_decay
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                 np.exp(-1. * agent.steps_done / effective_eps_decay)
            agent.steps_done += 1 # Still increment per action/experience
            
            action_tensor = agent.select_action(state, env.action_space, eps_threshold) # Modify select_action
            action = action_tensor.item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            agent.memory.push(state, action_tensor, next_state if not done else None, reward, done)
            state = next_state
            
            # Perform optimization step (or gradient accumulation)
            loss, optimizer_stepped = agent.optimize_model()
            
            if loss is not None:
                 total_loss_in_episode += loss
                 batches_in_episode += 1

            # << Update target network only when optimizer steps >>
            if optimizer_stepped:
                 agent.update_target_network()
                 optimizer_steps_in_episode += 1

            if done:
                break
        
        total_optimizer_steps += optimizer_steps_in_episode
        
        # --- Logging and Checkpointing (Rank 0 only) ---
        # Processes need to communicate results (e.g., avg reward/loss) to Rank 0 for accurate global logging
        # Simple approach: Rank 0 logs its own progress as an indicator
        if rank == 0:
            episode_rewards.append(total_reward)
            avg_loss_in_episode = total_loss_in_episode / batches_in_episode if batches_in_episode > 0 else 0
            episode_losses.append(avg_loss_in_episode)
            
            # Use estimated global episode for reporting and checkpointing
            estimated_total_episodes_done = current_global_episode # Use the estimated global count

            # Automatic Checkpoint (based on estimated total episodes)
            # Check if the *interval boundary* was crossed in this batch of episodes reported by rank 0
            if (estimated_total_episodes_done // CHECKPOINT_INTERVAL) > ((estimated_total_episodes_done - 1) // CHECKPOINT_INTERVAL):
                 agent.save_checkpoint(checkpoint_file, estimated_total_episodes_done)

            # Manual Checkpoint Trigger (Rank 0 checks)
            if os.path.exists(MANUAL_CHECKPOINT_TRIGGER):
                manual_filename = f"manual_checkpoint_ddp_ep{estimated_total_episodes_done}.pth"
                agent.save_checkpoint(manual_filename, estimated_total_episodes_done)
                try:
                    os.remove(MANUAL_CHECKPOINT_TRIGGER)
                    print(f"\nRank 0 removed trigger file: {MANUAL_CHECKPOINT_TRIGGER}")
                except OSError as e:
                    print(f"\nRank 0 Error removing trigger file {MANUAL_CHECKPOINT_TRIGGER}: {e}")


            # Logging Progress
            if estimated_total_episodes_done % REPORT_INTERVAL == 0 and len(episode_rewards) >= REPORT_INTERVAL:
                avg_reward = np.mean(episode_rewards[-REPORT_INTERVAL:])
                avg_loss_last_interval = np.mean(episode_losses[-REPORT_INTERVAL:])
                elapsed_time = time.time() - start_time
                
                # Estimate remaining time based on rank 0's progress
                time_per_episode_rank0 = elapsed_time / (estimated_total_episodes_done - start_episode + 1) # +1 to avoid div by zero
                est_total_time = time_per_episode_rank0 * (NUM_EPISODES - start_episode)
                est_remaining_time = est_total_time - elapsed_time

                # Add optimizer steps to log
                print(f"\rEp {estimated_total_episodes_done:,}/{NUM_EPISODES:,} | Avg Rwd (Rank 0): {avg_reward:.3f} | Avg Loss (Rank 0): {avg_loss_last_interval:.5f} | Opt Steps(R0): {total_optimizer_steps:,} | Time: {elapsed_time:.1f}s | ETA: {est_remaining_time/3600:.1f} hrs", end="")
            
            if estimated_total_episodes_done % PRINT_INTERVAL == 0:
                 print()
                 
        # Barrier to somewhat synchronize processes, especially before potential checkpointing
        # However, true synchronization requires more complex episode counting or communication
        # dist.barrier() # Optional: might slow down training if processes are uneven

    # --- End of Training (per process) ---
    if rank == 0:
        print(f"\nRank 0 finished its episodes.")
        # Final save by rank 0
        print("Saving final model...")
        # Need to get the state dict from the DDP-wrapped model
        final_policy_state = agent.policy_net.module.state_dict() 
        try:
            torch.save(final_policy_state, model_filename)
            print(f"Final DDP model saved successfully to {model_filename}")
        except Exception as e:
            print(f"Rank 0 Error saving final DDP model: {e}")

        # Optional: Plotting (only Rank 0 has meaningful reward/loss history)
        if episode_rewards and episode_losses:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 5))
                # Simple moving average calculation
                def moving_average(data, window_size):
                    if not data or window_size <= 0: return []
                    window = np.ones(int(window_size))/float(window_size)
                    return np.convolve(data, window, 'valid')

                # Plotting needs adjustment based on how many episodes rank 0 actually ran
                rewards_moving_avg = moving_average(episode_rewards, REPORT_INTERVAL)
                losses_moving_avg = moving_average(episode_losses, REPORT_INTERVAL)

                plt.subplot(1, 2, 1)
                if list(rewards_moving_avg): plt.plot(rewards_moving_avg)
                plt.title(f'Episode Rewards (Rank 0, MA {REPORT_INTERVAL})')
                plt.xlabel('Episode Window')
                plt.ylabel('Average Reward')
                plt.grid(True)
                
                plt.subplot(1, 2, 2)
                if list(losses_moving_avg): plt.plot(losses_moving_avg)
                plt.title(f'Training Loss (Rank 0, MA {REPORT_INTERVAL})')
                plt.xlabel('Episode Window')
                plt.ylabel('Average Loss')
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig("dqn_training_plot_ddp_50m.png")
                print("Rank 0 Training plot saved to dqn_training_plot_ddp_50m.png")
            except ImportError:
                print("Matplotlib not found on Rank 0. Skipping plot generation.")
            except Exception as e:
                print(f"Rank 0 Error generating plot: {e}")
                
    env.close()
    cleanup() # Destroy the process group for this worker

# --- Main Execution ---
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size == 0:
         print("No CUDA GPUs found. This script requires GPUs for DDP.")
         exit()
         
    # Find a free port before spawning
    free_port = find_free_port()
    print(f"Found {world_size} GPUs. Spawning DDP processes on port {free_port}...")
    
    # Pass the free port and other arguments to the worker function
    args = (world_size, free_port, DEFAULT_MODEL_FILENAME, DEFAULT_CHECKPOINT_FILE)
    mp.spawn(main_worker,
             args=args,
             nprocs=world_size,
             join=True)

    print("\n DDP Training script finished.") 