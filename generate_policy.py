import gymnasium as gym
import numpy as np
from collections import defaultdict
import pickle
import sys
import time
import multiprocessing
import os # To get CPU count
# NOTE: clear_output is specific to notebooks, we'll just print progress periodically.
# from IPython.display import clear_output 

# --- Helper Functions from Notebook ---

def create_epsilon_greedy_action_policy(env, Q, epsilon):
    """ Create epsilon greedy action policy
    Args:
        env: Environment
        Q: Q table
        epsilon: Probability of selecting random action instead of the 'optimal' action

    Returns:
        Epsilon-greedy-action Policy function with Probabilities of each action for each state
    """
    def policy(obs):
        # Ensure observation is hashable (tuple)
        if not isinstance(obs, tuple):
            obs_tuple = tuple(obs)
        else:
            obs_tuple = obs
        
        # Handle cases where state might not be in Q yet during parallel runs
        if obs_tuple not in Q:
            # Default to random action if state unseen
            return np.ones(env.action_space.n, dtype=float) / env.action_space.n
            
        P = np.ones(env.action_space.n, dtype=float) * epsilon / env.action_space.n
        best_action = np.argmax(Q[obs_tuple])
        P[best_action] += (1.0 - epsilon)
        return P
    return policy

def run_mc_episodes(args):
    """Function to be run by worker processes."""
    worker_id, num_episodes, total_episodes, discount_factor, epsilon = args
    # Each worker needs its own environment instance
    env = gym.make('Blackjack-v1')
    
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following (local to this worker initially)
    pol = create_epsilon_greedy_action_policy(env, Q, epsilon)

    start_time = time.time()
    # Use a smaller reporting interval within workers
    report_interval = max(1, num_episodes // 20) 

    for i in range(1, num_episodes + 1):
        # Optional: Progress within worker (can be verbose)
        # if i % report_interval == 0:
        #     elapsed = time.time() - start_time
        #     print(f"\rWorker {worker_id}: Episode {i}/{num_episodes} ({elapsed:.1f}s)...", end="")

        # Generate an episode.
        episode = []
        state, _ = env.reset()
        
        if not isinstance(state, tuple):
            state_tuple = tuple(state)
        else:
            state_tuple = state

        for t in range(100): # Limit episode length
            # Need to update policy based on current worker Q
            current_pol = create_epsilon_greedy_action_policy(env, Q, epsilon)
            probs = current_pol(state_tuple)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            if not isinstance(next_state, tuple):
                next_state_tuple = tuple(next_state)
            else:
                next_state_tuple = next_state
                
            episode.append((state_tuple, action, reward))
            
            done = terminated or truncated
            if done:
                break
                
            state_tuple = next_state_tuple

        # --- First Visit MC Update --- 
        sa_in_episode = set([(x[0], x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            first_occurence_idx = next(i for i, x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            G = sum([x[2]*(discount_factor**j) for j, x in enumerate(episode[first_occurence_idx:])])
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            # Update local Q table for this worker
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
            
    env.close()
    # print(f"\nWorker {worker_id} finished {num_episodes} episodes.")
    return Q, returns_sum, returns_count

# --- Main Execution ---
if __name__ == "__main__":
    # Freeze support needed for multiprocessing on Windows
    multiprocessing.freeze_support() 

    POLICY_FILENAME = "blackjack_policy_500m.pkl" # New filename
    TOTAL_EPISODES = 500_000_000 # 500 Million
    DISCOUNT_FACTOR = 0.9
    EPSILON = 0.05
    
    # Determine number of workers
    num_workers = os.cpu_count()
    if num_workers is None or num_workers < 1:
        num_workers = 2 # Default if detection fails
    print(f"Using {num_workers} worker processes.")

    episodes_per_worker = TOTAL_EPISODES // num_workers
    remaining_episodes = TOTAL_EPISODES % num_workers # Distribute remainder
    
    tasks = []
    for i in range(num_workers):
        worker_episodes = episodes_per_worker + (1 if i < remaining_episodes else 0)
        if worker_episodes > 0:
            tasks.append((i, worker_episodes, TOTAL_EPISODES, DISCOUNT_FACTOR, EPSILON))
        
    print(f"Starting On-Policy MC training for {TOTAL_EPISODES} episodes across {len(tasks)} workers...")
    start_time = time.time()

    # Use multiprocessing Pool
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(run_mc_episodes, tasks)

    total_duration = time.time() - start_time
    print(f"\nParallel training finished in {total_duration:.2f} seconds ({total_duration/60:.2f} minutes).")

    # --- Merge Results --- 
    print("Merging results from workers...")
    final_Q = defaultdict(lambda: np.zeros(2)) # Assuming 2 actions (stand, hit)
    final_returns_sum = defaultdict(float)
    final_returns_count = defaultdict(float)

    for q_worker, rs_worker, rc_worker in results:
        for sa_pair, sum_val in rs_worker.items():
            final_returns_sum[sa_pair] += sum_val
        for sa_pair, count_val in rc_worker.items():
            final_returns_count[sa_pair] += count_val

    # Calculate final Q-values from merged sums and counts
    num_states_updated = 0
    for sa_pair, total_sum in final_returns_sum.items():
        state, action = sa_pair
        total_count = final_returns_count[sa_pair]
        if total_count > 0:
            final_Q[state][action] = total_sum / total_count
            num_states_updated += 1
            
    print(f"Merging complete. Final policy has Q-values for {len(final_Q)} states ({num_states_updated} state-action pairs updated)." )       

    # --- Save the final Q-table --- 
    print(f"Saving final policy to {POLICY_FILENAME}...")
    try:
        # Convert defaultdicts before saving if needed, but main Q should be saved
        with open(POLICY_FILENAME, 'wb') as f:
            pickle.dump(dict(final_Q), f) 
        print(f"Policy Q-table saved successfully.")
    except Exception as e:
        print(f"Error saving policy: {e}") 