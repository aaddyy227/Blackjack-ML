import torch
import glob
import os

def average_models(checkpoint_dir='checkpoints', output_path='final_model.pth'):
    model_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'worker*_final.pth')))
    if not model_files:
        print(f"No worker checkpoint files found in {checkpoint_dir}")
        return
    avg_state = None
    n = 0
    for fname in model_files:
        print(f"Loading {fname}")
        state_dict = torch.load(fname, map_location='cpu')
        if avg_state is None:
            avg_state = {k: v.clone().float() for k, v in state_dict.items()}
        else:
            for k in avg_state:
                avg_state[k] += state_dict[k].float()
        n += 1
    for k in avg_state:
        avg_state[k] /= n
    torch.save(avg_state, output_path)
    print(f"Averaged {n} models into {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Average DQN model checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory with worker checkpoints')
    parser.add_argument('--output', type=str, default='final_model.pth', help='Output averaged model file')
    args = parser.parse_args()
    average_models(args.checkpoint_dir, args.output) 