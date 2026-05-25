"""Evaluate or render a saved Hopper-v4 policy.

This script loads a saved .pth file produced by train_reinforce.py
or future Actor-Critic training scripts.

Example usage:
    python evaluate_policy.py --model-path models/reinforce/reinforce_none_1000_seed42_best.pth --episodes 5
    python evaluate_policy.py --model-path models/reinforce/reinforce_none_1000_seed42_best.pth --episodes 3 --render
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from agent import Policy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved Hopper-v4 policy")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved .pth policy weights",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the MuJoCo window",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Evaluation seed",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using deterministic mean actions",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    render_mode = "human" if args.render else "rgb_array"
    env = gym.make("Hopper-v4", render_mode=render_mode)

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = Policy(
        state_space=state_space,
        action_space=action_space,
    ).to(device)

    state_dict = torch.load(
        model_path,
        map_location=device,
    )

    policy.load_state_dict(state_dict)
    policy.eval()

    print("=== Policy evaluation ===")
    print("Model path:", model_path)
    print("State space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Using device:", device)
    print("Episodes:", args.episodes)
    print("Render:", args.render)
    print("Stochastic:", args.stochastic)

    episode_returns = []
    episode_lengths = []

    for episode in range(1, args.episodes + 1):
        state, info = env.reset(seed=args.seed + episode)

        done = False
        episode_return = 0.0
        step_count = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().to(device)
                normal_dist, _ = policy(state_tensor)

                if args.stochastic:
                    action = normal_dist.sample()
                else:
                    action = normal_dist.mean

                action_np = action.detach().cpu().numpy()

            state, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            episode_return += float(reward)
            step_count += 1

            if args.render:
                env.render()

        episode_returns.append(episode_return)
        episode_lengths.append(step_count)

        print(
            f"Episode {episode:03d} | "
            f"return = {episode_return:8.2f} | "
            f"steps = {step_count:04d}"
        )

    env.close()

    returns = np.array(episode_returns, dtype=np.float32)
    lengths = np.array(episode_lengths, dtype=np.float32)

    print("\n=== Evaluation summary ===")
    print(f"Episodes: {args.episodes}")
    print(f"Mean return: {returns.mean():.2f}")
    print(f"Std return:  {returns.std():.2f}")
    print(f"Min return:  {returns.min():.2f}")
    print(f"Max return:  {returns.max():.2f}")
    print(f"Mean steps:  {lengths.mean():.2f}")


if __name__ == "__main__":
    main()