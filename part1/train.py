"""Train a control policy on the Hopper-v4 environment.

Current stage:
- REINFORCE without baseline
- Training controlled from the command line

Example usage:
    python train.py --episodes 10 --render
    python train.py --episodes 100
    python train.py --episodes 500 --seed 42
"""

import argparse
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch

from agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser(description="Train REINFORCE on Hopper-v4")

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of training episodes",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment with a MuJoCo window",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    render_mode = "human" if args.render else "rgb_array"
    env = gym.make("Hopper-v4", render_mode=render_mode)

    print("State space:", env.observation_space)
    print("Action space:", env.action_space)

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Episodes:", args.episodes)
    print("Render:", args.render)
    print("Seed:", args.seed)

    policy = Policy(state_space=state_space, action_space=action_space)
    agent = Agent(policy=policy, device=device)

    recent_returns = deque(maxlen=20)

    start_time = time.time()

    for episode in range(1, args.episodes + 1):
        state, info = env.reset(seed=args.seed + episode)

        done = False
        episode_reward = 0.0
        step_count = 0

        while not done:
            action, action_log_prob = agent.get_action(state)
            action_np = action.detach().cpu().numpy()

            next_state, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            agent.store_outcome(
                state=state,
                next_state=next_state,
                action_log_prob=action_log_prob,
                reward=reward,
                done=done,
            )

            state = next_state
            episode_reward += reward
            step_count += 1

            if args.render:
                env.render()

        loss = agent.update_policy()

        recent_returns.append(episode_reward)
        moving_avg_return = np.mean(recent_returns)

        print(
            f"Episode {episode:04d} | "
            f"steps = {step_count:04d} | "
            f"return = {episode_reward:8.2f} | "
            f"moving_avg_return = {moving_avg_return:8.2f} | "
            f"loss = {loss:8.4f}"
        )

    total_time = time.time() - start_time

    print("\nTraining finished.")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Average time per episode: {total_time / args.episodes:.2f} seconds")

    env.close()


if __name__ == "__main__":
    main()