"""Train Actor-Critic agents on Hopper-v4.

This script is dedicated only to Actor-Critic experiments.

Supported variants:
- Actor-Critic with one-step TD target
- Actor-Critic with Monte Carlo return target
- Optional advantage normalization
- Critic loss coefficient
- Entropy regularization

Example usage:
    python train_actor_critic.py --episodes 100 --ac-target-type td --run-name ac_td_100
    python train_actor_critic.py --episodes 100 --ac-target-type mc --normalize-advantages --run-name ac_mc_norm_100
    python train_actor_critic.py --episodes 1000 --ac-target-type mc --normalize-advantages --save-best-model --run-name ac_mc_norm_1000_seed42
"""

import argparse
import csv
import random
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser(description="Train Actor-Critic on Hopper-v4")

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of training episodes",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render Hopper during training",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--ac-target-type",
        type=str,
        default="td",
        choices=["td", "mc"],
        help="Actor-Critic target type: td or mc",
    )

    parser.add_argument(
        "--normalize-advantages",
        action="store_true",
        help="Normalize advantages before actor update",
    )

    parser.add_argument(
        "--critic-loss-coef",
        type=float,
        default=0.5,
        help="Weight of critic loss",
    )

    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Entropy bonus coefficient",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Base directory for CSV outputs",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Base directory for saved models",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="actor_critic_run",
        help="Experiment name",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save final policy weights",
    )

    parser.add_argument(
        "--save-best-model",
        action="store_true",
        help="Save best policy according to moving average return",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(policy, path):
    torch.save(policy.state_dict(), path)
    print(f"Model saved to: {path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir) / "actor_critic"
    model_dir = Path(args.model_dir) / "actor_critic"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    render_mode = "human" if args.render else "rgb_array"
    env = gym.make("Hopper-v4", render_mode=render_mode)

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== Actor-Critic training ===")
    print("State space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Using device:", device)
    print("Episodes:", args.episodes)
    print("Render:", args.render)
    print("Seed:", args.seed)
    print("AC target type:", args.ac_target_type)
    print("Normalize advantages:", args.normalize_advantages)
    print("Critic loss coef:", args.critic_loss_coef)
    print("Entropy coef:", args.entropy_coef)
    print("Run name:", args.run_name)

    policy = Policy(
        state_space=state_space,
        action_space=action_space,
    )

    agent = Agent(
        policy=policy,
        device=device,
        algorithm="actor_critic",
        normalize_advantages=args.normalize_advantages,
        critic_loss_coef=args.critic_loss_coef,
        entropy_coef=args.entropy_coef,
        ac_target_type=args.ac_target_type,
    )

    csv_path = output_dir / f"{args.run_name}.csv"
    final_model_path = model_dir / f"{args.run_name}.pth"
    best_model_path = model_dir / f"{args.run_name}_best.pth"

    recent_returns = deque(maxlen=20)
    results = []

    best_moving_avg_return = -float("inf")

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
            episode_reward += float(reward)
            step_count += 1

            if args.render:
                env.render()

        loss = agent.update_policy()

        recent_returns.append(episode_reward)
        moving_avg_return = float(np.mean(recent_returns))
        elapsed_time = time.time() - start_time

        if args.save_best_model and moving_avg_return > best_moving_avg_return:
            best_moving_avg_return = moving_avg_return
            save_checkpoint(policy, best_model_path)

        results.append(
            {
                "episode": episode,
                "steps": step_count,
                "return": episode_reward,
                "moving_avg_return": moving_avg_return,
                "loss": loss,
                "elapsed_time_sec": elapsed_time,
                "seed": args.seed,
                "algorithm": "actor_critic",
                "ac_target_type": args.ac_target_type,
                "normalize_advantages": args.normalize_advantages,
                "critic_loss_coef": args.critic_loss_coef,
                "entropy_coef": args.entropy_coef,
            }
        )

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

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "steps",
                "return",
                "moving_avg_return",
                "loss",
                "elapsed_time_sec",
                "seed",
                "algorithm",
                "ac_target_type",
                "normalize_advantages",
                "critic_loss_coef",
                "entropy_coef",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {csv_path}")

    if args.save_model:
        save_checkpoint(policy, final_model_path)

    env.close()


if __name__ == "__main__":
    main()