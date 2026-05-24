"""Inspect the Gymnasium Hopper-v4 environment and MuJoCo model.

This script is for Task 1 of the FAIML RL project.
It checks:
- state / observation space
- action space
- MuJoCo body names
- body masses
- degrees of freedom
- actuators
"""

import gymnasium as gym
import mujoco


def main():
    render = False

    if render:
        env = gym.make("Hopper-v4", render_mode="human")
    else:
        env = gym.make("Hopper-v4", render_mode="rgb_array")

    print("=== Gymnasium spaces ===")
    print("State space:", env.observation_space)
    print("Action space:", env.action_space)

    model = env.unwrapped.model

    print("\n=== MuJoCo model details ===")

    body_names = []
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        body_names.append(name)

    print("\nBody names:")
    print(body_names)

    print("\nBody masses:")
    print(model.body_mass)

    print("\nBody names and masses together:")
    for name, mass in zip(body_names, model.body_mass):
        print(f"{name}: {mass}")

    print("\nNumber of bodies / nbody:")
    print(model.nbody)

    # tells us how many independent movement-speed variables the physics engine tracks.
    print("\nNumber of degrees of freedom / nv:")
    print(model.nv)


    print("\nDoFs per body / body_dofnum:")
    print(model.body_dofnum)

    print("\nNumber of actuators / nu:")
    print(model.nu)

    env.close()


if __name__ == "__main__":
    main()