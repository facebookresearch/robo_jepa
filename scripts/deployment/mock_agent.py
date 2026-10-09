# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
mock_agent.py — Mock agent client for robot environment testing.

Connects to a GymEnvServer (real robot or mock_robot.py) over TCP and
sends scripted or random actions. Use this to:

  - Test a new robot environment implementation without writing a real agent.
  - Verify observation shapes, timing, and serialisation from the server side.
  - Run simple choreographed motions to sanity-check robot behaviour.

Modes
-----
  random        Send random delta-pose actions (default).
  choreography  Lift up 4 cm, then trace a small horizontal circle.

Usage:
  # Choreography test against live robot:
  python mock_agent.py --port 44568 --mode choreography

  # Random test against mock robot:
  python mock_robot.py --port 4754
  python mock_agent.py --port 4754 --mode random
"""

import sys
import time
import numpy as np
from argparse import ArgumentParser

from gym_tcp import GymEnvClient


def print_obs(step: str, obs: dict, t_sent: float) -> None:
    t_now = time.perf_counter() * 1000
    roundtrip = t_now - t_sent
    pose = np.array2string(obs["pose"][:3], precision=4, suppress_small=True)
    img_shapes = {k: obs[k].shape for k in obs if k.startswith("img_")}
    print(f"  {step:30s} | {roundtrip:6.0f}ms | xyz={pose} | images={img_shapes}")


# ---------------------------------------------------------------------------
# Mode 1: random
# ---------------------------------------------------------------------------

def run_random(env: GymEnvClient, episodes: int, steps: int, scale: float) -> None:
    for ep in range(episodes):
        print(f"\n[mock_agent] === random episode {ep} ===")
        obs, _ = env.reset()
        print(f"  reset → obs keys: {list(obs.keys())}")
        for step in range(steps):
            action = np.zeros(7, dtype=np.float32)
            action[:3] = (np.random.randn(3) * scale).astype(np.float32)
            action[3:6] = (np.random.randn(3) * scale).astype(np.float32)
            action[6] = float(np.random.randn() * scale * 0.5)
            t = time.perf_counter() * 1000
            obs, _, done, _ = env.step(action)
            print_obs(f"step {step}", obs, t)
            if done:
                print("  env done=True")
                break


# ---------------------------------------------------------------------------
# Mode 2: choreography — lift 4 cm then circle
# ---------------------------------------------------------------------------

def run_choreography(
    env: GymEnvClient,
    lift_cm: float = 4.0,
    lift_steps: int = 8,
    circle_radius: float = 0.015,
    circle_steps: int = 36,
) -> None:
    print("\n[mock_agent] === choreography ===")
    obs, _ = env.reset()
    print(f"  reset → start pose: {np.round(obs['pose'][:3], 4)}")

    # --- phase 1: lift up ---
    print(f"\n  phase 1: lift +{lift_cm} cm over {lift_steps} steps")
    dz = (lift_cm / 100.0) / lift_steps
    for i in range(lift_steps):
        action = np.zeros(7, dtype=np.float32)
        action[2] = dz
        t = time.perf_counter() * 1000
        obs, _, done, _ = env.step(action)
        print_obs(f"lift {i+1}/{lift_steps}", obs, t)

    # --- phase 2: horizontal circle ---
    print(f"\n  phase 2: circle  r={circle_radius*100:.1f} cm  over {circle_steps} steps")
    for i in range(circle_steps):
        a0 = 2 * np.pi * i / circle_steps
        a1 = 2 * np.pi * (i + 1) / circle_steps
        action = np.zeros(7, dtype=np.float32)
        action[0] = circle_radius * (np.cos(a1) - np.cos(a0))  # dx
        action[1] = circle_radius * (np.sin(a1) - np.sin(a0))  # dy
        t = time.perf_counter() * 1000
        obs, _, done, _ = env.step(action)
        print_obs(f"circle {i+1:2d}/{circle_steps}", obs, t)
        if done:
            print("  env done=True, stopping")
            break

    print(f"\n  end pose: {np.round(obs['pose'][:3], 4)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="Mock agent for testing robot environment servers.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=4754)
    parser.add_argument("--mode", type=str, default="choreography",
                        choices=["random", "choreography"],
                        help="random: random actions; choreography: lift + circle (default)")
    # random mode options
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--action_scale", type=float, default=0.005)
    # choreography options
    parser.add_argument("--lift_cm", type=float, default=4.0,
                        help="How many cm to lift up (default: 4.0)")
    parser.add_argument("--lift_steps", type=int, default=8,
                        help="Steps used to complete the lift (default: 8)")
    parser.add_argument("--circle_radius", type=float, default=0.015,
                        help="Circle radius in metres (default: 0.015 = 1.5 cm)")
    parser.add_argument("--circle_steps", type=int, default=36,
                        help="Steps to complete one full circle (default: 36)")
    parser.add_argument("--timeout_ms", type=int, default=60000)
    args = parser.parse_args()

    print(f"[mock_agent] connecting to {args.host}:{args.port} (mode={args.mode}) ...")
    env = GymEnvClient(host=args.host, port=args.port, timeout_ms=args.timeout_ms)

    try:
        if args.mode == "random":
            run_random(env, args.episodes, args.steps, args.action_scale)
        else:
            run_choreography(
                env,
                lift_cm=args.lift_cm,
                lift_steps=args.lift_steps,
                circle_radius=args.circle_radius,
                circle_steps=args.circle_steps,
            )
    finally:
        env.close()
        print("\n[mock_agent] done.")
