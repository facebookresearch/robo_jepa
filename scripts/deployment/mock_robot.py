# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
mock_robot.py — Mock robot environment server.

Starts a GymEnvServerWrapper on a local TCP port that returns synthetic
observations without requiring any physical hardware. Use this to:

  - Develop and test a new agent end-to-end before connecting to the real robot.
  - Verify that the TCP/observation serialisation works on both sides.
  - Run integration tests in CI without hardware.

Action format:
  - Components dict (preferred): ``{"eef_translation": (3,), "eef_rotation": (3,), "gripper": (1,)}``.
    The dict is packed into a 7-D delta vector via :func:`action_components.merge_components`
    using the default (non-RoboCasa) layout, then applied to the pose.
  - Legacy 7-D numpy array (``[dx, dy, dz, wx, wy, wz, dgripper]``) — unchanged behaviour.

Observation schema (identical to FrankaReacherEnv / LeRobotEnvironment):
  pose          (7,)  float64   EEF pose [x, y, z, wx, wy, wz, gripper]
  joints        (6,)  float64   joint angles in radians
  img_0         (H, W, 3) uint8 front camera (random noise by default)
  time_step_begin  float  ms timestamp when step started
  time_step_end    float  ms timestamp when robot motion finished
  time_send_obs    float  ms timestamp when observation was packed

Usage:
  python mock_robot.py --port 4754 --image_h 480 --image_w 640 --step_delay 0.05
"""

import sys
import time
import numpy as np
from argparse import ArgumentParser

from gym_tcp import GymEnvServer
from action_components import is_components_dict, merge_components


class MockRobotEnv:
    """Gym-compatible mock robot environment.

    Returns random images and a pose that drifts slowly with each applied
    action, making it easy to verify that an agent is receiving and acting
    on observations correctly.
    """

    def __init__(self, image_h: int = 480, image_w: int = 640, step_delay: float = 0.05):
        self.image_h = image_h
        self.image_w = image_w
        self.step_delay = step_delay  # simulated robot motion time (seconds)

        self.action_space = {}
        self.observation_space = {}

        # internal state
        self._pose = np.zeros(7, dtype=np.float64)  # [x, y, z, wx, wy, wz, gripper]
        self._pose[0] = 0.45  # reasonable x start position
        self._pose[2] = 0.35  # reasonable z start position
        self._joints = np.zeros(6, dtype=np.float64)
        self._step = 0

    def _make_observation(self) -> dict:
        now = time.perf_counter() * 1000
        return {
            "pose": self._pose.copy(),
            "joints": self._joints.copy(),
            "img_0": np.random.randint(0, 255, (self.image_h, self.image_w, 3), dtype=np.uint8),
            "time_step_begin": now,
            "time_step_end": now,
            "time_send_obs": time.perf_counter() * 1000,
        }

    def reset(self, **kwargs) -> tuple:
        self._pose = np.zeros(7, dtype=np.float64)
        self._pose[0] = 0.45
        self._pose[2] = 0.35
        self._joints = np.zeros(6, dtype=np.float64)
        self._step = 0
        print("[mock_robot] reset()")
        return self._make_observation(), []

    def step(self, action) -> tuple:
        """Apply a 7D EEF delta pose action and return a new observation.

        Accepts either:
        * Components dict (preferred) — packed via
          :func:`action_components.merge_components` into a 7-D delta
          ``[dx, dy, dz, wx, wy, wz, dgripper]``.
        * 7-D legacy numpy array — used as-is.
        """
        if is_components_dict(action):
            action = merge_components(action, "mock")
        action = np.asarray(action, dtype=np.float64)
        assert action.shape == (7,), f"expected action shape (7,), got {action.shape}"

        self._step += 1
        t_begin = time.perf_counter() * 1000

        # simulate robot motion latency
        time.sleep(self.step_delay)

        # accumulate delta pose (same semantics as real envs)
        self._pose += action
        # keep gripper in [0, 1]
        self._pose[6] = float(np.clip(self._pose[6], 0.0, 1.0))

        # small joint drift so joints are not always zero
        self._joints += np.random.randn(6) * 0.001

        t_end = time.perf_counter() * 1000
        obs = self._make_observation()
        obs["time_step_begin"] = t_begin
        obs["time_step_end"] = t_end

        done = False
        print(f"[mock_robot] step {self._step:4d} | action={np.round(action, 4)} | pose={np.round(self._pose, 3)}")
        return obs, 0.0, done, {}


if __name__ == "__main__":
    parser = ArgumentParser(description="Mock robot TCP server for agent testing.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4754)
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=640)
    parser.add_argument(
        "--step_delay",
        type=float,
        default=0.05,
        help="Simulated robot motion time per step in seconds (default: 0.05)",
    )
    args = parser.parse_args()

    print(f"[mock_robot] Starting mock robot server on {args.host}:{args.port}")
    print(f"[mock_robot] Image size: {args.image_h}x{args.image_w}, step delay: {args.step_delay}s")

    env = MockRobotEnv(
        image_h=args.image_h,
        image_w=args.image_w,
        step_delay=args.step_delay,
    )
    server = GymEnvServer(env, port=args.port)
    server.start()
    print("[mock_robot] server stopped.")
