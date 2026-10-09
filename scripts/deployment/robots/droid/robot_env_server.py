# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
DROID robot environment server for the robo_jepa framework.

Wraps the DROID ``RobotEnv`` (from droid-env) into the unified
observation / action interface expected by ``GymEnvServer`` so that
any robo_jepa agent (world-model planner, mock_agent, …) can
control the DROID hardware without modification.

Unified interface
-----------------
Action:  float vector whose dimensionality depends on the action space:
         - cartesian spaces (7): [dx, dy, dz, droll, dpitch, dyaw, d_gripper]
         - joint spaces     (8): [dj0, dj1, …, dj6, d_gripper]
Obs:     dict with keys  pose (7,), joints (7,), img_0 … img_N (H,W,3), timestamps

Usage
-----
    cd scripts/deployment/robots/droid
    python robot_env_server.py --port 4754

See ``--help`` for the full list of DROID-specific options.
"""

from argparse import ArgumentParser
import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from scipy.spatial.transform import Rotation

from droid.robot_env import RobotEnv
from droid.misc.transformations import quat_to_euler
from gym_tcp import GymEnvServer
from action_components import is_components_dict, merge_components


class DroidEnvWrapper:
    """Adapts a DROID ``RobotEnv`` to the robo_jepa unified Gym interface.

    The wrapper translates between DROID's rich observation dictionary and
    the flat ``{pose, joints, img_*, time_*}`` format that all robo_jepa
    agents expect, and forwards actions to ``RobotEnv.step()``.

    The action dimensionality is determined by the chosen action space
    (7 for cartesian, 8 for joint — matching ``RobotEnv.DoF``).

    Parameters
    ----------
    action_space : str
        DROID action space — one of ``cartesian_velocity``,
        ``cartesian_position``, ``joint_velocity``, ``joint_position``.
    gripper_action_space : str or None
        DROID gripper action space (e.g. ``"velocity"``).  ``None`` to
        use the DROID default.
    camera_kwargs : dict
        Per-camera-type keyword arguments forwarded to
        ``MultiCameraWrapper`` (resolution, etc.).
    do_reset : bool
        Whether to call ``reset()`` inside the DROID ``RobotEnv``
        constructor.
    """

    def __init__(
        self,
        action_space="cartesian_velocity",
        gripper_action_space=None,
        camera_kwargs=None,
        do_reset=True,
        blocking_step=False,
    ):
        if camera_kwargs is None:
            camera_kwargs = {}
        self._env = RobotEnv(
            action_space=action_space,
            gripper_action_space=gripper_action_space,
            camera_kwargs=camera_kwargs,
            do_reset=do_reset,
        )
        self._blocking_step = blocking_step

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_joint_positions(self):
        """Return the current 7-DoF joint positions as a numpy array."""
        state, _ = self._env.get_state()
        return np.asarray(state["joint_positions"], dtype=np.float64)

    def _read_gripper_position(self):
        """Return the current gripper position as a float (0=open, 1=closed)."""
        state, _ = self._env.get_state()
        g = state.get("gripper_position", 0.0)
        if isinstance(g, (list, np.ndarray)):
            g = float(np.asarray(g).flat[0])
        return float(g)

    def _wait_joints_arrived(
        self,
        target,
        atol=0.05,
        timeout=3.0,
        poll_hz=20.0,
    ):
        """Block until joint positions converge to *target*.

        Parameters
        ----------
        target : array-like, shape (7,)
        atol : float
            Max per-joint absolute tolerance (radians).
        timeout : float
            Give up after this many seconds.
        poll_hz : float
            Polling frequency.
        """
        target = np.asarray(target, dtype=np.float64)
        dt = 1.0 / poll_hz
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            cur = self._read_joint_positions()
            if np.all(np.abs(cur - target) < atol):
                return True
            time.sleep(dt)
        print(
            f"[DroidEnvWrapper] WARNING: joints did not converge within {timeout}s "
            f"(max err = {np.max(np.abs(self._read_joint_positions() - target)):.4f} rad)"
        )
        return False

    def _wait_gripper_arrived(
        self,
        target,
        atol=0.05,
        timeout=3.0,
        poll_hz=20.0,
    ):
        """Block until gripper position converges to *target*.

        Parameters
        ----------
        target : float
            Desired gripper position (0=open, 1=closed).
        atol : float
            Absolute tolerance.
        timeout : float
            Give up after this many seconds.
        poll_hz : float
            Polling frequency.
        """
        dt = 1.0 / poll_hz
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            cur = self._read_gripper_position()
            if abs(cur - target) < atol:
                return True
            time.sleep(dt)
        print(
            f"[DroidEnvWrapper] WARNING: gripper did not converge within {timeout}s "
            f"(err = {abs(self._read_gripper_position() - target):.4f})"
        )
        return False

    def _build_obs(self, include_images=True):
        """Read DROID sensors and return a unified observation dict.

        Parameters
        ----------
        include_images : bool
            If False, skip camera reads (much faster). The resulting dict
            will still have ``pose``, ``joints``, and timestamps but no
            ``img_*`` keys.
        """
        time_begin = time.perf_counter() * 1000

        if include_images:
            obs = self._env.get_observation()
        else:
            # Fast path: robot state only, no camera overhead
            state_dict, _ = self._env.get_state()
            obs = {"robot_state": state_dict}

        # --- pose (7,): [x, y, z, roll, pitch, yaw, gripper_width] ---
        state = obs["robot_state"]
        cart = np.array(state["cartesian_position"])
        if cart.ndim == 2:
            pos = cart[:3, 3]
            quat = Rotation.from_matrix(cart[:3, :3]).as_quat()
            euler = quat_to_euler(quat)
        else:
            pos = cart[:3]
            euler = cart[3:6]

        gripper_state = state.get("gripper_position", 0.0)
        if isinstance(gripper_state, (list, np.ndarray)):
            gripper_state = float(np.asarray(gripper_state).flat[0])
        pose = np.concatenate([pos, euler, [gripper_state]])

        # --- joints (7,) ---
        joints = np.array(state["joint_positions"])

        # --- camera images: img_0, img_1, … ---
        img_dict = {}
        if include_images and "image" in obs:
            idx = 0
            for cam_id in sorted(obs["image"].keys()):
                img = np.asarray(obs["image"][cam_id])
                if img.ndim == 3 and img.shape[-1] == 4:
                    img = img[..., :3]
                img_dict[f"img_{idx}"] = img[..., ::-1]
                idx += 1

        time_end = time.perf_counter() * 1000
        return {
            "pose": pose.astype(np.float64),
            "joints": joints.astype(np.float64),
            **img_dict,
            "time_step_begin": time_begin,
            "time_step_end": time_end,
            "time_send_obs": time.perf_counter() * 1000,
        }

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        """Reset the DROID environment and return ``(obs, obs_list)``.

        Supports the same keyword arguments as the Franka env's
        ``FrankaReacherEnv.reset()`` so that task JSON files with
        ``joints_sequence`` / ``gripper_sequence`` work out of the box.

        Parameters
        ----------
        randomize : bool
            Forwarded to ``RobotEnv.reset()`` when no joint sequence is
            given.  Default ``False``.
        go_home : bool
            If ``True`` (default), send the robot to its DROID home
            position (with gripper open) before replaying any sequence.
        joints_sequence : list[list[float]] | None
            Optional sequence of 7-DoF joint targets.  Each entry is
            moved to with ``_robot.update_joints()``.
        gripper_sequence : list[float] | None
            Gripper state for each waypoint.  ``0.0`` = open, ``1.0`` =
            closed (DROID convention).

        Returns
        -------
        obs : dict
            Final observation (with images).
        obs_list : list[dict]
            Observation captured after each waypoint in the sequence
            (empty list when no sequence is given).
        """
        joints_sequence = kwargs.get("joints_sequence", None)
        gripper_sequence = kwargs.get("gripper_sequence", None)
        go_home = kwargs.get("go_home", True)
        randomize = kwargs.get("randomize", False)

        obs_list = []

        if go_home:
            self._env.reset(randomize=False)

        if joints_sequence is not None:
            if gripper_sequence is None:
                gripper_sequence = [0.0] * len(joints_sequence)

            for joints, gripper in zip(joints_sequence, gripper_sequence):
                target_joints = np.asarray(joints, dtype=np.float64)
                target_gripper = float(gripper)

                self._env._robot.update_joints(
                    target_joints,
                    velocity=False,
                    blocking=True,
                )
                self._wait_joints_arrived(target_joints)

                self._env._robot.update_gripper(
                    target_gripper,
                    velocity=False,
                    blocking=True,
                )
                self._wait_gripper_arrived(target_gripper)

                obs_list.append(self._build_obs())
        elif not go_home:
            self._env.reset(randomize=randomize)

        obs = self._build_obs()
        return obs, obs_list

    @property
    def dof(self):
        """Action dimensionality (7 for cartesian spaces, 8 for joint spaces)."""
        return self._env.DoF

    def step(self, action):
        """Execute an action on the DROID robot.

        Parameters
        ----------
        action : dict | array-like, shape (DoF,)
            Either:
            * Components dict — packed via
              :func:`action_components.merge_components` into the default
              7-D ``[eef_translation, eef_rotation, gripper]`` layout.
              For joint action spaces (DoF=8) the dict's ``"joints"`` key
              is used (must be 7-D); the gripper is appended.
            * ``[dx, dy, dz, droll, dpitch, dyaw, d_gripper]`` for cartesian
              spaces (DoF=7).
            * ``[dj0, dj1, dj2, dj3, dj4, dj5, dj6, d_gripper]`` for joint
              spaces (DoF=8).

        Returns
        -------
        obs : dict
        reward : None
        done : bool
        info : dict
        """
        expected_dof = self._env.DoF
        if is_components_dict(action):
            if "joints" in action and expected_dof == 8:
                joints = np.asarray(action["joints"], dtype=np.float64).reshape(-1)
                assert joints.size == 7, f"DROID joints must be 7-D, got {joints.shape}"
                grip = action.get("gripper", np.zeros(1))
                grip = np.asarray(grip, dtype=np.float64).reshape(-1)[:1]
                action = np.concatenate([joints, grip])
            else:
                action = merge_components(action, "droid")
        action = np.asarray(action, dtype=np.float64)
        assert action.shape == (expected_dof,), (
            f"Expected action shape ({expected_dof},) for action space '{self._env.action_space}', got {action.shape}"
        )

        action_info = self._env.update_robot(
            action,
            action_space=self._env.action_space,
            gripper_action_space=self._env.gripper_action_space,
            blocking=False,
        )

        if self._blocking_step and action_info is not None:
            target_joints = np.asarray(action_info.get("joint_position", []), dtype=np.float64)
            if target_joints.size == 7:
                self._wait_joints_arrived(target_joints, atol=0.02)
            target_gripper = float(action_info.get("gripper_position", 0.0))
            self._wait_gripper_arrived(target_gripper)

        obs = self._build_obs()
        done = False
        return obs, None, done, {"action_info": action_info}

    def step_chunk(self, actions, dt=1.0 / 15):
        """Execute a chunk of actions on the DROID robot at the given rate.

        Only the final action triggers a full observation build (with camera
        reads). Intermediate actions read only robot state (no cameras),
        making chunk execution significantly faster than calling step() N
        times.

        Parameters
        ----------
        actions : list[dict] | array-like, shape (N, DoF)
            Sequence of N actions to execute. Either a list of components
            dicts (each routed through :meth:`step`) or a numpy array.
        dt : float
            Seconds between actions (default 1/15 ≈ 66 ms).

        Returns
        -------
        obs : dict         — full observation (with images) after the last action
        reward : None
        done : bool
        info : dict        — info["chunk_obs"] is a list of N observation dicts;
                              only the last one contains images.
        """
        if isinstance(actions, list):
            tick_actions = list(actions)
            tick_actions_arrays = [
                None if is_components_dict(a) else np.asarray(a, dtype=np.float64) for a in tick_actions
            ]
            # we'll dispatch each tick through self.step which handles dict and array uniformly
        else:
            actions = np.asarray(actions, dtype=np.float64)
            expected_dof = self._env.DoF
            assert actions.ndim == 2 and actions.shape[1] == expected_dof, (
                f"Expected actions shape (N, {expected_dof}), got {actions.shape}"
            )
            tick_actions = [actions[i] for i in range(actions.shape[0])]

        chunk_obs = []
        for action in tick_actions:
            t0 = time.perf_counter()
            obs, _, _, _ = self.step(action)
            chunk_obs.append(obs)
            elapsed = time.perf_counter() - t0
            sleep_left = dt - elapsed
            if sleep_left > 0:
                time.sleep(sleep_left)

        return chunk_obs[-1], None, False, {"chunk_obs": chunk_obs}

    def close(self):
        """Graceful shutdown."""
        pass


def main():
    parser = ArgumentParser(description="DROID robot environment server for robo_jepa")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4754)
    parser.add_argument(
        "--action_space",
        type=str,
        default="cartesian_velocity",
        choices=[
            "cartesian_velocity",
            "cartesian_position",
            "joint_velocity",
            "joint_position",
        ],
        help="DROID action space",
    )
    parser.add_argument(
        "--gripper_action_space",
        type=str,
        default=None,
        help="DROID gripper action space (e.g. 'velocity'). None uses the DROID default.",
    )
    parser.add_argument(
        "--camera_kwargs",
        type=str,
        default="{}",
        help="JSON string of camera kwargs forwarded to MultiCameraWrapper",
    )
    parser.add_argument(
        "--no_reset",
        action="store_true",
        default=False,
        help="Skip the initial reset in the DROID RobotEnv constructor",
    )
    parser.add_argument(
        "--blocking_step",
        action="store_true",
        default=False,
        help="Poll joint/gripper arrival after each step/step_chunk action",
    )
    args = parser.parse_args()

    camera_kwargs = json.loads(args.camera_kwargs)

    env = DroidEnvWrapper(
        action_space=args.action_space,
        gripper_action_space=args.gripper_action_space,
        camera_kwargs=camera_kwargs,
        do_reset=not args.no_reset,
        blocking_step=args.blocking_step,
    )
    server = GymEnvServer(env, host=args.host, port=args.port)
    server.start()
    print("exiting server...")


if __name__ == "__main__":
    main()
