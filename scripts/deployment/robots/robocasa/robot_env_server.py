# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
RoboCasa simulation environment server for the robo_jepa world model.

Starts a GymEnvServer that wraps :class:`RoboCasaEnv` (the WM-aligned thin
wrapper around robocasa) so that any robo_jepa agent can drive the env
over ZMQ using canonical PandaOmron action chunks.

Usage
-----
    conda activate jepa_311
    cd scripts/deployment/robots/robocasa
    python robot_env_server.py --port 4754
"""

import os
import sys
from argparse import ArgumentParser

# Headless rendering must be set before importing mujoco/robosuite
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from gym_tcp import GymEnvServer
from robots.robocasa.robot_env import DEFAULT_CHUNK_SIZE, RoboCasaEnv

# Side-effect import: registers ``ObjectReachKitchen`` with robosuite's
# environment registry so ``RoboCasaEnv(task_name="ObjectReachKitchen")``
# can find it. Without this, the env construction below (BEFORE the
# conditional ``blocking_robot_env`` import) raises
# ``robosuite.environments.base.Exception: Environment
# ObjectReachKitchen not found``. Cheap import (no MuJoCo deps).
import robots.robocasa.object_reach_kitchen  # noqa: F401

# Side-effect import: registers ``ObjectPushKitchen`` with robosuite's env
# registry so ``RoboCasaEnv(task_name="ObjectPushKitchen")`` can find it when
# --object_push_task is set. Cheap import (no MuJoCo deps at import time).
import robots.robocasa.object_push_kitchen  # noqa: F401


def main():
    parser = ArgumentParser(description="RoboCasa simulation environment server for robo_jepa")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4754)
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help=(
            "RoboCasa task name. Defaults to 'CloseStandMixerHead', or to "
            "'ObjectReachKitchen' when --object_reach_task is set."
        ),
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["robot0_agentview_left"],
        help="Camera names to render (default: robot0_agentview_left)",
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=256,
        help="Camera width in pixels (default: 256)",
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=256,
        help="Camera height in pixels (default: 256)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: None)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="PandaOmron",
        help="Robot configuration (default: PandaOmron)",
    )
    parser.add_argument(
        "--no_init_home",
        action="store_true",
        default=False,
        help="Skip resetting arm to home pose after scene setup",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Number of robosuite OSC sub-steps per WM tick (i.e. per env.step). "
            f"Default {DEFAULT_CHUNK_SIZE} (= raw_fps 20 / target fps 5)."
        ),
    )
    parser.add_argument(
        "--no_lock_base",
        action="store_true",
        default=False,
        help=(
            "Disable the camera-stability lock that restores mobile-base "
            "joints after each arm-mode (control_mode=-1) sub-step."
        ),
    )
    parser.add_argument(
        "--wm_embodiment",
        type=str,
        default="franka_omron",
        choices=["franka_omron", "franka_omron_state_delta"],
        help=(
            "Selects the WM-side action representation that the env's "
            "components-dict packer should target. Both produce a 12-D "
            "layout-A action for the OSC controller; the state-delta "
            "variant additionally asserts the planner sent zero rotation "
            "and zeros the rotation slots before execution."
        ),
    )
    parser.add_argument(
        "--blocking_control",
        action="store_true",
        default=False,
        help=(
            "Wrap the env with BlockingRoboCasaEnv so each env.step "
            "interprets the planner's xyz delta as a target offset "
            "(in base-frame metres) and runs OSC sub-steps until the EE "
            "converges (or hits the max-steps cap). Without this flag "
            "every env.step runs exactly chunk_size OSC sub-steps and "
            "physically realises only a fraction of the commanded delta."
        ),
    )
    parser.add_argument(
        "--blocking_pos_tol",
        type=float,
        default=0.001,
        help="World-frame Euclidean tolerance (m) for declaring convergence.",
    )
    parser.add_argument(
        "--blocking_max_steps",
        type=int,
        default=200,
        help="Hard cap on OSC sub-steps per env.step in blocking mode.",
    )
    task_mode_group = parser.add_mutually_exclusive_group()
    task_mode_group.add_argument(
        "--reach_task",
        action="store_true",
        default=False,
        help=(
            "Enable the custom scene-agnostic reach task: each\n"
            "  reset(mode='goal') drives the EE to a random GOAL pose,\n"
            "  reset(mode='init') drives the EE to a random START pose,\n"
            "both sampled uniformly inside a 3-D ball of radius\n"
            "--reach_max_distance around the home EEF position. The\n"
            "underlying robocasa task (--task_name) is used only as a\n"
            "kitchen backdrop; the EXPERT_ACTIONS waypoint table is\n"
            "bypassed."
        ),
    )
    task_mode_group.add_argument(
        "--object_reach_task",
        action="store_true",
        default=False,
        help=(
            "Enable the object-reach task: the robot starts already\n"
            "grasping a single graspable object (default 'banana')\n"
            "spawned by ObjectReachKitchen. reset(mode='init') caches\n"
            "the object's start pose; reset(mode='goal') caches the\n"
            "target pose AND d_0. Reward is dense:\n"
            "  r = max(0, 1 - dist_obj_to_target / d_0).\n"
            "Mutually exclusive with --reach_task. Implies\n"
            "--task_name=ObjectReachKitchen unless explicitly set."
        ),
    )
    task_mode_group.add_argument(
        "--obstacle_reach_task",
        action="store_true",
        default=False,
        help=(
            "Enable the obstacle-reach task: same as --reach_task, but\n"
            "one in-front-of-robot countertop fixture (default:\n"
            "CoffeeMachine) is kept visible / colliding while the\n"
            "sampled (start, goal) EEF positions straddle it on\n"
            "opposite Y_base sides. Implies\n"
            "--reach_task semantics on the env side (the wrapper sets\n"
            "_reach_task=True). Mutually exclusive with --reach_task /\n"
            "--object_reach_task."
        ),
    )
    task_mode_group.add_argument(
        "--object_push_task",
        action="store_true",
        default=False,
        help=(
            "Enable the object-push task: a single free object (default\n"
            "'flour_bag') rests on the counter; reset(mode='init') places\n"
            "it at a sampled START xy, reset(mode='goal') at a small-\n"
            "displacement GOAL xy, with the EEF parked at a fixed top-down\n"
            "overhead pose above the object (mode-independent, so init/goal\n"
            "robot poses match). Reward is dense object-to-goal:\n"
            "  r = max(0, 1 - dist_obj_to_goal / d_0).\n"
            "Implies\n"
            "--task_name=ObjectPushKitchen unless explicitly set. Mutually\n"
            "exclusive with the other task modes."
        ),
    )
    parser.add_argument(
        "--reach_task_seed",
        type=int,
        default=0,
        help=(
            "Dedicated RNG seed for sampling the reach start/goal\n"
            "poses. INDEPENDENT of --seed (which controls the\n"
            "underlying scene's layout/objects), so changing this\n"
            "reseeds only the targets and leaves the kitchen alone."
        ),
    )
    parser.add_argument(
        "--reach_min_distance",
        type=float,
        default=0.0,
        help=(
            "Minimum 3-D EEF distance (m) between start and goal for the\n"
            "reach-task sampler (rejection sampling enforces \n"
            "reach_min_distance <= ||goal - start|| <= reach_max_distance).\n"
            "Default 0.0 m (no lower bound, legacy behaviour)."
        ),
    )
    parser.add_argument(
        "--reach_max_distance",
        type=float,
        default=0.20,
        help=(
            "Maximum 3-D EEF distance (m) from the home pose for the\nreach-task start/goal sampling. Default 0.20 m."
        ),
    )
    parser.add_argument(
        "--reach_clear_radius",
        type=float,
        default=1.0,
        help=(
            "When --reach_task is set, hide all NON-structural\n"
            "fixtures (StandMixer, Toaster, Blender, CoffeeMachine,\n"
            "DishRack, ElectricKettle, ToasterOven, Accessory, ...)\n"
            "within this radius (m) of the home EEF position so the\n"
            "reach task has a clean workspace. Structural fixtures\n"
            "(cabinets, drawers, counters, walls, floor, stove, hood,\n"
            "sink, microwave, oven, dishwasher, fridge, stool, window,\n"
            "box) are kept regardless of distance. Class-based filter,\n"
            "so it works against any robocasa scene. Set <= 0 to\n"
            "disable. Default 1.0 m."
        ),
    )
    parser.add_argument(
        "--kitchen_split",
        type=str,
        default="pretrain",
        choices=("pretrain", "target"),
        help=(
            "Which robocasa kitchen pool to draw the layout from.\n"
            "  pretrain (default) — ~2,500-scene pretraining pool;\n"
            "                        the kitchen is picked by env.rng\n"
            "                        from the full pool.\n"
            "  target           — the 10 held-out target kitchens\n"
            "                        (layout, style) = (1,1) … (10,10).\n"
            "                        ``seed % 10`` picks one specific\n"
            "                        kitchen per env instance, so seeds\n"
            "                        0..9 cover the full target set."
        ),
    )
    parser.add_argument(
        "--object_reach_object_name",
        type=str,
        default="banana",
        help=(
            "Name of the object to spawn / grasp in the object-reach\n"
            "task (must be a key in OBJECT_REACH_REGISTRY). Default 'banana'."
        ),
    )
    parser.add_argument(
        "--object_reach_task_seed",
        type=int,
        default=0,
        help=("Dedicated RNG seed for sampling the object-reach\nstart/target world poses. Independent of --seed."),
    )
    parser.add_argument(
        "--object_reach_max_distance",
        type=float,
        default=0.15,
        help=(
            "Maximum 3-D distance (m) from the home EEF position for\n"
            "the object-reach start/target sampling. Default 0.15 m."
        ),
    )
    parser.add_argument(
        "--object_reach_min_distance",
        type=float,
        default=0.0,
        help=(
            "Minimum start->goal EEF distance (m), enforced at reset via\n"
            "rejection sampling. Default 0.0 = OFF (legacy: no minimum)."
        ),
    )
    parser.add_argument(
        "--object_reach_max_pair_distance",
        type=float,
        default=None,
        help=(
            "Maximum start->goal EEF distance (m), enforced at reset via\n"
            "rejection sampling. Default None = OFF (no upper cap). Distinct\n"
            "from --object_reach_max_distance, which is the sampling ball radius."
        ),
    )
    parser.add_argument(
        "--object_reach_clear_radius",
        type=float,
        default=1.0,
        help=(
            "Hide all non-structural fixtures within this radius (m)\n"
            "of the home EEF position so the object-reach task has a\n"
            "clean workspace. Default 1.0 m."
        ),
    )
    parser.add_argument(
        "--object_reach_grasp_substeps",
        type=int,
        default=10,
        help=(
            "Number of OSC sub-steps the wrapper holds the gripper\n"
            "closed during the scripted initial grasp. Default 10."
        ),
    )
    parser.add_argument(
        "--object_reach_max_retries",
        type=int,
        default=3,
        help=("Maximum number of grasp retries before flagging\ninfo['object_reach_grasp_failed'] = True. Default 3."),
    )
    parser.add_argument(
        "--object_reach_kinematic_carry",
        action="store_true",
        default=False,
        help=(
            "If set, the OBJECT_REACH wrapper snaps the object's free\n"
            "joint qpos to the EEF every sub-step during carry, so the env\n"
            "carries the object with the EEF (bypassing physics-based\n"
            "contact transmission). Useful for noise-free reward signals."
        ),
    )
    parser.add_argument(
        "--object_reach_target_kitchen_id",
        type=int,
        default=None,
        help=(
            "Optional: pin one of robocasa's 10 target kitchens for the\n"
            "OBJECT_REACH task by passing layout_and_style_ids=[(ID, ID)]\n"
            "directly to robosuite.make (bypasses the seed%%10 mapping\n"
            "used by --kitchen_split target). Recommended over\n"
            "--kitchen_split target for OBJECT_REACH because per-kitchen\n"
            "banana spawn heuristics are tuned for scene_seed=0 — use this\n"
            "flag to keep scene_seed=0 while still selecting the kitchen."
        ),
    )
    parser.add_argument(
        "--object_push_object_name",
        type=str,
        default="flour_bag",
        help=(
            "Name of the object to spawn / push in the object-push task\n"
            "(must be a key in OBJECT_PUSH_REGISTRY). Default 'flour_bag'."
        ),
    )
    parser.add_argument(
        "--object_push_task_seed",
        type=int,
        default=0,
        help=(
            "Dedicated RNG seed for sampling the object-push start/goal xy\n"
            "pair. Independent of --seed. Same seed -> same (start, goal)."
        ),
    )
    parser.add_argument(
        "--object_push_min_distance",
        type=float,
        default=None,
        help=(
            "Minimum in-plane push displacement (m) ||goal - start||.\n"
            "Default None = use the wrapper default (0.05 m)."
        ),
    )
    parser.add_argument(
        "--object_push_max_distance",
        type=float,
        default=None,
        help=(
            "Maximum in-plane push displacement (m) ||goal - start||.\n"
            "Default None = use the wrapper default (0.15 m)."
        ),
    )
    parser.add_argument(
        "--object_push_forward_cone_deg",
        type=float,
        default=None,
        help=(
            "If set, restrict the push direction to a FORWARD cone of this full\n"
            "angular width (deg) centred on the robot base +x axis (away from\n"
            "the base). E.g. 90 = push within +/-45deg of straight-forward.\n"
            "Default None = uniform 360deg (original behaviour). Forward pushes\n"
            "were ~2x more successful in the confounder analysis."
        ),
    )
    parser.add_argument(
        "--object_push_clear_radius",
        type=float,
        default=1.0,
        help=(
            "Hide all non-structural fixtures within this radius (m) of the\n"
            "robot home EEF so the object-push task has a clean workspace.\n"
            "Default 1.0 m."
        ),
    )
    parser.add_argument(
        "--object_push_target_kitchen_id",
        type=int,
        default=None,
        help=(
            "Optional: pin one of robocasa's 10 target kitchens for the\n"
            "OBJECT_PUSH task via layout_and_style_ids=[(ID, ID)] (bypasses\n"
            "the seed%%10 mapping). Mirrors --object_reach_target_kitchen_id;\n"
            "keeps scene_seed fixed while selecting the kitchen."
        ),
    )
    parser.add_argument(
        "--obstacle_classes",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Allow-listed robocasa Fixture class names eligible to be the\n"
            "obstacle when --obstacle_reach_task is set. Default uses the\n"
            "wrapper's _OBSTACLE_REACH_DEFAULT_CLASSES (CoffeeMachine)."
        ),
    )
    parser.add_argument(
        "--obstacle_clearance_m",
        type=float,
        default=None,
        help=(
            "Extra clearance (m) added to the obstacle's Y_base half-extent\n"
            "when sampling the canonical left/right midpoints. Default\n"
            "uses the wrapper's _OBSTACLE_REACH_DEFAULT_CLEARANCE_M."
        ),
    )
    parser.add_argument(
        "--obstacle_lateral_jitter_std",
        type=float,
        default=None,
        help=(
            "Gaussian sigma (m, base frame) used to jitter the canonical\n"
            "left/right endpoint midpoints. Default uses the wrapper's\n"
            "_OBSTACLE_REACH_DEFAULT_JITTER_STD_M."
        ),
    )
    args = parser.parse_args()

    # Resolve --task_name default based on the active task mode (kept
    # explicit so --reach_task / upstream-task defaults stay untouched).
    if args.task_name is None:
        if args.object_reach_task:
            args.task_name = "ObjectReachKitchen"
        elif args.object_push_task:
            args.task_name = "ObjectPushKitchen"
        else:
            args.task_name = "CloseStandMixerHead"

    print(
        f"[robocasa_server] Creating RoboCasaEnv: task={args.task_name}, "
        f"cameras={args.camera_names}, resolution={args.camera_width}x{args.camera_height}, "
        f"robot={args.robot}, seed={args.seed}, chunk_size={args.chunk_size}, "
        f"lock_base_in_arm_mode={not args.no_lock_base}, wm_embodiment={args.wm_embodiment}, "
        f"blocking_control={args.blocking_control}, "
        f"reach_task={args.reach_task}, object_reach_task={args.object_reach_task}, "
        f"obstacle_reach_task={args.obstacle_reach_task}, "
        f"object_push_task={args.object_push_task}"
        + (
            f" [reach_task_seed={args.reach_task_seed}, min_dist={args.reach_min_distance} m,"
            f" max_dist={args.reach_max_distance} m,"
            f" clear_radius={args.reach_clear_radius} m,"
            f" kitchen_split={args.kitchen_split}]"
            if args.reach_task or args.obstacle_reach_task
            else ""
        )
        + (
            f" [object={args.object_reach_object_name},"
            f" object_reach_task_seed={args.object_reach_task_seed},"
            f" max_dist={args.object_reach_max_distance} m,"
            f" min_d0={args.object_reach_min_distance} m,"
            f" max_d0={args.object_reach_max_pair_distance} m,"
            f" clear_radius={args.object_reach_clear_radius} m,"
            f" grasp_substeps={args.object_reach_grasp_substeps},"
            f" max_retries={args.object_reach_max_retries},"
            f" kinematic_carry={args.object_reach_kinematic_carry}]"
            if args.object_reach_task
            else ""
        )
        + (
            f" [obstacle_classes={args.obstacle_classes},"
            f" clearance={args.obstacle_clearance_m} m,"
            f" jitter_std={args.obstacle_lateral_jitter_std} m]"
            if args.obstacle_reach_task
            else ""
        )
    )

    # Convert --object_reach_target_kitchen_id into layout_and_style_ids
    # for the inner env's robosuite.make() call. Bypasses the
    # --kitchen_split target seed%10 mapping (which would force a
    # specific scene_seed) and lets the caller keep scene_seed=0
    # (matching test/golden config) while still pinning the kitchen.
    _layout_and_style_ids = None
    if args.object_reach_task and args.object_reach_target_kitchen_id is not None:
        K = int(args.object_reach_target_kitchen_id)
        _layout_and_style_ids = [(K, K)]
        print(
            f"[robocasa_server] OBJECT_REACH target kitchen pinned to "
            f"K={K}: layout_and_style_ids={_layout_and_style_ids}"
        )
    if args.object_push_task and args.object_push_target_kitchen_id is not None:
        K = int(args.object_push_target_kitchen_id)
        _layout_and_style_ids = [(K, K)]
        print(
            f"[robocasa_server] OBJECT_PUSH target kitchen pinned to "
            f"K={K}: layout_and_style_ids={_layout_and_style_ids}"
        )

    env = RoboCasaEnv(
        task_name=args.task_name,
        camera_names=args.camera_names,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        seed=args.seed,
        robot=args.robot,
        init_robot_to_home=not args.no_init_home,
        chunk_size=args.chunk_size,
        lock_base_in_arm_mode=not args.no_lock_base,
        wm_embodiment=args.wm_embodiment,
        reach_task=args.reach_task or args.obstacle_reach_task,
        reach_task_seed=args.reach_task_seed,
        reach_min_distance=args.reach_min_distance,
        reach_max_distance=args.reach_max_distance,
        reach_clear_radius=args.reach_clear_radius,
        kitchen_split=args.kitchen_split,
        layout_and_style_ids=_layout_and_style_ids,
    )
    if args.blocking_control:
        from robots.robocasa.blocking_robot_env import (
            BlockingRoboCasaEnv,
            ObjectPushBlockingRoboCasaEnv,
            ObjectReachBlockingRoboCasaEnv,
            ObstacleReachBlockingRoboCasaEnv,
            ReachBlockingRoboCasaEnv,
        )

        if args.object_reach_task:
            env = ObjectReachBlockingRoboCasaEnv(
                env,
                object_reach_object_name=args.object_reach_object_name,
                object_reach_task_seed=args.object_reach_task_seed,
                object_reach_max_distance=args.object_reach_max_distance,
                object_reach_min_distance=args.object_reach_min_distance,
                object_reach_max_pair_distance=args.object_reach_max_pair_distance,
                object_reach_clear_radius=args.object_reach_clear_radius,
                object_reach_grasp_substeps=args.object_reach_grasp_substeps,
                object_reach_max_retries=args.object_reach_max_retries,
                object_reach_kinematic_carry=args.object_reach_kinematic_carry,
                pos_tol=args.blocking_pos_tol,
                max_steps=args.blocking_max_steps,
            )
            print(
                f"[robocasa_server] Wrapped with ObjectReachBlockingRoboCasaEnv "
                f"(object={args.object_reach_object_name}, "
                f"pos_tol={args.blocking_pos_tol} m, max_steps={args.blocking_max_steps})"
            )
        elif args.object_push_task:
            push_kwargs: dict = {}
            if args.object_push_min_distance is not None:
                push_kwargs["object_push_min_distance"] = args.object_push_min_distance
            if args.object_push_max_distance is not None:
                push_kwargs["object_push_max_distance"] = args.object_push_max_distance
            if args.object_push_forward_cone_deg is not None:
                push_kwargs["object_push_forward_cone_deg"] = args.object_push_forward_cone_deg
            env = ObjectPushBlockingRoboCasaEnv(
                env,
                object_push_object_name=args.object_push_object_name,
                object_push_task_seed=args.object_push_task_seed,
                object_push_clear_radius=args.object_push_clear_radius,
                pos_tol=args.blocking_pos_tol,
                max_steps=args.blocking_max_steps,
                **push_kwargs,
            )
            print(
                f"[robocasa_server] Wrapped with ObjectPushBlockingRoboCasaEnv "
                f"(object={args.object_push_object_name}, "
                f"forward_cone_deg={args.object_push_forward_cone_deg}, "
                f"pos_tol={args.blocking_pos_tol} m, max_steps={args.blocking_max_steps})"
            )
        elif args.obstacle_reach_task:
            obstacle_kwargs: dict = {}
            if args.obstacle_classes is not None:
                obstacle_kwargs["obstacle_classes"] = tuple(args.obstacle_classes)
            if args.obstacle_clearance_m is not None:
                obstacle_kwargs["obstacle_clearance_m"] = args.obstacle_clearance_m
            if args.obstacle_lateral_jitter_std is not None:
                obstacle_kwargs["obstacle_lateral_jitter_std"] = args.obstacle_lateral_jitter_std
            env = ObstacleReachBlockingRoboCasaEnv(
                env,
                reach_task_seed=args.reach_task_seed,
                reach_min_distance=args.reach_min_distance,
                reach_max_distance=args.reach_max_distance,
                reach_clear_radius=args.reach_clear_radius,
                kitchen_split=args.kitchen_split,
                pos_tol=args.blocking_pos_tol,
                max_steps=args.blocking_max_steps,
                **obstacle_kwargs,
            )
            print(
                f"[robocasa_server] Wrapped with ObstacleReachBlockingRoboCasaEnv "
                f"(pos_tol={args.blocking_pos_tol} m, max_steps={args.blocking_max_steps})"
            )
        elif args.reach_task:
            env = ReachBlockingRoboCasaEnv(
                env,
                reach_task_seed=args.reach_task_seed,
                reach_min_distance=args.reach_min_distance,
                reach_max_distance=args.reach_max_distance,
                reach_clear_radius=args.reach_clear_radius,
                kitchen_split=args.kitchen_split,
                pos_tol=args.blocking_pos_tol,
                max_steps=args.blocking_max_steps,
            )
            print(
                f"[robocasa_server] Wrapped with ReachBlockingRoboCasaEnv "
                f"(pos_tol={args.blocking_pos_tol} m, max_steps={args.blocking_max_steps})"
            )
        else:
            env = BlockingRoboCasaEnv(
                env,
                pos_tol=args.blocking_pos_tol,
                max_steps=args.blocking_max_steps,
            )
            print(
                f"[robocasa_server] Wrapped with BlockingRoboCasaEnv "
                f"(pos_tol={args.blocking_pos_tol} m, max_steps={args.blocking_max_steps})"
            )

    print(f"[robocasa_server] Starting server on {args.host}:{args.port}")
    server = GymEnvServer(env, host=args.host, port=args.port)
    server.start()
    print("[robocasa_server] Server stopped.")


if __name__ == "__main__":
    main()
