# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generic ObjectPush kitchen task.

Spawns exactly ONE free (pushable) object on a reachable, non-corner counter
in an otherwise unmodified robocasa kitchen. The object name is selected via
the ``object_name`` constructor kwarg (default: ``"banana"``) and is looked up
in :data:`OBJECT_PUSH_REGISTRY`. The task carries no reward / no success
condition of its own — those live on the wrapper
(``ObjectPushBlockingRoboCasaEnv``). This module's only responsibility is to
make a single pushable object available in the scene under the body name
``"obj"`` (so the wrapper can read it via ``env.obj_body_id["obj"]``).

This mirrors :mod:`object_reach_kitchen` but is intentionally lean: the push
task never grasps the object, so the per-object friction / gripper-pad /
grasp-approach tuning that object_reach needs is omitted here. The scene just
needs a single free object at a deterministic, reachable spot; the wrapper
then re-positions it to the per-episode start / goal xy and drives the EEF to
a standoff behind it.

The class is registered with robosuite at module import time via
``register_env`` so it can be instantiated by string name
(``robosuite.make("ObjectPushKitchen", ...)``). Importing this module is
side-effect-only with respect to robosuite's env registry; no env is
constructed at import time.
"""

import math
import os

import numpy as np
import robocasa
from scipy.spatial.transform import Rotation

from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.fixtures import FixtureType
from robosuite.environments.base import register_env

from robots.robocasa.robot_env import _REACH_STRUCTURAL_BASE_CLASSES

# Root of the RoboCasa "lightwheel" flour_bag asset variants (FlourBag0XX).
# The random flour_bag sampler draws from this pool; the pinned-variant push
# entries below reference a single ``FlourBagXXX/model.xml`` here so the same
# object spawns in every kitchen (decoupling object identity from layout).
_FLOURBAG_ROOT = os.path.join(robocasa.models.assets_root, "objects", "lightwheel", "flour_bag")


def _flourbag_mjcf(variant: str) -> str:
    """Absolute ``model.xml`` path for a pinned flour_bag variant (e.g. 'FlourBag001')."""
    return os.path.join(_FLOURBAG_ROOT, variant, "model.xml")


# Reachability spawn overrides shared by every flour_bag entry (random + the
# pinned variants). On kitchens 1/2/3 the default robocasa placement sampler
# lands the bag out of the Panda's reach, so a base-frame xy is pinned; the
# push wrapper then jitters the START around this and adds the push vector.
_FLOURBAG_PER_SEED_SPAWN_XY = {
    1: (0.50, 0.0),
    6: (0.50, 0.0),
    7: (0.50, 0.0),
    8: (0.40, -0.15),
}
_FLOURBAG_PER_KITCHEN_SPAWN_XY = {
    1: (0.25, 0.05),
    2: (0.40, -0.30),
    3: (0.30, -0.15),
}


# ---------------------------------------------------------------------------
# Object registry
# ---------------------------------------------------------------------------
# Each entry drives this kitchen task's ``_get_obj_cfgs`` (``obj_groups`` +
# ``sample_region_kwargs`` + ``placement_size`` + ``placement_pos``) and the
# deterministic base-spawn overrides consumed in ``_reset_internal``.
#
# The ``per_kitchen_spawn_xy`` / ``per_seed_spawn_xy`` reachability overrides
# for the banana are inherited from the object_reach tuning: on kitchens 1/2/3
# the default robocasa placement sampler lands the banana out of the Panda's
# reach, so a base-frame xy is pinned. The push wrapper jitters the START a
# few cm around whatever the kitchen spawns and adds the push vector on top,
# so these centres only need to put the object in a comfortably reachable spot
# (the wrapper bounds the goal to stay within a reachable disk of the centre).
#
# Optional fields (default empty / inactive if missing):
#   * ``per_kitchen_spawn_xy``  : dict[layout_id, (x_base, y_base)]
#   * ``per_seed_spawn_xy``     : dict[scene_seed, (x_base, y_base)]
#   * ``per_kitchen_spawn_quat_wxyz`` : dict[layout_id, (w, x, y, z)]
OBJECT_PUSH_REGISTRY: dict[str, dict] = {
    "flour_bag": {
        "obj_groups": "flour_bag",
        "graspable": False,  # flour_bag has no graspable variant; push doesn't grasp
        "sample_region_kwargs": {},
        "placement_size": (0.40, 0.30),
        "placement_pos": (0.0, -1.0),
        "per_seed_spawn_xy": _FLOURBAG_PER_SEED_SPAWN_XY,
        # Sink-free base-frame spots. k1/k3 use the object_reach *can*
        # positions (off the sink), since the push task keeps the object
        # resting on the counter (it is not grasped + lifted away).
        "per_kitchen_spawn_xy": _FLOURBAG_PER_KITCHEN_SPAWN_XY,
    },
    # ── Pinned flour_bag variants ──────────────────────────────────────────
    # The random ``flour_bag`` sampler spawns a different variant per kitchen;
    # the ceiling analysis over the 22m/50m/100m planning sweep found only a
    # few variants actually get pushed with positive reward. These entries pin
    # exactly one ``FlourBagXXX`` mesh (via obj_variant_mjcf) so the SAME
    # object can be used in every kitchen:
    #   * FlourBag001 — best; positive reward in BOTH k8 and k10.
    #   * FlourBag010 — the cyan bag; positive reward in k1.
    #   * FlourBag002 — weaker; k4.
    "flourbag001": {
        "obj_groups": "flour_bag",
        "obj_variant_mjcf": _flourbag_mjcf("FlourBag001"),
        "graspable": False,
        "sample_region_kwargs": {},
        "placement_size": (0.40, 0.30),
        "placement_pos": (0.0, -1.0),
        "per_seed_spawn_xy": _FLOURBAG_PER_SEED_SPAWN_XY,
        "per_kitchen_spawn_xy": _FLOURBAG_PER_KITCHEN_SPAWN_XY,
    },
    "flourbag010": {
        "obj_groups": "flour_bag",
        "obj_variant_mjcf": _flourbag_mjcf("FlourBag010"),
        "graspable": False,
        "sample_region_kwargs": {},
        "placement_size": (0.40, 0.30),
        "placement_pos": (0.0, -1.0),
        "per_seed_spawn_xy": _FLOURBAG_PER_SEED_SPAWN_XY,
        "per_kitchen_spawn_xy": _FLOURBAG_PER_KITCHEN_SPAWN_XY,
    },
    "flourbag002": {
        "obj_groups": "flour_bag",
        "obj_variant_mjcf": _flourbag_mjcf("FlourBag002"),
        "graspable": False,
        "sample_region_kwargs": {},
        "placement_size": (0.40, 0.30),
        "placement_pos": (0.0, -1.0),
        "per_seed_spawn_xy": _FLOURBAG_PER_SEED_SPAWN_XY,
        "per_kitchen_spawn_xy": _FLOURBAG_PER_KITCHEN_SPAWN_XY,
    },
    "banana": {
        "obj_groups": "banana",
        "sample_region_kwargs": {},
        "placement_size": (0.40, 0.30),
        "placement_pos": (0.0, -1.0),
        "per_seed_spawn_xy": _FLOURBAG_PER_SEED_SPAWN_XY,
        "per_kitchen_spawn_xy": _FLOURBAG_PER_KITCHEN_SPAWN_XY,
        "per_kitchen_spawn_quat_wxyz": {
            3: (0.99, -0.006, 0.076, -0.120),
        },
    },
}


# Fixed world-frame yaw (rad) the spawned object's free-joint quaternion is
# pinned to after the upstream placement loop runs. Mirrors
# :data:`object_reach_kitchen._OBJECT_REACH_FIXED_WORLD_YAW` so a banana lies
# at a consistent orientation across kitchen layouts.
_OBJECT_PUSH_FIXED_WORLD_YAW: float = -math.pi / 6


class ObjectPushKitchen(Kitchen):
    """RoboCasa kitchen that spawns a single pushable object named ``"obj"``.

    Lean sibling of :class:`object_reach_kitchen.ObjectReachKitchen`. Selects
    the object via the ``object_name`` ctor kwarg (default ``"banana"``,
    overridable through the class-level :attr:`DEFAULT_OBJECT_NAME` which the
    wrapper mutates), looks it up in :data:`OBJECT_PUSH_REGISTRY`, pins its
    world-frame yaw, applies the optional per-kitchen / per-seed base-frame xy
    override, and hides every other scene object + non-structural fixture so
    the workspace contains only the pushable object on an otherwise static
    kitchen backdrop.
    """

    DEFAULT_OBJECT_NAME = "flour_bag"

    def __init__(self, object_name: str | None = None, *args, **kwargs):
        if object_name is None:
            object_name = type(self).DEFAULT_OBJECT_NAME
        if object_name not in OBJECT_PUSH_REGISTRY:
            raise KeyError(
                f"object_name={object_name!r} not in OBJECT_PUSH_REGISTRY; available: {sorted(OBJECT_PUSH_REGISTRY)}"
            )
        self.object_name = str(object_name)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER_NON_CORNER, size=(0.6, 0.6)),
        )
        self.init_robot_base_ref = self.counter

    def _get_obj_cfgs(self):
        spec = OBJECT_PUSH_REGISTRY[self.object_name]
        # If the spec pins ``obj_variant_mjcf``, spawn exactly that mesh
        # (robocasa's sample_kitchen_object_helper fast-paths ``obj_groups``
        # ending in ``.xml``), bypassing the per-episode random variant draw —
        # so the SAME flour_bag is used in every kitchen. Otherwise fall back
        # to the ``obj_groups`` category (random variant).
        obj_groups_spec = spec.get("obj_variant_mjcf") or spec["obj_groups"]
        return [
            dict(
                name="obj",
                obj_groups=obj_groups_spec,
                # Free joint is added regardless (placed objects rest on the
                # counter and are pushable); ``graspable`` only filters the
                # object-category pool. flour_bag has no graspable variant, so
                # the registry sets graspable=False for it.
                graspable=bool(spec.get("graspable", True)),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(spec.get("sample_region_kwargs", {})),
                    size=tuple(spec["placement_size"]),
                    pos=tuple(spec["placement_pos"]),
                ),
            )
        ]

    def _reset_internal(self):
        """Pin the object's world yaw, apply the optional base-frame xy /
        quat override, and hide every non-``"obj"`` scene object + every
        non-structural fixture.

        The override semantics match
        :meth:`object_reach_kitchen.ObjectReachKitchen._reset_internal`: the
        per-target-kitchen xy (keyed by ``self.layout_id``) takes precedence
        over the per-seed xy (keyed by ``self.seed``); the dict value is an
        ABSOLUTE position in the robot base frame, mapped to world via
        ``base_pos[:2] + R_wb[:2,:2] @ (x_base, y_base)``. A per-target-kitchen
        quat override, if present, fully supersedes the fixed-yaw quaternion
        for the listed layouts.
        """
        super()._reset_internal()

        if "obj" not in self.objects:
            return
        obj = self.objects["obj"]
        joint_name = obj.joints[0]

        spec = OBJECT_PUSH_REGISTRY[self.object_name]
        seed = getattr(self, "seed", None)
        layout_id = getattr(self, "layout_id", None)

        # Free-joint qpos layout: [x, y, z, qw, qx, qy, qz].
        qpos = np.array(self.sim.data.get_joint_qpos(joint_name)).copy()

        half = _OBJECT_PUSH_FIXED_WORLD_YAW / 2.0
        qpos[3:7] = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])

        per_target_xy = spec.get("per_kitchen_spawn_xy", {}).get(layout_id)
        per_seed_xy = spec.get("per_seed_spawn_xy", {}).get(seed)
        override_xy = per_target_xy if per_target_xy is not None else per_seed_xy
        if override_xy is not None:
            base_pos, R_wb = self._get_base_world_pose()
            xy_base = np.asarray(override_xy, dtype=np.float64).reshape(2)
            qpos[0:2] = base_pos[:2] + R_wb[:2, :2] @ xy_base

        self.sim.data.set_joint_qpos(joint_name, qpos)

        per_target_quat = spec.get("per_kitchen_spawn_quat_wxyz", {}).get(layout_id)
        if per_target_quat is not None:
            qpos2 = np.array(self.sim.data.get_joint_qpos(joint_name)).copy()
            qpos2[3:7] = np.asarray(per_target_quat, dtype=np.float64).reshape(4)
            self.sim.data.set_joint_qpos(joint_name, qpos2)

        self.sim.forward()

        # Hide every non-"obj" scene object and every non-structural fixture.
        self._hide_bodies(
            self.objects.items(),
            should_skip=lambda name, _obj: name == "obj",
        )
        fixtures = getattr(self, "fixtures", None) or {}
        self._hide_bodies(
            fixtures.items(),
            should_skip=lambda _name, fixture: any(
                base.__name__ in _REACH_STRUCTURAL_BASE_CLASSES for base in type(fixture).__mro__
            ),
        )

    # ------------------------------------------------------------------
    # Helpers (mirrors of the ObjectReachKitchen variants)
    # ------------------------------------------------------------------

    def _get_base_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(base_pos_world, R_world_from_base)`` for the mobile base."""
        for body_name in ("mobilebase0_base", "robot0_base", "base"):
            try:
                bid = self.sim.model.body_name2id(body_name)
                pos = np.asarray(self.sim.data.body_xpos[bid], dtype=np.float64).copy()
                quat_wxyz = self.sim.data.body_xquat[bid]
                q = np.array(
                    [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
                    dtype=np.float64,
                )
                return pos, Rotation.from_quat(q).as_matrix()
            except Exception:
                continue
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)

    def _body_subtree_ids(self, root_bid: int) -> list:
        """Return ``[root_bid, *descendant_body_ids]`` for ``root_bid``."""
        model = self.sim.model
        out = [root_bid]
        for bid in range(model.nbody):
            if bid == root_bid:
                continue
            ancestor = bid
            while ancestor != 0:
                parent = model.body_parentid[ancestor]
                if parent == ancestor:
                    break
                if parent == root_bid:
                    out.append(bid)
                    break
                ancestor = parent
        return out

    def _hide_bodies(self, items, should_skip) -> None:
        """Hide every body subtree in ``items`` not matched by ``should_skip``.

        Zeroes ``geom_rgba[:, 3]`` (alpha), ``geom_contype`` and
        ``geom_conaffinity`` for every geom in the entity's body subtree.
        """
        sim = self.sim
        model = sim.model
        any_hidden = False
        for name, entity in items:
            if should_skip(name, entity):
                continue
            root_body_name = getattr(entity, "root_body", None)
            if root_body_name is None:
                continue
            try:
                root_bid = model.body_name2id(root_body_name)
            except Exception:
                continue
            for bid in self._body_subtree_ids(root_bid):
                n_geoms = int(model.body_geomnum[bid])
                geom_start = int(model.body_geomadr[bid])
                for i in range(n_geoms):
                    gid = geom_start + i
                    model.geom_rgba[gid, 3] = 0.0
                    model.geom_contype[gid] = 0
                    model.geom_conaffinity[gid] = 0
            any_hidden = True
        if any_hidden:
            sim.forward()

    def _check_success(self):
        # Wrapper owns task success; this env reports no intrinsic success.
        return False

    def reward(self, action=None):
        # Wrapper owns task reward; this env emits a constant 0.0.
        return 0.0


# Expose under ``robosuite.make("ObjectPushKitchen", ...)``. This only
# registers the class; no env is instantiated at import time.
register_env(ObjectPushKitchen)
