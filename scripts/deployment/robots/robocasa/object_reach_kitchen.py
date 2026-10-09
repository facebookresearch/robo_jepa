# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generic ObjectReach kitchen task.

Spawns exactly ONE free (graspable) object on a reachable, non-corner counter
in an otherwise unmodified robocasa kitchen. The object name is selected via
the ``object_name`` constructor kwarg (default: ``"banana"``) and is looked up
in :data:`OBJECT_REACH_REGISTRY`. The task carries no reward / no success
condition of its own — those live on the wrapper
(``ObjectReachBlockingRoboCasaEnv``). This module's only responsibility is to
make a single graspable object available in the scene under the body name
``"obj"`` (so the wrapper can read it via ``env.obj_body_id["obj"]``).

The class is registered with robosuite at module import time via
``register_env`` so it can be instantiated by string name (``robosuite.make(
"ObjectReachKitchen", ...)``). Importing this module is side-effect-only with
respect to robosuite's env registry; no env is constructed at import time.
"""

import math
import os

import numpy as np
from scipy.spatial.transform import Rotation

import robocasa
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.fixtures import FixtureType
from robosuite.environments.base import register_env

from robots.robocasa.robot_env import _REACH_STRUCTURAL_BASE_CLASSES

# Absolute path to robocasa's objaverse asset root. Used to compose
# fixed-mesh ``obj_groups`` strings for objects that need to pin to a
# single variant (e.g. ``"mug"`` — randomly sampling the 17 variants
# causes per-episode geometry variation that makes the grasp pipeline
# unreliable on some kitchens). When ``obj_groups`` ends in ``.xml``,
# robocasa's ``sample_kitchen_object_helper`` (in
# ``robocasa/models/objects/kitchen_object_utils.py``) takes the
# fast-path that pins the spawn to exactly that mesh — the category
# kwargs (scale, friction, ...) are still inherited from
# ``OBJ_CATEGORIES`` via reverse-lookup.
_ROBOCASA_OBJAVERSE_ROOT = os.path.join(
    robocasa.models.assets_root, "objects", "objaverse"
)


# ---------------------------------------------------------------------------
# Object registry
# ---------------------------------------------------------------------------
# Each entry drives this kitchen task's ``_get_obj_cfgs`` (``obj_groups`` +
# ``sample_region_kwargs`` + ``placement_size`` + ``placement_pos``).
#
# ``obj_groups`` values come from upstream robocasa's
# ``robocasa/models/objects/kitchen_objects.py``.
OBJECT_REACH_REGISTRY: dict[str, dict] = {
    "banana": {
        "obj_groups": "banana",
        "sample_region_kwargs": {},
        "placement_size": (0.40, 0.30),
        "placement_pos": (0.0, -1.0),
        "grasp_offset_xyz": (0.0, 0.0, 0.025),
        # --- Per-object placement biases (consumed by ObjectReachKitchen._reset_internal) ---
        # Per-seed world-XY (in robot base frame) FORCE-OVERRIDE for the
        # spawned object. Keyed on the scene seed. Banana-specific values
        # tuned by trial-and-error on the pretrain split before the
        # OBJECT_REACH wrapper supported target kitchens.
        "per_seed_spawn_xy": {
            1: (0.50, 0.0),
            6: (0.50, 0.0),
            7: (0.50, 0.0),
            8: (0.40, -0.15),  # in front of sink + 15cm right (avoid falling into sink)
        },
        # Per-target-kitchen world-XY (base frame) override, keyed on
        # ``layout_id`` (1..10). Takes precedence over per_seed_spawn_xy.
        "per_kitchen_spawn_xy": {
            1: (0.40, -0.15),
            2: (0.40, -0.30),
            3: (0.40, -0.15),
        },
        # Per-target-kitchen world-frame quaternion (wxyz) override for
        # the spawned object, applied AFTER the spawn-XY and fixed-yaw
        # overrides. Useful for kitchens where the default fixed yaw
        # leaves the object in a pose that the OSC drive can't reach.
        "per_kitchen_spawn_quat_wxyz": {
            3: (0.99, -0.006, 0.076, -0.120),  # k4-stable quat applied to k3 to stop yaw drift
        },
        # --- Per-object approach biases (consumed by
        # ObjectReachBlockingRoboCasaEnv._grasp_object_topdown) ---
        # Baseline lateral approach/descend offset (m) in the robot
        # base frame, added to the object's XY before the top-down
        # grasp targets are built. Biases the EEF site away from the
        # wrist so closing fingertips land on the object center rather
        # than the near edge. Default (0.04, 0, 0) shifts +4 cm along
        # the robot base +X (forward) — empirically right for a banana
        # lying flat with the EEF approaching from the wrist side.
        "approach_offset_base_xyz": (0.04, 0.0, 0.0),
        # Per-target-kitchen additive delta on top of approach_offset_base_xyz.
        # Takes precedence over per_seed_approach_offset_delta.
        "per_kitchen_approach_offset_delta": {
            2: (-0.04, 0.0, 0.0),
            6: (-0.05, 0.0, 0.0),
            9: (-0.05, 0.0, 0.0),
        },
        # Per-seed additive delta on top of approach_offset_base_xyz.
        "per_seed_approach_offset_delta": {
            1: (-0.08, 0.0, 0.0),
            4: (-0.08, 0.0, 0.0),
        },
    },
    "can": {
        # Soda-can style: upright cylinder, ~6.9 cm dia × ~11.1 cm tall
        # (objaverse/can, 14 variants in robocasa/models/assets/objects/
        # objaverse/can/). Object center is ~5.5 cm above the counter
        # surface so the grasp pipeline can aim at object center (no
        # ``grasp_offset_xyz`` z compensation needed).
        "obj_groups": "can",
        # Pin to ONE specific variant (``can_11``, identified as the
        # variant the random sampler happened to pick on k4 — where
        # the grasp pipeline works reliably). Empirical reporting
        # showed that the random sampling produced kitchens with
        # variant-specific grasp failures (off-center approach, slip
        # during lift) on most other kitchens. Pinning to a single
        # known-good variant eliminates the per-episode geometry
        # variance and lets the same approach/descend parameters
        # work across all kitchens. Pinning mechanism mirrors the
        # mug entry — see :data:`_ROBOCASA_OBJAVERSE_ROOT` and
        # ``sample_kitchen_object_helper`` in robocasa's
        # ``kitchen_object_utils.py:361``.
        "obj_variant_mjcf": os.path.join(_ROBOCASA_OBJAVERSE_ROOT, "can", "can_11", "model.xml"),
        "sample_region_kwargs": {},
        # Tighter sampling region than banana — can footprint is 6.9 cm
        # vs. banana's 12.5 cm, so a 30×30 cm region keeps the spawn
        # well away from the counter edges on every kitchen.
        "placement_size": (0.30, 0.30),
        # Same robot-facing bias as banana.
        "placement_pos": (0.0, -1.0),
        # Aim grasp at object center (cylinder, EEF site is between the
        # fingertips so no lateral offset needed). The Z value here is
        # IGNORED by the current grasp pipeline (descend phase derives
        # Z from ``obj_z - descend_depth_m`` with physics
        # stopping contact); kept (0, 0, 0) for clarity.
        "grasp_offset_xyz": (0.0, 0.0, 0.0),
        # Descend slightly ABOVE obj center: wrist target = obj_z + 0.01 m.
        # With ``descend_depth_m=0`` the OSC was still pushing the wrist
        # ~0.5 cm down against the counter (target = obj_z = counter+5.5 cm,
        # wrist physically stops at counter + finger_length ~6 cm → 1 cm
        # downward residual push). At ``-0.01`` (target = counter + 6.5 cm)
        # the OSC pushes UP slightly, so the wrist hovers and the
        # fingertips lightly graze the can body during close — no
        # destabilising press.
        "descend_depth_m": -0.01,
        "grasp_finger_offset_xyz": (0.0, 0.0, 0.0),
        # Bumped tangential friction further (was 1.5 then 3.0 then 5.0)
        # to 10.0. Asset default in objaverse/can/*/model.xml is
        # (0.95, 0.3, 0.1). User reported t5 still slipping heavily at
        # friction=(5.0, 2.0, 0.5); bumping more aggressively here.
        # Tuple: (μ_tangential, μ_torsional, μ_rolling).
        "geom_friction": (10.0, 3.0, 0.8),
        # K5 needs even more friction — user reports the can still
        # slips out of the gripper after lift on k5 specifically.
        # Empirical 5x bump on top of the baseline. Per-kitchen value
        # takes precedence over ``geom_friction`` for that layout.
        "per_kitchen_geom_friction": {
            5: (50.0, 15.0, 4.0),
        },
        # Also bump the gripper finger pad friction (asset default is
        # (2.0, 0.05, 0.0001) on the Panda); MuJoCo per-pair contact
        # friction is bottlenecked by the weaker side for torsional /
        # rolling. Without this, raising the can's friction past 2.0
        # gave diminishing returns; user reported persistent slip on
        # k5 at object friction 10.0. (10.0, 1.0, 0.5) lifts the
        # torsional + rolling caps so the can stays put during lift.
        "gripper_pad_friction": (10.0, 1.0, 0.5),
        # K5-specific bump on the gripper pads too — paired with the
        # k5 object-friction bump above.
        "per_kitchen_gripper_pad_friction": {
            5: (50.0, 5.0, 2.5),
        },
        # --- Per-object placement biases ---
        "per_seed_spawn_xy": {},
        # Reachability overrides: on k1/k2/k3 the default robocasa sampler
        # lands the object out of arm reach. k1/k2 use the banana
        # values verbatim; k3 needed an empirical retune — the banana
        # value (0.40, -0.15) places the upright can near the counter
        # edge where the descend phase topples it into the sink
        # (post-failure z ≈ 0.77 m, ~25 cm below counter top).
        # (0.30, -0.15) keeps the can well clear of the sink edge.
        # Reachability overrides: k1/k2/k3 land objects out of reach
        # without an XY override. Copied from the banana entry. (The
        # can needed an empirical retune for k3 because the mug-tall
        # upright geometry hit the sink edge with the banana value;
        # the mug works fine with the banana value.)
        "per_kitchen_spawn_xy": {
            1: (0.25, 0.05),  # CAN-specific: prior (0.30,-0.15) and (0.40,-0.15) both put can in k1's sink corner; (0.25, 0.05) lands on counter
            2: (0.40, -0.30),
            3: (0.30, -0.15),  # CAN-specific: keeps upright can clear of sink edge
        },
        "per_kitchen_spawn_quat_wxyz": {},
        # No baseline approach bias: can body is cylindrically symmetric
        # from above and the EEF site is already between the fingertips.
        # The banana's +4 cm forward bias compensates for its flat-lying
        # wrist-side edge — not applicable to the upright can.
        "approach_offset_base_xyz": (0.0, 0.0, 0.0),
        "per_kitchen_approach_offset_delta": {},
        "per_seed_approach_offset_delta": {},
    },
    "mug": {
        # Ceramic coffee mug: bimodal upright cylinder + handle.
        #   Short / wide variants (mug_7, mug_10, …): full height
        #     ~5.8–6.3 cm, center ~3.0 cm above counter; z_half ≈ 0.030 m.
        #   Tall variants (mug_1, mug_5, mug_18, …): full height
        #     ~8.7–9.6 cm, center ~4.3–4.8 cm above counter; z_half ≈ 0.045 m.
        # 17 active objaverse variants in
        #   robocasa/models/assets/objects/objaverse/mug/
        # (``mug_0`` excluded as "too dark"; ``mug_6`` absent from the
        # asset directory). The mug has a rigid mesh handle baked into
        # the body geometry (no joint); the handle protrudes in the +Y
        # body frame direction and is the reason the bbox Y extent is
        # always wider than the X extent (10.5–12.6 cm body+handle vs.
        # 7.4–11.1 cm in X).
        "obj_groups": "mug",
        # Pin to ONE specific variant (``mug_1``, a tall variant ~9 cm
        # tall, ~9 cm wide body + handle). The default obj_groups
        # behavior samples one of 17 variants per episode, which causes
        # per-episode geometry variation (handle orientation, height
        # cluster) that makes the grasp pipeline behave inconsistently
        # across kitchens. The user observed that random sampling
        # produces malformed grasps on k5 (presumably a degenerate
        # variant in the random pool); pinning to a single known-good
        # variant makes the grasp deterministic and per-kitchen
        # behaviour easier to reason about.
        #
        # The pinning mechanism: ``sample_kitchen_object_helper`` in
        # ``robocasa/models/objects/kitchen_object_utils.py:361`` takes
        # a fast-path when ``obj_groups`` (resolved at lookup time)
        # ends in ``.xml`` — it pins to exactly that mesh and inherits
        # the category kwargs (scale / friction / etc.) from
        # ``OBJ_CATEGORIES["mug"]`` via reverse-lookup.
        "obj_variant_mjcf": os.path.join(_ROBOCASA_OBJAVERSE_ROOT, "mug", "mug_1", "model.xml"),
        "sample_region_kwargs": {},
        # Mug footprint 8–13 cm; 30 × 30 cm sampling region matches the
        # can and keeps the spawn clear of counter edges on every kitchen.
        "placement_size": (0.30, 0.30),
        # Same robot-facing edge bias as banana + can.
        "placement_pos": (0.0, -1.0),
        # Z slot is ignored by the descend phase (physics-contact-stops);
        # body center ≈ bbox center for both height clusters (handle
        # Y-offset 5–18 mm is negligible at EEF-site precision). Mug is
        # cylindrically symmetric viewed from above, so no lateral bias
        # like the banana's 2.5 cm Z compensation.
        "grasp_offset_xyz": (0.0, 0.0, 0.0),
        # No world-frame finger offset — the per-mug grasp position is
        # biased in BASE FRAME via ``approach_offset_base_xyz`` below
        # (3 cm forward + 3 cm left), per user feedback that the
        # previous world-frame offset was in the wrong direction.
        "grasp_finger_offset_xyz": (0.0, 0.0, 0.0),
        # --- Per-object placement biases ---
        "per_seed_spawn_xy": {},
        # Reachability overrides: on k1/k2/k3 the default robocasa sampler
        # lands the can in a corner / behind a fixture / out of arm
        # reach. The same XY base-frame positions that fix the banana
        # also fix the can for k1/k2. K3 needed a MUG-SPECIFIC retune:
        # banana's (0.40, -0.15) is right at k3's sink edge, and the
        # descend phase's lateral push knocks the upright mug into the
        # sink (100% fall in real deployment). (0.30, -0.30) places
        # the mug deeper into the counter, well clear of the sink
        # edge — grasp pipeline succeeds and the carry holds.
        "per_kitchen_spawn_xy": {
            1: (0.40, -0.15),
            2: (0.40, -0.30),
            3: (0.30, -0.30),  # MUG-specific: away from k3 sink edge
        },
        "per_kitchen_spawn_quat_wxyz": {},
        # Base-frame approach shift for the mug: +3 cm in robot base X
        # (forward) and +3 cm in robot base Y (robot's left). Per user
        # feedback on the prior round of videos: the world-frame
        # finger-offset attempt put the wrist on the wrong side of the
        # mug; expressing the shift in BASE frame lets the wrapper
        # rotate it into world via ``R_wb`` on every kitchen so the
        # "left + forward" direction is consistent in robot-relative
        # terms regardless of how the robot is oriented in the kitchen.
        "approach_offset_base_xyz": (0.03, 0.03, 0.0),
        # Per-kitchen base-frame approach deltas, added on top of
        # ``approach_offset_base_xyz``. Per user feedback on the
        # 3 cm-forward + 3 cm-left round of videos:
        #   * k2, k9 still need to be moved 4 cm BACK (−X) + 3 cm
        #     more LEFT (+Y) → delta (−0.04, +0.03, 0)
        #     → total approach = (−0.01, +0.06, 0) on these layouts
        #   * k6 needs to be moved 4 cm RIGHT (−Y) → delta
        #     (0, −0.04, 0) → total approach = (+0.03, −0.01, 0)
        "per_kitchen_approach_offset_delta": {
            2: (-0.04, 0.03, 0.0),
            6: (0.0, -0.04, 0.0),
            9: (-0.04, 0.03, 0.0),
        },
        # Mug-side friction override, matching the can: asset default
        # is (0.95, 0.3, 0.1); user reported t5 grasp was good but
        # slipped heavily on lift. Same (5.0, 2.0, 0.5) tuning the can
        # uses (the can needed a further bump to 10.0 on top of this).
        # Mug-side friction override, matching the can: asset default
        # is (0.95, 0.3, 0.1); user reported t5 grasp was good but
        # slipped heavily on lift. Earlier we bumped the baseline to
        # the can's value (10.0, 3.0, 0.8), but that doubled friction
        # caused mug k3 grasps to FAIL (the higher friction on the
        # mug-finger contact during the open-gripper approach knocks
        # the mug off the counter). Reverting to (5.0, 2.0, 0.5).
        "geom_friction": (5.0, 2.0, 0.5),
        "per_kitchen_geom_friction": {
            5: (50.0, 15.0, 4.0),
        },
        # Same gripper-pad friction bump as the can. See the can entry
        # for the rationale (MuJoCo per-pair friction is bottlenecked
        # by the weaker side; the Panda pads' default torsional/rolling
        # of (0.05, 0.0001) caps the effective contact friction).
        "gripper_pad_friction": (10.0, 1.0, 0.5),
        "per_kitchen_gripper_pad_friction": {
            5: (50.0, 5.0, 2.5),
        },
        "per_seed_approach_offset_delta": {},
    },
}


# ---------------------------------------------------------------------------
# Fixed world-frame yaw applied to the spawned object after reset.
# ---------------------------------------------------------------------------
# Rationale: the placement-sampler ``rotation=(...)`` cfg key applies its
# angle in the *counter-relative* frame (it is composed with the counter's
# ``reference_rot`` before being written to ``body_xquat``). A cfg of
# ``rotation=-pi/6`` therefore yields a world yaw of
# ``counter_reference_rot + (-pi/6)``, which depends on the layout / seed
# (e.g. counter at ``-pi/2`` -> world yaw ``-2*pi/3``, not ``-pi/6``).
#
# To pin the banana to a deterministic world-frame yaw of ``-pi/6`` (30 deg
# clockwise around +z), we override the free joint's quaternion at the end
# of ``_reset_internal`` (after the upstream placement loop + settling has
# run).
_OBJECT_REACH_FIXED_WORLD_YAW: float = -math.pi / 6


class ObjectReachKitchen(Kitchen):
    """Empty-kitchen task with a single graspable object on a counter.

    Subclasses :class:`robocasa.environments.kitchen.kitchen.Kitchen` directly
    (whose default ``_get_obj_cfgs`` returns ``[]``, i.e. an empty kitchen)
    and adds exactly ONE free object via ``_get_obj_cfgs``. The object body
    name is fixed to ``"obj"`` so consumers can look it up via
    ``env.obj_body_id["obj"]`` regardless of which registry entry is active.

    Reward / success are intentionally inert: the wrapper drives task logic.

    Args:
        object_name: Key into :data:`OBJECT_REACH_REGISTRY`. Default
            ``"banana"``. Stored on the instance and consulted lazily by
            ``_get_obj_cfgs`` so a single class can serve every registered
            object.
    """

    DEFAULT_OBJECT_NAME = "banana"

    def __init__(self, object_name: str | None = None, *args, **kwargs):
        # ``object_name`` defaults to None and is resolved at CALL TIME
        # to ``type(self).DEFAULT_OBJECT_NAME``. The wrapper
        # ``ObjectReachBlockingRoboCasaEnv`` mutates the class attribute
        # to propagate ``object_reach_object_name`` through robosuite.make
        # (which constructs the kitchen without passing object_name
        # explicitly). A literal default like ``= DEFAULT_OBJECT_NAME``
        # would NOT pick up subsequent class-attr mutations because
        # Python freezes function default values at definition time
        # (``__init__.__defaults__`` would forever be ``("banana",)``
        # regardless of class-attr edits).
        if object_name is None:
            object_name = type(self).DEFAULT_OBJECT_NAME
        if object_name not in OBJECT_REACH_REGISTRY:
            raise KeyError(
                f"object_name={object_name!r} not in OBJECT_REACH_REGISTRY; "
                f"available keys: {sorted(OBJECT_REACH_REGISTRY)}"
            )
        self.object_name = str(object_name)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """Pin a non-corner counter so the placement_initializer has a target.

        Mirrors the minimal-kitchen pattern used by
        :class:`robocasa.environments.kitchen.atomic.kitchen_pick_place.CheesyBread`
        (no reference fixture; just any reachable, non-corner counter).
        """
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER_NON_CORNER, size=(0.6, 0.6)),
        )
        self.init_robot_base_ref = self.counter

    def _get_obj_cfgs(self):
        spec = OBJECT_REACH_REGISTRY[self.object_name]
        # If the spec has ``obj_variant_mjcf``, pin the spawn to that
        # exact mesh. ``sample_kitchen_object_helper`` in robocasa
        # (kitchen_object_utils.py:361) takes a fast-path when
        # ``obj_groups`` ends in ``.xml``, looks up the category kwargs
        # via reverse-mjcf-lookup, and spawns exactly the requested
        # variant — bypassing the per-episode random sampling that
        # would otherwise pick from the full category.
        obj_groups_spec = spec.get("obj_variant_mjcf") or spec["obj_groups"]
        cfg = dict(
            name="obj",
            obj_groups=obj_groups_spec,
            graspable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(spec.get("sample_region_kwargs", {})),
                size=tuple(spec["placement_size"]),
                pos=tuple(spec["placement_pos"]),
            ),
        )
        return [cfg]

    def _reset_internal(self):
        """Pin the spawned object's world-frame yaw to a fixed value, apply
        the optional per-seed xy override, and hide every non-``"obj"`` scene
        object so the workspace contains only the banana.

        The placement-sampler ``rotation`` cfg key applies its angle in the
        counter-relative frame, so it cannot be used to deterministically set
        a world-frame yaw across kitchen layouts. We override the free
        joint's quaternion to a fixed yaw of
        :data:`_OBJECT_REACH_FIXED_WORLD_YAW` after the upstream placement
        loop + settling has run.

        When the active scene seed is present in
        :attr:`_OBJECT_REACH_PER_SEED_BANANA_BASE_FRAME_XY`, the banana's
        free-joint xy is force-overridden to ``base_pos[:2] + R_wb[:2,:2] @
        (x_base, y_base)`` (the dict value is an absolute position in the
        robot base frame, not a delta).

        Finally, every entry in ``self.objects`` whose name is NOT ``"obj"``
        has every geom in its body subtree hidden — alpha → 0, contype →
        0, conaffinity → 0 — mirroring the
        :meth:`RoboCasaEnv._hide_nearby_appliances` pattern. Keeps the
        banana as the only visible / collidable scene object regardless of
        what the upstream kitchen task added (containers, auxiliaries, …).
        """
        super()._reset_internal()

        if "obj" not in self.objects:
            return
        obj = self.objects["obj"]
        joint_name = obj.joints[0]

        # ── Per-object geom friction override ─────────────────────────
        # Applied post-compile via direct write to ``sim.model.geom_friction``
        # (an in-memory numpy array backed by ``mjModel.geom_friction``;
        # MuJoCo permits writes at any time and uses the new values on
        # every subsequent ``mj_step`` / ``mj_forward``). Only collision
        # geoms (``geom_contype != 0``) are mutated; visual + region
        # geoms (visual mesh + ``reg_bbox``) are untouched. The override
        # fires on every rebuild (every ``robosuite.make`` call) so it
        # composes naturally with ``force_rebuild=True`` resets.
        # Registry fields:
        #   * ``geom_friction = (mu_t, mu_torsional, mu_rolling)`` —
        #     baseline applied on every kitchen.
        #   * ``per_kitchen_geom_friction = {layout_id: (...)}`` —
        #     per-layout override that TAKES PRECEDENCE over baseline.
        #     Used for kitchens where the baseline isn't enough to
        #     prevent slip (e.g. mug + can on k5 still slipped at
        #     object friction 10.0; per-kitchen value 50.0 added).
        # Default (None) leaves the asset's friction untouched.
        spec = OBJECT_REACH_REGISTRY[self.object_name]
        layout_id_int = getattr(self, "layout_id", None)
        per_kitchen_friction = spec.get("per_kitchen_geom_friction", {}).get(
            layout_id_int
        )
        geom_friction_override = (
            per_kitchen_friction
            if per_kitchen_friction is not None
            else spec.get("geom_friction", None)
        )
        if geom_friction_override is not None:
            friction_arr = np.asarray(geom_friction_override, dtype=np.float64)
            model = self.sim.model
            try:
                root_bid = model.body_name2id(obj.root_body)
            except Exception:
                root_bid = None
            if root_bid is not None:
                for bid in self._body_subtree_ids(root_bid):
                    g0 = int(model.body_geomadr[bid])
                    ng = int(model.body_geomnum[bid])
                    for i in range(ng):
                        gid = g0 + i
                        if model.geom_contype[gid] != 0:
                            model.geom_friction[gid] = friction_arr

        # ── Gripper finger pad friction override ──────────────────────
        # The Panda gripper's finger-pad collision geoms ship with
        # ``friction = "2 0.05 0.0001"`` (tangential, torsional,
        # rolling). Even after bumping the OBJECT friction to 10.0+,
        # MuJoCo's per-pair contact friction caps at the LOWER side's
        # value for torsional/rolling, so the object still rolls out
        # of the gripper during lift on some kitchens (user reports
        # heavy slip on k5 for both can and mug). Bumping the finger
        # pad friction here raises the effective contact friction.
        # Geom names: ``gripper0_right_finger1_pad_collision`` and
        # ``gripper0_right_finger2_pad_collision`` (the robosuite
        # gripper-naming convention prefixes with ``gripper0_right_``).
        # Registry fields:
        #   * ``gripper_pad_friction = (...)`` — baseline.
        #   * ``per_kitchen_gripper_pad_friction = {layout_id: (...)}``
        #     — per-layout override (e.g. k5 needs higher than baseline).
        per_kitchen_pad = spec.get("per_kitchen_gripper_pad_friction", {}).get(
            layout_id_int
        )
        gripper_pad_friction_override = (
            per_kitchen_pad
            if per_kitchen_pad is not None
            else spec.get("gripper_pad_friction", None)
        )
        if gripper_pad_friction_override is not None:
            pad_friction_arr = np.asarray(
                gripper_pad_friction_override, dtype=np.float64
            )
            model = self.sim.model
            n_pads = 0
            for gid in range(model.ngeom):
                try:
                    gn = model.geom_id2name(gid) or ""
                except Exception:
                    gn = ""
                if "finger" in gn and "pad" in gn:
                    model.geom_friction[gid] = pad_friction_arr
                    n_pads += 1
            # Defensive: if no pads were found by name, the gripper
            # might be using a different naming convention; the
            # mutation is a no-op which preserves the asset default.

        # Joint qpos layout for a free joint: [x, y, z, qw, qx, qy, qz].
        qpos = np.array(self.sim.data.get_joint_qpos(joint_name)).copy()

        half = _OBJECT_REACH_FIXED_WORLD_YAW / 2.0
        qpos[3:7] = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])

        # Per-seed xy FORCE-OVERRIDE: replace world xy with
        # ``base_pos[:2] + R_wb[:2,:2] @ (x_base, y_base)``. The dict value
        # is an absolute position in the robot base frame. Per-target-kitchen
        # override (keyed on ``self.layout_id``) takes precedence over the
        # per-seed override when both are set. Both dicts are PER-OBJECT
        # and live in ``OBJECT_REACH_REGISTRY[self.object_name]`` so a new
        # object can supply its own spawn biases without touching the
        # kitchen class.
        seed = getattr(self, "seed", None)
        layout_id = getattr(self, "layout_id", None)
        # NB: ``spec`` already resolved above for the geom_friction override.
        per_target_xy = spec.get("per_kitchen_spawn_xy", {}).get(layout_id)
        per_seed_xy = spec.get("per_seed_spawn_xy", {}).get(seed)
        override_xy = per_target_xy if per_target_xy is not None else per_seed_xy
        if override_xy is not None:
            base_pos, R_wb = self._get_base_world_pose()
            xy_base = np.asarray(override_xy, dtype=np.float64).reshape(2)
            qpos[0:2] = base_pos[:2] + R_wb[:2, :2] @ xy_base

        self.sim.data.set_joint_qpos(joint_name, qpos)

        # Per-target-kitchen quat FORCE-OVERRIDE: replace the quat portion
        # (indices 3:7, wxyz) of the object free-joint qpos with the dict
        # value. Applied AFTER the xy/yaw write above, so it fully
        # supersedes the fixed-yaw quaternion for listed layouts. Also
        # PER-OBJECT (looked up via the same ``OBJECT_REACH_REGISTRY``
        # entry as the xy overrides above).
        per_target_quat = spec.get("per_kitchen_spawn_quat_wxyz", {}).get(layout_id)
        if per_target_quat is not None:
            qpos2 = np.array(self.sim.data.get_joint_qpos(joint_name)).copy()
            qpos2[3:7] = np.asarray(per_target_quat, dtype=np.float64).reshape(4)
            self.sim.data.set_joint_qpos(joint_name, qpos2)

        self.sim.forward()

        # Hide every non-banana scene object (alpha + contype + conaffinity)
        # and every non-structural fixture (toaster, blender, ...). Mirrors
        # RoboCasaEnv._hide_nearby_appliances but unconditional (no radius).
        self._hide_bodies(
            self.objects.items(),
            should_skip=lambda name, _obj: name == "obj",
        )
        fixtures = getattr(self, "fixtures", None) or {}
        self._hide_bodies(
            fixtures.items(),
            should_skip=lambda _name, fixture: any(
                base.__name__ in _REACH_STRUCTURAL_BASE_CLASSES
                for base in type(fixture).__mro__
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_base_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(base_pos_world, R_world_from_base)`` for the mobile base.

        Mirrors :meth:`RoboCasaEnv._get_base_world_pose` — tries the usual
        body names and falls back to ``(zeros(3), eye(3))`` if none resolve.
        """
        for body_name in ("mobilebase0_base", "robot0_base", "base"):
            try:
                bid = self.sim.model.body_name2id(body_name)
                pos = np.asarray(
                    self.sim.data.body_xpos[bid], dtype=np.float64
                ).copy()
                quat_wxyz = self.sim.data.body_xquat[bid]
                # mujoco wxyz → scipy xyzw.
                q = np.array(
                    [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
                    dtype=np.float64,
                )
                return pos, Rotation.from_quat(q).as_matrix()
            except Exception:
                continue
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)

    def _body_subtree_ids(self, root_bid: int) -> list:
        """Return ``[root_bid, *descendant_body_ids]`` for ``root_bid``.

        Mirrors :meth:`RoboCasaEnv._body_subtree_ids` so we can walk every
        body that belongs to a scene object and reach all of its geoms.
        """
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
        """Hide every body subtree in ``items`` that does NOT match ``should_skip``.

        For each ``(name, mj_obj_or_fixture)`` pair, looks up the entity's
        ``root_body``, walks its body subtree, and zeroes ``geom_rgba[:, 3]``
        (alpha), ``geom_contype`` and ``geom_conaffinity`` for every geom.
        Kinematic tree is left intact (no joint surgery). Mirrors
        :meth:`RoboCasaEnv._hide_nearby_appliances`.

        Args:
            items: Iterable of ``(name, entity)`` pairs (e.g.
                ``self.objects.items()`` or ``self.fixtures.items()``).
            should_skip: Callable ``(name, entity) -> bool``. When True,
                the entity is left visible / collidable.
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


# Expose under ``robosuite.make("ObjectReachKitchen", ...)``. This only
# registers the class; no env is instantiated at import time.
register_env(ObjectReachKitchen)
