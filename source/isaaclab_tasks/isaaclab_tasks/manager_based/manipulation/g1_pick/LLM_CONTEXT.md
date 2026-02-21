# G1 Picking Global Context

## Overview
This file contains the full source code for the G1 picking task.

## g1_pick_env_cfg.py
```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for G1 + Inspire-hand picking environment with curriculum learning."""

import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import CuboidCfg, SphereCfg, CapsuleCfg, RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from .robot_cfg import G1_INSPIRE_CFG
from . import mdp

##
# Height constants – keep in sync with scene geometry
#   table  size=(0.6, 1.2, 0.80), centre z=0.40  → top at 0.800, front edge at x=0.10
#   tray   size=(0.4, 0.60, 0.02), centre z=0.81  → top at 0.820
#   robot  pelvis x=-0.1, fix_base=True
##
_OBJ_INIT_Z = 0.850   # object centre resting on tray (tray top 0.820 + half-size 0.025 + gap 0.005)
_LIFT_Z     = 0.900   # must exceed this to count as "lifted" (10 cm above table top)
_DROP_Z     = 0.500   # below this → object fell off table → episode terminates

##
# Scene definition
##


@configclass
class G1PickSceneCfg(InteractiveSceneCfg):
    """Scene: G1 + Inspire hands, table, tray, target object, 3 distractors."""

    # Required so MultiAssetSpawnerCfg assigns different shapes per env instead of
    # copying env_0's PhysX mesh to all other environments.
    replicate_physics: bool = False

    # ── Robot ─────────────────────────────────────────────────────────────────
    robot: ArticulationCfg = G1_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ── Table – solid floor-to-surface block (like a real table) ─────────────
    # size=(0.6 depth, 1.2 width, 0.80 height); centre z=0.40 → top at 0.800 m.
    # Centre x=0.40; front edge at x=0.10 → 0.20 m clearance from robot pelvis at x=-0.10.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.4, 0.0, 0.40], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 1.2, 0.80),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.4, 0.3)),
        ),
    )

    # ── Tray (top at z=0.820 m) ───────────────────────────────────────────────
    tray = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tray",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.4, 0.0, 0.810], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.6, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        ),
    )

    # ── Target object (RED) ───────────────────────────────────────────────────
    # collision_props is required: MultiAssetSpawnerCfg inherits RigidObjectSpawnerCfg
    # but does NOT enable CollisionAPI by default for primitive shapes.
    target_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                CuboidCfg(
                    size=(0.05, 0.05, 0.05),
                    physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                SphereCfg(
                    radius=0.03,
                    physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                # CapsuleCfg(
                #     radius=0.02,
                #     height=0.06,
                #     physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
                #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                # ),
            ],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, _OBJ_INIT_Z]),
    )

    # ── Distractor objects (BLUE, GREEN, YELLOW) ──────────────────────────────
    # Hidden by default (z = -5 m); reset_clutter_based_on_difficulty moves them
    # onto the tray when curriculum difficulty is high enough.
    distractor_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Distractor0",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.2, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, -5.0]),
    )
    distractor_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Distractor1",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.07, 0.04),
            physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, -5.0]),
    )
    distractor_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Distractor2",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.07),
            physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.7, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, -5.0]),
    )

    left_contact_sensor = None
    right_contact_sensor = None

    # ── Ground + lighting ─────────────────────────────────────────────────────
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            texture_file=f"{ISAACLAB_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """26-DOF action space: 14 arm DOF + 6 hand leader-joint DOF per hand.

    Inspire hand mimic joints (thumb_3/4, index/middle/ring/little _2) are
    excluded from the action space.  They are held at zero by the actuator's
    PD stiffness and reset to 0 each episode via freeze_mimic_joints event.
    """

    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            # ── Arms (7 per side = 14 total) ──────────────────────────────────
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
            # ── Hand leader joints (6 per side = 12 total) ────────────────────
            ".*_thumb_1_joint",
            ".*_thumb_2_joint",
            ".*_index_1_joint",
            ".*_middle_1_joint",
            ".*_ring_1_joint",
            ".*_little_1_joint",
        ],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot proprioception
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # Target object position in robot frame (privileged)
        target_object_position = ObsTerm(
            func=mdp.target_object_position_b,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("target_object")},
        )

        # Distractor positions in robot frame (needed for left-hand declutter)
        distractor_0_position = ObsTerm(
            func=mdp.target_object_position_b,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("distractor_0")},
        )
        distractor_1_position = ObsTerm(
            func=mdp.target_object_position_b,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("distractor_1")},
        )
        distractor_2_position = ObsTerm(
            func=mdp.target_object_position_b,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("distractor_2")},
        )

        # Fingertip positions in robot frame (privileged)
        left_fingertip_positions = ObsTerm(
            func=mdp.fingertip_positions_b,
            params={
                "robot_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_thumb_4", "left_index_2", "left_middle_2", "left_ring_2", "left_little_2"],
                )
            },
        )
        right_fingertip_positions = ObsTerm(
            func=mdp.fingertip_positions_b,
            params={
                "robot_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["right_thumb_4", "right_index_2", "right_middle_2", "right_ring_2", "right_little_2"],
                )
            },
        )

        # Last action
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Reset full scene to defaults first
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Randomise arm + hand leader joints slightly
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint", ".*_wrist_pitch_joint", ".*_wrist_yaw_joint",
                    ".*_thumb_1_joint", ".*_thumb_2_joint",
                    ".*_index_1_joint", ".*_middle_1_joint", ".*_ring_1_joint", ".*_little_1_joint",
                ],
            ),
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Freeze lower body to exact default (no randomisation, no velocity)
    freeze_lower_body = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint",
                    ".*_knee_joint",
                    ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
                    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                ],
            ),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Hold mimic joints (thumb_3/4, finger _2 joints) at zero
    freeze_mimic_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_thumb_3_joint", ".*_thumb_4_joint",
                    ".*_index_2_joint", ".*_middle_2_joint",
                    ".*_ring_2_joint", ".*_little_2_joint",
                ],
            ),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Reset target object onto tray
    # reset_root_state_uniform: new_pos = default_pos + env_origin + rand_sample
    # default_pos = (0.5, 0.0, 0.850); rand z=(0.01,0.01) → lands 1 cm above tray
    reset_target_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.14, 0.14), "y": (-0.18, 0.18), "z": (0.01, 0.01)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("target_object"),
        },
    )

    # Reset distractors based on curriculum difficulty
    reset_clutter = EventTerm(
        func=mdp.reset_clutter_based_on_difficulty,
        mode="reset",
        params={
            "distractor_names": ["distractor_0", "distractor_1", "distractor_2"],
            "tray_surface_height": _OBJ_INIT_Z,
            "hidden_height": -5.0,
            "tray_x_half": 0.14,
            "tray_y_half": 0.17,
            "table_center_x": 0.4,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Phase gating summary (managed by PickingCurriculumScheduler):
      Phase 0  →  reaching_target + left_penalty (always on)
      Phase 1  →  + lifting_target + declutter
      Phase 2  →  + pick_success

    Terms that start at weight=0 are enabled by the curriculum at runtime.
    """

    # ── Phase 0: RIGHT hand reaches target ────────────────────────────────────
    # Only RIGHT fingertips are included in body_names so the left hand is free
    # to learn the declutter task without conflicting reaching gradients.
    reaching_target = RewTerm(
        func=mdp.target_object_reaching_reward,
        params={
            "std": 0.1,
            "robot_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "right_thumb_4", "right_index_2", "right_middle_2",
                    "right_ring_2", "right_little_2",
                ],
            ),
            "object_cfg": SceneEntityCfg("target_object"),
        },
        weight=1.0,
    )

    # Always active: soft penalty when LEFT hand drifts toward the target.
    # std=0.15 m gives a gentle gradient rather than a hard boundary.
    left_penalty = RewTerm(
        func=mdp.left_hand_near_target_penalty,
        params={
            "std": 0.15,
            "robot_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_thumb_4", "left_index_2", "left_middle_2",
                    "left_ring_2", "left_little_2",
                ],
            ),
            "object_cfg": SceneEntityCfg("target_object"),
        },
        weight=0.3,  # small, just enough to break symmetry
    )

    # ── Phase 1: RIGHT hand grasps, LEFT hand declutters ─────────────────────
    # weight=0 → curriculum enables this at phase 1 (target weight 3.0)
    grasping_target = None

    # weight=0 → curriculum enables this at phase 1 (target weight 5.0)
    lifting_target = RewTerm(
        func=mdp.target_object_lift_reward,
        params={"minimal_height": _LIFT_Z, "object_cfg": SceneEntityCfg("target_object")},
        weight=0.0,
    )

    # weight=0 → curriculum enables this at phase 1 (target weight 2.0)
    # LEFT hand fingertips in body_names; distractors passed by name.
    declutter = RewTerm(
        func=mdp.left_hand_declutter_reward,
        params={
            "std": 0.12,
            "robot_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_thumb_4", "left_index_2", "left_middle_2",
                    "left_ring_2", "left_little_2",
                ],
            ),
            "distractor_names": ["distractor_0", "distractor_1", "distractor_2"],
            "target_cfg": SceneEntityCfg("target_object"),
        },
        weight=0.0,
    )

    # ── Phase 2: pick-success bonus ───────────────────────────────────────────
    # weight=0 → curriculum enables this at phase 2 (target weight 1.0)
    pick_success = RewTerm(
        func=mdp.pick_success_reward,
        params={
            "minimal_height": _LIFT_Z,
            "hold_time_threshold": 1.0,
            "object_cfg": SceneEntityCfg("target_object"),
        },
        weight=0.0,
    )

    # ── Action smoothness penalties (always active) ───────────────────────────
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate when object falls well below the table top
    target_dropped = DoneTerm(
        func=mdp.target_object_dropped,
        params={"minimum_height": _DROP_Z, "object_cfg": SceneEntityCfg("target_object")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    picking_curriculum = CurrTerm(
        func=mdp.PickingCurriculumScheduler,
        params={
            # ── Clutter difficulty ────────────────────────────────────────────
            "object_cfg": SceneEntityCfg("target_object"),
            "lift_height_threshold": _LIFT_Z,
            "init_difficulty": 0,
            "min_difficulty": 0,
            "max_difficulty": 60,
            "promotion_only": False,
            # ── Phase gating ──────────────────────────────────────────────────
            # Rolling window size (number of completed episodes tracked).
            "history_size": 500,
            # Phase 0→1: mean weighted reaching-reward/step must exceed this.
            # reaching weight=1.0, max raw reward/step=1.0 →
            #   0.5 means hand is within ~5 cm of target for >50% of each episode.
            "phase1_reaching_threshold": 0.5,
            # Phase 1→2: mean weighted grasping-reward/step must exceed this.
            # grasping weight=3.0, max raw reward/step=3.0 (binary×3) →
            #   0.75 means right hand in contact ~25% of episode steps.
            "phase2_lifting_threshold": 0.75,
            # Reward weights to enable at each phase transition.
            "phase1_terms": {"lifting_target": 5.0, "declutter": 2.0},
            "phase2_terms": {"pick_success": 1.0},
        },
    )


##
# Environment configuration
##


@configclass
class G1PickEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 + Inspire-hand picking environment."""

    scene: G1PickSceneCfg = G1PickSceneCfg(num_envs=4096, env_spacing=3.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 10.0

        self.sim.dt = 1 / 120  # 120 Hz physics
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15

        # ── Freeze lower body ────────────────────────────────────────────────
        # deepcopy prevents mutation of the global G1_INSPIRE_CFG.
        # Raise effort_limit_sim so the PD controller can always generate enough
        # torque to hold the joints rigid against gravity.
        legs_act = copy.deepcopy(self.scene.robot.actuators["legs"])
        legs_act.stiffness = 10000.0
        legs_act.damping = 1000.0
        legs_act.effort_limit_sim = 10000.0
        self.scene.robot.actuators["legs"] = legs_act

        feet_act = copy.deepcopy(self.scene.robot.actuators["feet"])
        feet_act.stiffness = 10000.0
        feet_act.damping = 1000.0
        feet_act.effort_limit_sim = 10000.0
        self.scene.robot.actuators["feet"] = feet_act


@configclass
class G1PickEnvCfg_PLAY(G1PickEnvCfg):
    """Play/evaluation config: fewer environments, start at max difficulty."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.curriculum.picking_curriculum.params["init_difficulty"] = 60

```

## robot_cfg.py
```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 29-DOF humanoid with Inspire FTP dexterous hands.

Robot highlights:
- 12 leg DOF (hip yaw/roll/pitch, knee, ankle pitch/roll × 2)
- 3 waist DOF (yaw, roll, pitch)
- 14 arm DOF (shoulder 3DOF + elbow 1DOF + wrist 3DOF × 2)
- 24 hand DOF (Inspire FTP: thumb ×4, index/middle/ring/little ×2 each × 2 hands)

URDF issues noted:
- Mimic joints (thumb_3/4 mimic thumb_2/3; finger _2 joints mimic _1): PhysX does NOT
  support <mimic> tags; all joints are treated as independent DOFs in simulation.
  In real deployment, the hardware's mechanical coupling enforces the mimic constraint.
- Mesh files must exist at
  /home/daatsi-aeres/ARCLab_ws/unitree_ros/robots/g1_description/meshes/ (160 STL files).
  The URDF is loaded from that directory so relative mesh paths resolve correctly.
- Four sensor-only links (imu_in_torso, imu_in_pelvis, d435_link, mid360_link) have no
  inertial definition — this is intentional (massless sensor frames).
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg

# URDF must live next to its meshes/ folder so relative paths resolve
_URDF_PATH = (
    "/home/daatsi-aeres/ARCLab_ws/unitree_ros/robots/g1_description/"
    "g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"
)

G1_INSPIRE_CFG = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path=_URDF_PATH,
        activate_contact_sensors=True,
        fix_base=True,
        merge_fixed_joints=True,   # merges force-sensor fixed links → fewer bodies, faster sim
        self_collision=False,
        make_instanceable=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        # joint_drive=None: use URDF-parsed PD values for USD conversion.
        # IsaacLab ImplicitActuatorCfg overrides these at runtime anyway.
        joint_drive=None,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Pelvis fixed at x=-0.1 (fix_base=True).
        # Table front edge at x=0.1 (table centre 0.4, half-depth 0.3) → 0.2 m clearance.
        # Tray centre at x=0.4 → arms need ~0.5 m forward reach, achieved with high
        # shoulder_pitch + bent elbow so hands hover just above the tray surface.
        pos=(-0.1, 0.0, 0.74),
        joint_pos={
            # ── Legs: standard crouched-standing pose (cosmetic, base is fixed) ─
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            # ── Waist: neutral ─────────────────────────────────────────────────
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            # ── Arms: hands pre-positioned over tray ──────────────────────────
            # G1 elbow convention: NEGATIVE = forward (toward table), POSITIVE = backward.
            # Joint range: -1.05 to +2.09 rad.
            # shoulder_pitch =  1.0 rad → arms swing ~57° forward
            # shoulder_roll  = ±0.1 rad → slight inward to center over tray
            # elbow          = -0.8 rad → forward reach toward table; 0.25 rad from limit
            # wrist_pitch    = -0.8 rad → palms face down toward objects
            ".*_shoulder_pitch_joint": -0.5,
            "left_shoulder_roll_joint": -0.1,
            "right_shoulder_roll_joint": 0.1,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.3,
            ".*_wrist_roll_joint": 0.0,
            ".*_wrist_pitch_joint": -0.8,
            ".*_wrist_yaw_joint": 0.0,
            # ── Inspire hands: fully open ──────────────────────────────────────
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # ── Lower body (frozen during manipulation) ────────────────────────────
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=300,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "waist_.*_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "waist_.*_joint": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=20,
            stiffness=20.0,
            damping=2.0,
        ),
        # ── Arms (controlled) ─────────────────────────────────────────────────
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=25,
            stiffness=40.0,
            damping=10.0,
        ),
        # ── Inspire hands (controlled) ────────────────────────────────────────
        # All 12 joints per hand are in the actuator group.
        # The policy controls only the 6 leader joints per hand (thumb_1/2, index/middle/ring/little_1).
        # The 6 mimic joints (thumb_3/4, *_2 fingers) are excluded from the action space but
        # are still actuated; they are reset to 0 each episode and held there by stiffness.
        "left_hand": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_thumb_1_joint", "left_thumb_2_joint",
                "left_thumb_3_joint", "left_thumb_4_joint",
                "left_index_1_joint", "left_index_2_joint",
                "left_middle_1_joint", "left_middle_2_joint",
                "left_ring_1_joint", "left_ring_2_joint",
                "left_little_1_joint", "left_little_2_joint",
            ],
            effort_limit_sim=10,
            stiffness=20.0,
            damping=2.0,
        ),
        "right_hand": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_thumb_1_joint", "right_thumb_2_joint",
                "right_thumb_3_joint", "right_thumb_4_joint",
                "right_index_1_joint", "right_index_2_joint",
                "right_middle_1_joint", "right_middle_2_joint",
                "right_ring_1_joint", "right_ring_2_joint",
                "right_little_1_joint", "right_little_2_joint",
            ],
            effort_limit_sim=10,
            stiffness=20.0,
            damping=2.0,
        ),
    },
)
"""G1 29-DOF humanoid with Inspire FTP dexterous hands (loaded from URDF)."""

```

## mdp/curriculum.py
```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum learning functions for G1 picking environment."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PickingCurriculumScheduler(ManagerTermBase):
    """Adaptive difficulty scheduler with three-phase reward gating.

    Phases
    ------
    Phase 0 – Reaching only.
        The agent learns to move the RIGHT hand toward the target.
        Grasping / lifting / declutter / pick-success weights are all 0.

    Phase 1 – Grasping + Lifting + Declutter unlocked.
        Triggered when the rolling mean *reaching* reward per step exceeds
        ``phase1_reaching_threshold``.  The reward manager weights for
        "grasping_target", "lifting_target", and "declutter" are set to their
        configured values.

    Phase 2 – Pick-success unlocked.
        Triggered when the rolling mean *lifting* reward per step exceeds
        ``phase2_lifting_threshold``.  The "pick_success" weight is enabled.

    Difficulty tracks clutter count independently of phase and continues to
    ramp up throughout training.

    Curriculum params (all optional, tunable via CurriculumCfg)
    -----------------------------------------------------------
    init_difficulty          : int   = 0
    min_difficulty           : int   = 0
    max_difficulty           : int   = 60
    promotion_only           : bool  = False
    history_size             : int   = 500   # rolling episode window
    phase1_reaching_threshold: float = 0.5   # weighted reward/step > this → phase 1
    phase2_lifting_threshold: float = 0.75  # weighted reward/step > this → phase 2
    phase1_terms             : dict  = {"lifting_target": 5.0,
                                        "declutter": 2.0}
    phase2_terms             : dict  = {"pick_success": 1.0}
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        # ── Difficulty (clutter) ──────────────────────────────────────────────
        init_difficulty = cfg.params.get("init_difficulty", 0)
        self.current_difficulties = torch.ones(env.num_envs, device=env.device) * init_difficulty
        self.difficulty_frac = 0.0

        # ── Phase management ─────────────────────────────────────────────────
        self._phase = 0
        history_size = cfg.params.get("history_size", 500)
        self._reaching_history: deque[float] = deque(maxlen=history_size)
        self._lifting_history: deque[float] = deque(maxlen=history_size)

        self._phase1_threshold = cfg.params.get("phase1_reaching_threshold", 0.5)
        self._phase2_threshold = cfg.params.get("phase2_lifting_threshold", 0.75)

        self._phase1_terms: dict[str, float] = cfg.params.get(
            "phase1_terms",
            {"lifting_target": 5.0, "declutter": 2.0},
        )
        self._phase2_terms: dict[str, float] = cfg.params.get(
            "phase2_terms",
            {"pick_success": 1.0},
        )

        # Reward-manager term indices resolved lazily on first __call__
        self._rm_initialized = False
        self._reaching_key: str | None = None
        self._lifting_key: str | None = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _init_rm(self, env: ManagerBasedRLEnv) -> None:
        """Resolve reward term names → indices once the reward manager is ready."""
        rm = env.reward_manager
        names = list(rm._term_names)

        # Keys for reading episode sums
        self._reaching_key = "reaching_target" if "reaching_target" in names else None
        self._lifting_key = "lifting_target" if "lifting_target" in names else None

        # Indices for modifying weights
        self._phase1_indices: dict[str, int] = {
            n: names.index(n) for n in self._phase1_terms if n in names
        }
        self._phase2_indices: dict[str, int] = {
            n: names.index(n) for n in self._phase2_terms if n in names
        }
        self._rm_initialized = True

    def _set_phase_weights(self, env: ManagerBasedRLEnv, phase: int) -> None:
        """Write reward weights for the given phase into the reward manager config."""
        rm = env.reward_manager
        if phase >= 1:
            for name, idx in self._phase1_indices.items():
                rm._term_cfgs[idx].weight = self._phase1_terms[name]
                print(f"[PickingCurriculum] Phase 1 → enabled reward '{name}' "
                      f"(weight={self._phase1_terms[name]})")
        if phase >= 2:
            for name, idx in self._phase2_indices.items():
                rm._term_cfgs[idx].weight = self._phase2_terms[name]
                print(f"[PickingCurriculum] Phase 2 → enabled reward '{name}' "
                      f"(weight={self._phase2_terms[name]})")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_state(self) -> torch.Tensor:
        return self.current_difficulties

    def set_state(self, state: torch.Tensor) -> None:
        self.current_difficulties = state.clone().to(self._env.device)

    @property
    def phase(self) -> int:
        return self._phase

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
        lift_height_threshold: float = 0.15,
        init_difficulty: int = 0,
        min_difficulty: int = 0,
        max_difficulty: int = 60,
        promotion_only: bool = False,
        # The remaining params are read from cfg.params in __init__ instead:
        history_size: int = 500,
        phase1_reaching_threshold: float = 0.5,
        phase2_lifting_threshold: float = 0.75,
        phase1_terms: dict | None = None,
        phase2_terms: dict | None = None,
    ):
        """Update difficulty and advance the training phase when ready.

        Args:
            env: The environment.
            env_ids: Environments that are resetting this step.
            object_cfg: Scene entity for the target object.
            lift_height_threshold: Height above env origin to count as a pick.
            min/max_difficulty: Clamp range for clutter difficulty.
            promotion_only: If True, difficulty never decreases.
            (remaining args) : Declared so IsaacLab doesn't treat them as unknown;
                               their values are consumed in __init__ instead.
        """
        # ── Lazy reward-manager init ──────────────────────────────────────────
        if not self._rm_initialized:
            try:
                self._init_rm(env)
            except Exception:
                pass  # reward manager not ready yet; retry next call

        # ── Phase tracking ────────────────────────────────────────────────────
        if self._rm_initialized and len(env_ids) > 0:
            rm = env.reward_manager
            max_ep_len = max(env.max_episode_length, 1)

            # Reaching: always track
            if self._reaching_key and self._reaching_key in rm._episode_sums:
                ep_sums = rm._episode_sums[self._reaching_key][env_ids]
                for v in (ep_sums / max_ep_len).tolist():
                    self._reaching_history.append(float(v))

            # Lifting: only meaningful after phase 1 (weight > 0)
            if self._phase >= 1 and self._lifting_key and self._lifting_key in rm._episode_sums:
                ep_sums = rm._episode_sums[self._lifting_key][env_ids]
                for v in (ep_sums / max_ep_len).tolist():
                    self._lifting_history.append(float(v))

            # ── Phase advancement ─────────────────────────────────────────────
            MIN_HISTORY = 50  # require at least this many completed episodes

            if self._phase == 0 and len(self._reaching_history) >= MIN_HISTORY:
                mean_reaching = sum(self._reaching_history) / len(self._reaching_history)
                if mean_reaching >= self._phase1_threshold:
                    self._phase = 1
                    self._set_phase_weights(env, 1)
                    print(f"[PickingCurriculum] *** PHASE 0→1 *** "
                          f"(mean reaching/step={mean_reaching:.3f} ≥ {self._phase1_threshold})")

            elif self._phase == 1 and len(self._lifting_history) >= MIN_HISTORY:
                mean_lifting = sum(self._lifting_history) / len(self._lifting_history)
                if mean_lifting >= self._phase2_threshold:
                    self._phase = 2
                    self._set_phase_weights(env, 2)
                    print(f"[PickingCurriculum] *** PHASE 1→2 *** "
                          f"(mean lifting/step={mean_lifting:.3f} ≥ {self._phase2_threshold})")

        # ── Clutter difficulty ────────────────────────────────────────────────
        target_object: RigidObject = env.scene[object_cfg.name]
        object_height = target_object.data.root_pos_w[env_ids, 2] - env.scene.env_origins[env_ids, 2]
        success = object_height > lift_height_threshold

        demote = self.current_difficulties[env_ids] if promotion_only else (self.current_difficulties[env_ids] - 1)
        self.current_difficulties[env_ids] = torch.where(
            success,
            self.current_difficulties[env_ids] + 1,
            demote,
        ).clamp(min=min_difficulty, max=max_difficulty)

        self.difficulty_frac = torch.mean(self.current_difficulties) / max(max_difficulty, 1)
        return self.difficulty_frac

    def get_num_clutter_objects(self, env_id: int) -> int:
        """Return the number of clutter objects for the given environment."""
        difficulty = self.current_difficulties[env_id].item()
        if difficulty < 30:
            return 0
        elif difficulty < 40:
            return int(torch.randint(1, 3, (1,)).item())
        elif difficulty < 50:
            return int(torch.randint(3, 6, (1,)).item())
        else:
            return int(torch.randint(5, 8, (1,)).item())

```

## mdp/rewards.py
```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for G1 picking environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_object_reaching_reward(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Reward the RIGHT hand fingertips for moving close to the target object.

    Only right-hand fingertip body_ids should be supplied via robot_cfg.body_names.
    Left hand is left free to learn the declutter task.

    Returns:
        Tensor (num_envs,): values in [0, 1].
    """
    robot: Articulation = env.scene[robot_cfg.name]
    target_object: RigidObject = env.scene[object_cfg.name]

    fingertip_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids]   # (N, 5, 3)
    object_pos_w = target_object.data.root_pos_w.unsqueeze(1)         # (N, 1, 3)

    distances = torch.norm(fingertip_pos_w - object_pos_w, dim=-1)    # (N, 5)
    min_distance = distances.min(dim=-1).values                        # (N,)
    return 1.0 - torch.tanh(min_distance / std)


def target_object_grasping_reward(
    env: ManagerBasedRLEnv,
    threshold: float,
    contact_sensor_name: str,
    min_contacts: int = 1,
) -> torch.Tensor:
    """Binary reward when the RIGHT hand detects ≥ min_contacts fingertip contacts.

    Args:
        threshold: Force magnitude [N] to count as a contact.
        contact_sensor_name: Name of the RIGHT hand contact sensor in the scene.
        min_contacts: Minimum number of fingertips that must be in contact.

    Returns:
        Tensor (num_envs,): 1.0 if grasping, 0.0 otherwise.
    """
    sensor: ContactSensor = env.scene.sensors.get(contact_sensor_name)
    if sensor is None:
        return torch.zeros(env.num_envs, device=env.device)

    # net_forces_w: (N_envs, N_bodies, 3) — always populated
    forces = torch.norm(sensor.data.net_forces_w, dim=-1)   # (N, N_bodies)
    num_contacts = (forces > threshold).sum(dim=-1)          # (N,)
    return (num_contacts >= min_contacts).float()


def target_object_lift_reward(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Smooth reward for lifting the target object above minimal_height.

    Returns:
        Tensor (num_envs,): values in [0, 1].
    """
    target_object: RigidObject = env.scene[object_cfg.name]
    object_height = target_object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.tanh((object_height - minimal_height) / 0.02).clamp(0.0, 1.0)


def left_hand_declutter_reward(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distractor_names: list[str] | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Reward the LEFT hand for moving distractors away from the target object.

    Two equally-weighted components:
      1. Near reward  – LEFT fingertips close to the nearest active distractor.
      2. Spread reward – mean distance of active distractors from the target.

    When no distractors are active (all hidden at z=-5) the function returns 0.

    Args:
        std: Length scale [m] for the near-reward tanh kernel.
        robot_cfg: Must contain the LEFT hand fingertip body_ids.
        distractor_names: List of scene entity names for distractor rigid objects.
        target_cfg: Scene entity for the target object.

    Returns:
        Tensor (num_envs,): values in [0, 1].
    """
    if distractor_names is None:
        distractor_names = []

    robot: Articulation = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    n_envs = env.num_envs
    device = env.device

    left_tips = robot.data.body_pos_w[:, robot_cfg.body_ids]  # (N, 5, 3)
    target_pos = target.data.root_pos_w                        # (N, 3)

    # Accumulators
    best_near = torch.zeros(n_envs, device=device)
    spread_sum = torch.zeros(n_envs, device=device)
    n_active = torch.zeros(n_envs, device=device)

    for name in distractor_names:
        try:
            obj: RigidObject = env.scene[name]
        except Exception:
            continue

        obj_pos = obj.data.root_pos_w                  # (N, 3)
        active = (obj_pos[:, 2] > -4.0).float()        # 1 if on table, 0 if hidden

        # Left fingertips → this distractor (minimum over 5 tips)
        dists = torch.norm(left_tips - obj_pos.unsqueeze(1), dim=-1).min(dim=-1).values  # (N,)
        near = (1.0 - torch.tanh(dists / std)) * active
        best_near = torch.max(best_near, near)

        # Distractor → target distance (reward grows as they separate)
        dist_to_tgt = torch.norm(obj_pos - target_pos, dim=-1) * active
        spread_sum += dist_to_tgt
        n_active += active

    has_distractor = (n_active > 0)
    # Mean spread (0 when no distractors)
    mean_spread = torch.where(has_distractor, spread_sum / n_active.clamp(min=1.0), torch.zeros(n_envs, device=device))
    # Normalize: 0.15 m apart = good separation
    spread_reward = torch.tanh(mean_spread / 0.15) * has_distractor.float()

    return 0.5 * best_near + 0.5 * spread_reward


def left_hand_near_target_penalty(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Soft penalty when LEFT hand fingertips are close to the target object.

    Discourages the left hand from interfering with the right-hand pick.
    Uses position only (no contact sensor filter needed).

    Returns:
        Tensor (num_envs,): values in [-1, 0].
    """
    robot: Articulation = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]

    left_tips = robot.data.body_pos_w[:, robot_cfg.body_ids]      # (N, 5, 3)
    target_pos = target.data.root_pos_w.unsqueeze(1)               # (N, 1, 3)

    dists = torch.norm(left_tips - target_pos, dim=-1).min(dim=-1).values  # (N,)
    return -(1.0 - torch.tanh(dists / std))


def pick_success_reward(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    hold_time_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Sparse bonus for picking and holding the target for hold_time_threshold seconds.

    Returns:
        Tensor (num_envs,): 10.0 when success, 0.0 otherwise.
    """
    target_object: RigidObject = env.scene[object_cfg.name]
    object_height = target_object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    is_lifted = object_height > minimal_height

    if not hasattr(env, "object_hold_time"):
        env.object_hold_time = torch.zeros(env.num_envs, device=env.device)

    env.object_hold_time = torch.where(
        is_lifted,
        env.object_hold_time + env.step_dt,
        torch.zeros_like(env.object_hold_time),
    )
    return (env.object_hold_time >= hold_time_threshold).float() * 10.0


def non_target_penalty(
    env: ManagerBasedRLEnv,
    threshold: float,
    contact_sensor_names: list[str],
) -> torch.Tensor:
    """Penalty for any fingertip contact with non-target objects (unused by default)."""
    total_contact = torch.zeros(env.num_envs, device=env.device)
    for sensor_name in contact_sensor_names:
        sensor = env.scene.sensors.get(sensor_name)
        if sensor is None:
            continue
        force = torch.norm(sensor.data.net_forces_w, dim=-1).max(dim=-1).values
        total_contact += (force > threshold).float()
    return -total_contact

```

## mdp/observations.py
```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for G1 picking environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_object_position_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Target object position in the robot's root frame (privileged information).

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame).
        object_cfg: Scene entity for the target object.

    Returns:
        Tensor of shape (num_envs, 3): target object position [x, y, z] in robot root frame.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    target_object: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(
        robot.data.root_quat_w, target_object.data.root_pos_w - robot.data.root_pos_w
    )


def fingertip_positions_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fingertip positions in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot with fingertip bodies.

    Returns:
        Tensor of shape (num_envs, num_fingertips * 3): fingertip positions concatenated.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    # Get fingertip positions in world frame
    fingertip_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids]  # (num_envs, num_tips, 3)
    
    # Transform to robot base frame
    robot_pos_w = robot.data.root_pos_w.unsqueeze(1)  # (num_envs, 1, 3)
    robot_quat_w = robot.data.root_quat_w.unsqueeze(1)  # (num_envs, 1, 4)
    
    fingertip_pos_b = quat_apply_inverse(
        robot_quat_w.expand(-1, fingertip_pos_w.shape[1], -1).reshape(-1, 4),
        (fingertip_pos_w - robot_pos_w.expand(-1, fingertip_pos_w.shape[1], -1)).reshape(-1, 3)
    )
    
    return fingertip_pos_b.view(env.num_envs, -1)


def target_object_id(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """One-hot encoding of the target object ID (privileged information).

    This function assumes the environment stores target_object_ids as an attribute.

    Args:
        env: The environment.

    Returns:
        Tensor of shape (num_envs, max_objects): one-hot encoding of target object.
    """
    # This will be populated by the environment's reset logic
    if hasattr(env, "target_object_ids"):
        max_objects = 10  # Maximum number of objects in scene
        one_hot = torch.zeros(env.num_envs, max_objects, device=env.device)
        one_hot.scatter_(1, env.target_object_ids.unsqueeze(1), 1.0)
        return one_hot
    else:
        # Return zeros if not available (during initialization)
        return torch.zeros(env.num_envs, 10, device=env.device)

```

## mdp/events.py
```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event functions for G1 picking environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_target_object_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
):
    """Reset target object position on the tray.

    Args:
        env: The environment.
        env_ids: Environment IDs to reset.
        pose_range: Dictionary with position ranges for x, y, z.
        object_cfg: Scene entity for the target object.
    """
    target_object: RigidObject = env.scene[object_cfg.name]
    
    # Sample random positions within range
    num_resets = len(env_ids)
    pos_x = torch.rand(num_resets, device=env.device) * (pose_range["x"][1] - pose_range["x"][0]) + pose_range["x"][0]
    pos_y = torch.rand(num_resets, device=env.device) * (pose_range["y"][1] - pose_range["y"][0]) + pose_range["y"][0]
    pos_z = torch.rand(num_resets, device=env.device) * (pose_range["z"][1] - pose_range["z"][0]) + pose_range["z"][0]
    
    # Set positions
    target_object.data.root_pos_w[env_ids, 0] = env.scene.env_origins[env_ids, 0] + pos_x
    target_object.data.root_pos_w[env_ids, 1] = env.scene.env_origins[env_ids, 1] + pos_y
    target_object.data.root_pos_w[env_ids, 2] = env.scene.env_origins[env_ids, 2] + pos_z
    
    # Reset velocities
    target_object.data.root_lin_vel_w[env_ids] = 0.0
    target_object.data.root_ang_vel_w[env_ids] = 0.0
    
    # Write to simulation
    target_object.write_root_state_to_sim(target_object.data.root_state_w[env_ids], env_ids)


def reset_clutter_based_on_difficulty(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    distractor_names: list[str],
    tray_surface_height: float = 0.845,
    hidden_height: float = -5.0,
    tray_x_half: float = 0.14,
    tray_y_half: float = 0.17,
    table_center_x: float = 0.5,
) -> None:
    """Place distractor objects on the tray or hide them based on curriculum difficulty.

    Clutter activation schedule (difficulty 0–60):
    - distractor 0: appears at difficulty ≥ 30
    - distractor 1: appears at difficulty ≥ 40
    - distractor 2: appears at difficulty ≥ 50

    Args:
        env: The environment.
        env_ids: Environment IDs to reset.
        distractor_names: Scene entity names of the distractor rigid objects.
        tray_surface_height: Z of tray surface (from env origin).
        hidden_height: Z to hide inactive distractors (below table).
        tray_x_half: Half-size of tray in x for random placement.
        tray_y_half: Half-size of tray in y for random placement.
        table_center_x: X offset from env origin to table/tray centre.
    """
    # Get per-environment difficulty from the curriculum manager.
    # After init, term_cfg.func holds the PickingCurriculumScheduler instance.
    difficulties = torch.zeros(env.num_envs, device=env.device)
    try:
        cm = env.curriculum_manager
        idx = cm._term_names.index("picking_curriculum")
        difficulties = cm._term_cfgs[idx].func.current_difficulties
    except Exception:
        pass  # No curriculum yet or name mismatch → default to no clutter

    diff_for_ids = difficulties[env_ids]  # (num_resets,)
    num_resets = len(env_ids)

    for dist_idx, name in enumerate(distractor_names):
        obj: RigidObject = env.scene[name]
        default_states = obj.data.default_root_state[env_ids].clone()
        orientations = default_states[:, 3:7]  # keep default orientation (identity)

        # Each distractor activates at a progressively higher difficulty threshold
        activation_threshold = 30.0 + dist_idx * 10.0
        is_active = diff_for_ids >= activation_threshold  # (num_resets,) bool

        # Random on-tray positions
        rand_x = (torch.rand(num_resets, device=env.device) - 0.5) * 2.0 * tray_x_half
        rand_y = (torch.rand(num_resets, device=env.device) - 0.5) * 2.0 * tray_y_half

        pos = torch.zeros(num_resets, 3, device=env.device)
        pos[:, 0] = env.scene.env_origins[env_ids, 0] + table_center_x + rand_x
        pos[:, 1] = env.scene.env_origins[env_ids, 1] + rand_y
        active_z = env.scene.env_origins[env_ids, 2] + tray_surface_height
        hidden_z = env.scene.env_origins[env_ids, 2] + hidden_height
        pos[:, 2] = torch.where(is_active, active_z, hidden_z)

        velocities = torch.zeros(num_resets, 6, device=env.device)

        obj.write_root_pose_to_sim(torch.cat([pos, orientations], dim=-1), env_ids=env_ids)
        obj.write_root_velocity_to_sim(velocities, env_ids=env_ids)

```

## mdp/terminations.py
```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom termination functions for G1 picking environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_object_dropped(
    env: ManagerBasedRLEnv,
    minimum_height: float = -0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Termination condition when target object falls below minimum height.

    Args:
        env: The environment.
        minimum_height: Minimum height threshold (relative to environment origin).
        object_cfg: Scene entity for the target object.

    Returns:
        Tensor of shape (num_envs,): boolean termination flags.
    """
    target_object: RigidObject = env.scene[object_cfg.name]
    object_height = target_object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return object_height < minimum_height

```

## mdp/__init__.py
```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP functions for G1 picking environment."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .curriculum import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403

```

## agents/rsl_rl_ppo_cfg.py
```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1PickPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_pick"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

```

## agents/rl_games_ppo_cfg.yaml
```yaml
params:
  seed: 42

  # environment wrapper clipping (required by rl_games train.py)
  env:
    clip_observations: 100.0
    clip_actions: 100.0

  algo:
    name: a2c_continuous

  model:
    name: continuous_a2c_logstd

  network:
    name: actor_critic
    separate: False
    
    space:
      continuous:
        mu_activation: None
        sigma_activation: None
        mu_init:
          name: default
        sigma_init:
          name: const_initializer
          val: 0
        fixed_sigma: True

    mlp:
      units: [512, 256, 128]
      activation: elu
      d2rl: False
      
      initializer:
        name: default
      regularizer:
        name: None

  load_checkpoint: False
  load_path: ''

  config:
    name: G1Pick
    full_experiment_name: G1Pick
    device: 'cuda:0'
    device_name: 'cuda:0'
    env_name: rlgpu
    multi_gpu: False
    ppo: True
    mixed_precision: False
    normalize_input: True
    normalize_value: True
    value_bootstrap: True
    num_actors: -1  # Configured by the script
    reward_shaper:
      scale_value: 0.01
    normalize_advantage: True
    gamma: 0.99
    tau: 0.95
    learning_rate: 3.e-4
    lr_schedule: adaptive
    schedule_type: standard
    kl_threshold: 0.016
    score_to_win: 100000
    max_epochs: 10000
    save_best_after: 50
    save_frequency: 100
    print_stats: True
    grad_norm: 1.0
    entropy_coef: 0.0
    truncate_grads: True
    e_clip: 0.2
    horizon_length: 32
    minibatch_size: 28800
    mini_epochs: 8
    critic_coef: 4
    clip_value: True
    seq_len: 4
    bounds_loss_coef: 0.0001

```

## __init__.py
```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
G1 Object Picking Environment with Curriculum Learning.

This environment trains a G1 humanoid robot to pick target objects from a cluttered tray.
The robot stands with frozen lower body and uses its arms and dexterous hands to manipulate objects.
"""

import gymnasium as gym

from . import agents
from .g1_pick_env_cfg import G1PickEnvCfg, G1PickEnvCfg_PLAY

##
# Register Gym environments.
##

gym.register(
    id="Isaac-G1-Pick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1PickEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PickPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Pick-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1PickEnvCfg_PLAY,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PickPPORunnerCfg",
    },
)

```

