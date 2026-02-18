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
