# env_cfg.py — G1 right-arm-only lift, fresh start

from __future__ import annotations
import copy
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import CuboidCfg, RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from .robot_cfg import G1_INSPIRE_CFG
from . import mdp

_OBJ_INIT_Z   = 0.852   # cube resting on tray (tray top ~0.820 + half-cube 0.025 + gap)
_SUCCESS_Z    = 0.920   # 7 cm above resting = meaningful lift
_DROP_Z       = 0.600   # below this → fell off table → terminate

# Fingertip body names for the RIGHT hand only
_RIGHT_TIPS = [
    "right_thumb_4",
    "right_index_2",
    "right_middle_2",
    "right_ring_2",
    "right_little_2",
]


@configclass
class SceneCfg(InteractiveSceneCfg):
    replicate_physics: bool = True

    robot: ArticulationCfg = G1_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.4, 0.0, 0.40]),
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 1.2, 0.80),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.4, 0.3)),
        ),
    )

    tray = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tray",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.4, 0.0, 0.810]),
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.6, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        ),
    )

    target_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            physics_material=RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.40, 0.00, _OBJ_INIT_Z]),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )


@configclass
class ActionsCfg:
    right_arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        scale=1.5,
        use_default_offset=True,
    )
    right_hand_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_thumb_1_joint",
            "right_thumb_2_joint",
            "right_index_1_joint",
            "right_middle_1_joint",
            "right_ring_1_joint",
            "right_little_1_joint",
        ],
        scale=1.0,   # smaller scale for fingers — tighter joint ranges
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Joint state — right arm + hand only keeps obs small and focused
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint", "right_elbow_joint",
                "right_wrist_roll_joint", "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
                "right_thumb_1_joint", "right_thumb_2_joint",
                "right_index_1_joint", "right_middle_1_joint",
                "right_ring_1_joint", "right_little_1_joint",
            ])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint", "right_elbow_joint",
                "right_wrist_roll_joint", "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
                "right_thumb_1_joint", "right_thumb_2_joint",
                "right_index_1_joint", "right_middle_1_joint",
                "right_ring_1_joint", "right_little_1_joint",
            ])},
        )

        # Object state in robot frame — position + velocity
        object_pos_b = ObsTerm(
            func=mdp.target_object_position_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("target_object"),
            },
        )
        object_vel = ObsTerm(
            func=mdp.object_root_velocity,   # see obs.py below
            params={"object_cfg": SceneEntityCfg("target_object")},
        )

        # RIGHT fingertip positions in robot frame — 5 tips × 3 = 15 values
        right_fingertip_pos = ObsTerm(
            func=mdp.fingertip_positions_b,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=_RIGHT_TIPS),
            },
        )

        # # Fingertip-to-object delta vectors — most direct grasp signal
        # # shape: (N, 5*3=15)  each vector points from tip to cube centre
        fingertip_to_object = ObsTerm(
            func=mdp.fingertip_to_object_vectors,  # see obs.py below
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=_RIGHT_TIPS),
                "object_cfg": SceneEntityCfg("target_object"),
            },
        )

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False   # no noise during initial training
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:

    # ==========================================
    # STAGE 1: APPROACH (Low Weights: 1.0 - 5.0)
    # Goal: Get the hand to the tray
    # ==========================================
    fingertip_proximity = RewTerm(
        func=mdp.fingertip_proximity_reward,
        weight=2.0,  # Base breadcrumb
        params={
            "std": 0.08,
            "robot_cfg": SceneEntityCfg("robot", body_names=_RIGHT_TIPS),
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )

    # approach_velocity = RewTerm(
    #     func=mdp.approach_velocity_reward,
    #     weight=3.0,  # Higher than proximity so it actively moves, doesn't just sit
    #     params={
    #         "robot_cfg": SceneEntityCfg("robot", body_names=_RIGHT_TIPS),
    #         "object_cfg": SceneEntityCfg("target_object"),
    #     },
    # )

    # ==========================================
    # STAGE 2: GRASP (Medium Weights: 8.0 - 15.0)
    # Goal: Wrap fingers and make physical contact
    # ==========================================
    finger_closure = RewTerm(
        func=mdp.finger_closure_reward,
        weight=5.0,  # Curling fingers is good...
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=_RIGHT_TIPS),
            "object_cfg": SceneEntityCfg("target_object"),
            "max_closure_dist": 0.1 ,
        },
    )

    contact_detection = RewTerm(
        func=mdp.contact_detection_reward,
        weight=5.0, # ...but actually touching the cube is TWICE as good!
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=_RIGHT_TIPS),
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )

    # ==========================================
    # STAGE 3: LIFT (High Weights: 20.0 - 50.0)
    # Goal: Break gravity
    # ==========================================
    upward_velocity = RewTerm(
        func=mdp.upward_velocity_reward,
        weight=10.0, # Immediate reward for yanking upward
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=_RIGHT_TIPS),
            "object_cfg": SceneEntityCfg("target_object"),
            "gate_std": 0.13,
        },
    )

    lift_height = RewTerm(
    func=mdp.lift_height_reward,
    weight=20.0, # Continuous payout for staying in the air
    params={
        "robot_cfg": SceneEntityCfg("robot", body_names=_RIGHT_TIPS),
        "object_cfg": SceneEntityCfg("target_object"),
        "resting_height": _OBJ_INIT_Z,
        "max_height": _SUCCESS_Z,
        "gate_std": 0.13,
    },
    )

    # height_progress = RewTerm(
    #     func=mdp.height_progress_reward,
    #     weight=40.0, # The massive breakthrough payout
    #     params={
    #         "object_cfg": SceneEntityCfg("target_object"),
    #         "resting_height": _OBJ_INIT_Z,
    #     },
    # )

    # ==========================================
    # STAGE 4: SUCCESS (Max Weights: 50.0+)
    # Goal: Freeze at the target height
    # ==========================================
    hold_height = RewTerm(
        func=mdp.hold_height_reward,
        weight=50.0, # Match height_progress so it prefers holding over throwing
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=_RIGHT_TIPS),
            "object_cfg": SceneEntityCfg("target_object"),
            "target_height": _SUCCESS_Z + 0.03,
            "min_height": 0.870,
            "std": 0.05,
            "gate_std": 0.13,
        },
    )

    success = RewTerm(
        func=mdp.success_bonus,
        weight=50.0, # Cherry on top
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=_RIGHT_TIPS),
            "object_cfg": SceneEntityCfg("target_object"),
            "success_height": 0.880,
            "gate_std": 0.13,
        },
    )

    # ==========================================
    # PENALTIES (Zeroed out for exploration)
    # ==========================================
    early_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=0.0,
        params={"term_keys": "target_dropped"},
    )

    action_smoothness = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=0.0,
    )
    
    joint_limit_penalty = RewTerm(
        func=mdp.joint_pos_limit_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
                "right_thumb_1_joint", "right_thumb_2_joint", "right_index_1_joint", "right_middle_1_joint",
                "right_ring_1_joint", "right_little_1_joint",
            ]),
        },
    )
    


@configclass
class EventCfg:
    # Full scene reset
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_right_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint", "right_elbow_joint",
                "right_wrist_roll_joint", "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_right_hand = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "right_thumb_1_joint", "right_thumb_2_joint",
                "right_index_1_joint", "right_middle_1_joint",
                "right_ring_1_joint", "right_little_1_joint",
            ]),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    # LEFT arm: frozen at default — zero offset, zero velocity
    freeze_left_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint", "left_elbow_joint",
                "left_wrist_roll_joint", "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "left_thumb_1_joint", "left_thumb_2_joint",
                "left_index_1_joint", "left_middle_1_joint",
                "left_ring_1_joint", "left_little_1_joint",
            ]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Freeze lower body, mimic joints, waist
    freeze_lower_body = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint",
                ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            ]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    freeze_mimic_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_thumb_3_joint", ".*_thumb_4_joint",
                ".*_index_2_joint", ".*_middle_2_joint",
                ".*_ring_2_joint", ".*_little_2_joint",
            ]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Object: random position on tray
    reset_target_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.10, 0.10), "y": (-0.05, 0.05), "z": (0.01, 0.01)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("target_object"),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    target_dropped = DoneTerm(
        func=mdp.target_object_dropped,
        params={
            "minimum_height": _DROP_Z,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )

    joint_limit = DoneTerm(
        func=mdp.joint_pos_out_of_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                # Arm joints only — finger joints have lower=0 which random actions always violate
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]),
        },
    )


@configclass
class G1RightArmLiftEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3.0)
    viewer: ViewerCfg = ViewerCfg(
        resolution=(800, 600),
        eye=(12.0, -4.0, 2.5),
        lookat=(8.0, 0.0, 0.85),
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # In __post_init__
        self.sim.physx.gpu_max_rigid_patch_count = 2 * 5 * 2**15  # halve it from 4 * 5 * 2**15

        # Freeze left hand — high stiffness holds it at reset pose
        left_hand_act = copy.deepcopy(self.scene.robot.actuators["left_hand"])
        left_hand_act.stiffness = 10000.0
        left_hand_act.damping = 1000.0
        left_hand_act.effort_limit_sim = 10000.0
        self.scene.robot.actuators["left_hand"] = left_hand_act

        # Freeze legs/feet
        for part in ["legs", "feet"]:
            act = copy.deepcopy(self.scene.robot.actuators[part])
            act.stiffness = 10000.0
            act.damping = 1000.0
            act.effort_limit_sim = 10000.0
            self.scene.robot.actuators[part] = act


@configclass
class G1RightArmLiftEnvCfg_PLAY(G1RightArmLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64

# Aliases for backward compatibility with __init__.py
G1PickEnvCfg = G1RightArmLiftEnvCfg
G1PickEnvCfg_PLAY = G1RightArmLiftEnvCfg_PLAY