# g1_env_cfg.py

from __future__ import annotations
import copy
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg, ManagerBasedRLEnv
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

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

from .robot_cfg import G1_INSPIRE_CFG
from . import mdp

_OBJ_INIT_Z   = 0.850
_SUCCESS_Z    = 0.950
_DROP_Z       = 0.600


# Updated to match the actual Rigid Bodies parsed by PhysX
_RIGHT_HAND_BODIES = [
    "right_wrist_yaw_link",  # Index 0: The rigid palm center
    "R_thumb_distal",        # Index 1: Thumb tip (physically tracked)
    "R_index_intermediate",  # Index 2: Index tip (physically tracked)
    "R_middle_intermediate", # Index 3: Middle tip (physically tracked)
    "R_ring_intermediate",   # Index 4: Ring tip (physically tracked)
    "R_pinky_intermediate",  # Index 5: Pinky tip (physically tracked)
]

def wuji_monolithic_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    action_penalty_scale: float = 1.0,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w.clone()
    object_pos[:, 2] = object_pos[:, 2] - 0.03

    hand_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    tips_w = robot.data.body_pos_w[:, robot_cfg.body_ids[1:]]

    dists = torch.linalg.norm(tips_w - object_pos.unsqueeze(1), dim=-1)
    fingertip_dist_avg = dists.mean(dim=1)
    hand_finger_rew = 1.0 - torch.tanh(fingertip_dist_avg / 0.1) 

    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    action_rate_penalty = torch.sum(torch.square(actions - prev_actions), dim=-1)

    hand_object_pos = object_pos.clone()
    hand_object_pos[:, 0] = hand_object_pos[:, 0] - 0.160 
    hand_object_pos[:, 2] = hand_object_pos[:, 2] + 0.040  

    hand_object_dist = torch.linalg.norm(hand_pos - hand_object_pos, dim=-1)
    hand_object_rew = 1.0 - torch.tanh(hand_object_dist / 0.3)

    lift_rew = torch.where(object_pos[:, 2] > _SUCCESS_Z, 1.0, 0.0)
    lift_cont_rew = (object_pos[:, 2] - _OBJ_INIT_Z).clamp(min=0.0)
    
    goal_rew = (object_pos[:, 2] > _SUCCESS_Z).float() * (1.0 - torch.tanh(hand_object_dist / 0.3))
    goal_rew_fine_grained = (object_pos[:, 2] > _SUCCESS_Z).float() * (
        1.0 - torch.tanh(hand_object_dist / 0.05)
    )

    r_close = 0.1
    close_bonus = (r_close - hand_object_dist).clamp(min=0.0) / r_close

    reward = (
        0.25 * hand_object_rew
        + close_bonus
        + 0.5 * hand_finger_rew 
        - action_rate_penalty * action_penalty_scale
        + lift_rew * 25.0
        + goal_rew * 16.0
        + goal_rew_fine_grained * 5.0
        + lift_cont_rew * 80.0
        - 0.005
    )

    resets = torch.where(
        obj.data.root_pos_w[:, 2] < _DROP_Z,
        torch.ones_like(reward),
        torch.zeros_like(reward)
    )
    reward = torch.where(resets == 1, torch.ones_like(reward) * -5.0, reward)

    return reward

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
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.4, 0.0, _OBJ_INIT_Z]),
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
            size=(0.06, 0.06, 0.06),
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

class InspireMimicAction(JointPositionAction):
    def __init__(self, cfg: JointPositionActionCfg, env):
        super().__init__(cfg, env)
        
        def get_idx(joint_name):
            return self._asset.find_joints(joint_name)[0][0]

        # 1. Primary joints (USD Names)
        self.src_thumb_2 = get_idx("R_thumb_proximal_pitch_joint")
        self.src_idx_1 = get_idx("R_index_proximal_joint")
        self.src_mid_1 = get_idx("R_middle_proximal_joint")
        self.src_rng_1 = get_idx("R_ring_proximal_joint")
        self.src_lit_1 = get_idx("R_pinky_proximal_joint")

        # 2. Mimic joints (USD Names)
        self.mimic_joint_indices = [
            get_idx("R_thumb_intermediate_joint"),
            get_idx("R_thumb_distal_joint"),
            get_idx("R_index_intermediate_joint"),
            get_idx("R_middle_intermediate_joint"),
            get_idx("R_ring_intermediate_joint"),
            get_idx("R_pinky_intermediate_joint")
        ]

    def apply_actions(self):
        super().apply_actions()
        targets = self._asset.data.joint_pos_target.clone()
        
        t3 = (targets[:, self.src_thumb_2] * 0.8024).unsqueeze(1)
        t4 = t3 * 0.9487 
        i2 = (targets[:, self.src_idx_1] * 1.0843).unsqueeze(1)
        m2 = (targets[:, self.src_mid_1] * 1.0843).unsqueeze(1)
        r2 = (targets[:, self.src_rng_1] * 1.0843).unsqueeze(1)
        l2 = (targets[:, self.src_lit_1] * 1.0843).unsqueeze(1)
        
        mimic_targets = torch.cat([t3, t4, i2, m2, r2, l2], dim=1)
        self._asset.set_joint_position_target(mimic_targets, joint_ids=self.mimic_joint_indices)

@configclass
class InspireMimicActionCfg(JointPositionActionCfg):
    class_type: type = InspireMimicAction

@configclass
class ActionsCfg:
    right_arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ],
        scale=2.5,
        use_default_offset=True,
    )
    # Updated to USD proximal joints
    right_hand_action = InspireMimicActionCfg(
        asset_name="robot",
        joint_names=[
            "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint",
            "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint",
        ],
        scale=1.0,
        use_default_offset=True,
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", 
                "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
                "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint", 
                "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint",
            ])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", 
                "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
                "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint", 
                "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint",
            ])},
            clip=(-50.0, 50.0),
        )
        object_pos_b = ObsTerm(
            func=mdp.target_object_position_b,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("target_object")},
        )
        object_vel = ObsTerm(
            func=mdp.object_root_velocity,
            params={"object_cfg": SceneEntityCfg("target_object")},
            clip=(-50.0, 50.0),
        )
        right_fingertip_pos = ObsTerm(
            func=mdp.fingertip_positions_b,
            params={"robot_cfg": SceneEntityCfg("robot", body_names=_RIGHT_HAND_BODIES)},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    wuji_total = RewTerm(
        func=wuji_monolithic_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=_RIGHT_HAND_BODIES),
            "object_cfg": SceneEntityCfg("target_object"),
            "action_penalty_scale": 0.0,
        },
    )

@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_right_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", 
                "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
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
                "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint", 
                "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint",
            ]),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    freeze_left_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", 
                "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                "L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint", "L_index_proximal_joint", 
                "L_middle_proximal_joint", "L_ring_proximal_joint", "L_pinky_proximal_joint",
            ]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

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

@configclass
class G1RightArmLiftEnvCfg_V2(ManagerBasedRLEnvCfg):
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
        self.sim.physx.gpu_max_rigid_patch_count = 2 * 5 * 2**15 

        # Freeze left hand
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
class G1RightArmLiftEnvCfg_V2_PLAY(G1RightArmLiftEnvCfg_V2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64

# Aliases for registering the version in __init__.py
G1PickEnvCfg = G1RightArmLiftEnvCfg_V2
G1PickEnvCfg_PLAY = G1RightArmLiftEnvCfg_V2_PLAY