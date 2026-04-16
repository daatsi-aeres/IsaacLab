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

# --- PERFECTED HEIGHT CALCULATIONS (0.05m Cube) ---
_OBJ_INIT_Z   = 0.845   # Cube center (Bottom sits perfectly flush at 0.820m tray surface)
_SUCCESS_Z    = 1.134   # 1.2x the original 0.945 — ~29cm lift from tray
_DROP_Z       = 0.600   # Triggers early termination if the policy knocks it off the table

# Provides the exact physical links the policy needs to track for dense distance rewards
_RIGHT_HAND_BODIES = [
    "right_wrist_yaw_link",  # Index 0: Palm anchor for pre-grasp positioning
    "R_thumb_distal",        # Index 1: Thumb tip (physically tracked)
    "R_index_intermediate",  # Index 2: Index tip (physically tracked)
    "R_middle_intermediate", # Index 3: Middle tip (physically tracked)
    "R_ring_intermediate",   # Index 4: Ring tip (physically tracked)
    "R_pinky_intermediate",  # Index 5: Pinky tip (physically tracked)
]

def distractor_velocity_penalty(
    env: ManagerBasedRLEnv,
    distractor_1_cfg: SceneEntityCfg,
    distractor_2_cfg: SceneEntityCfg,
    distractor_3_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize disturbing non-target objects. Averaged across distractors so
    adding more distractors later doesn't scale up the penalty magnitude."""
    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in [distractor_1_cfg, distractor_2_cfg, distractor_3_cfg]:
        obj: RigidObject = env.scene[cfg.name]
        speed = torch.norm(obj.data.root_lin_vel_w, dim=-1)
        total += torch.tanh(speed / 0.15)
    return total / 3.0


def distractor_drop_penalty(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    distractor_1_cfg: SceneEntityCfg,
    distractor_2_cfg: SceneEntityCfg,
    distractor_3_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Sparse penalty: returns 1.0 if ANY distractor fell below minimum_height."""
    any_dropped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for cfg in [distractor_1_cfg, distractor_2_cfg, distractor_3_cfg]:
        obj: RigidObject = env.scene[cfg.name]
        any_dropped |= obj.data.root_pos_w[:, 2] < minimum_height
    return any_dropped.float()


def wuji_monolithic_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    action_rate_scale: float = 0.005, 
    joint_vel_scale: float = 0.0005,   
    action_l2_scale: float = 0.001,    
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # 1. EXTRACT
    cube_pos   = obj.data.root_pos_w.clone()
    palm_pos   = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    tips_pos   = robot.data.body_pos_w[:, robot_cfg.body_ids[1:]]
    joint_vels = robot.data.joint_vel

    # Distances - scalars [N]
    raw_thumb_dist  = torch.linalg.norm(tips_pos[:, 0:1] - cube_pos.unsqueeze(1), dim=-1).squeeze(1)
    raw_finger_dist = torch.linalg.norm(tips_pos[:, 1:]  - cube_pos.unsqueeze(1), dim=-1).mean(dim=1)

    # Shift the zero-point to the surface of the 5cm cube (radius = 0.025m)
    thumb_dist = torch.clamp(raw_thumb_dist - 0.025, min=0.0)
    finger_dist = torch.clamp(raw_finger_dist - 0.025, min=0.0)

    # 2. POSTURE - palm 8cm above cube
    palm_target = cube_pos.clone()
    palm_target[:, 2] += 0.08
    palm_dist   = torch.linalg.norm(palm_pos - palm_target, dim=-1)
    posture_rew = 1.0 - torch.tanh(torch.clamp(palm_dist - 0.03, min=0.0) / 0.3)

    # 3. REACH - 3D midpoint of fingertips toward cube center
    # tips_pos is [N, 5, 3], mean over finger dim gives [N, 3] position
    fingertip_midpoint = tips_pos.mean(dim=1)                              # [N, 3]
    midpoint_dist = torch.linalg.norm(fingertip_midpoint - cube_pos, dim=-1)  # [N]
    reach_rew = 1.0 - torch.tanh(midpoint_dist / 0.25)

    # 4. GRASP - separate thumb vs finger shaping, mirrors is_grasped structure
    thumb_grasp_rew  = 1.0 - torch.tanh(thumb_dist  / 0.055)
    finger_grasp_rew = 1.0 - torch.tanh(finger_dist / 0.055)
    grasp_rew        = (thumb_grasp_rew + finger_grasp_rew) / 2.0

    # 5. SOFT GATE
    _T        = 0.06
    is_grasped = (1.0 - torch.tanh(thumb_dist / _T)) * (1.0 - torch.tanh(finger_dist / _T)) 


    lift_height   = (cube_pos[:, 2] - _OBJ_INIT_Z).clamp(min=0.0, max=0.30)

    lift_cont_rew = lift_height * 2.0 * is_grasped
    is_lifted     = cube_pos[:, 2] > _SUCCESS_Z
    success_bonus = is_lifted.float() * is_grasped

    # 6. PENALTIES & AGGREGATION
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    
    action_rate_penalty = torch.sum(torch.square(actions - prev_actions), dim=-1)
    joint_vel_penalty = torch.sum(torch.square(joint_vels), dim=-1)
    action_l2_penalty = torch.sum(torch.square(actions), dim=-1)

    is_dropped = cube_pos[:, 2] < _DROP_Z

    reward = (
        posture_rew * 1.0 +
        reach_rew * 1.0 +
        grasp_rew * 2.0 +
        lift_cont_rew +
        success_bonus * 1000.0 -
        (action_rate_penalty * action_rate_scale) -
        (joint_vel_penalty * joint_vel_scale) -
        (action_l2_penalty * action_l2_scale)
    )

    reward = torch.where(is_dropped, torch.ones_like(reward) * -0.5, reward)

    return reward

@configclass
class SceneCfg(InteractiveSceneCfg):
    replicate_physics: bool = True # Ensures determinism across all 4096 parallel environments
    robot: ArticulationCfg = G1_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.4, 0.0, 0.40]),
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 1.2, 0.80), # Static collision boundary to prevent arm clipping
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.4, 0.3)),
        ),
    )

    tray = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tray",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.4, 0.0, 0.81]), 
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.6, 0.02), # Kinematic obstacle the policy must learn to navigate around
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        ),
    )

    target_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05), # Reduced to 5cm to require higher dexterity from the policy
            physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0), # High friction prevents slipping
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, # High iterations ensure stable grasping physics
                solver_velocity_iteration_count=1,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2), # Light enough to lift, heavy enough to drop realistically
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.00, _OBJ_INIT_Z]),
    )

    distractor_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Distractor1",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),  # Blue
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.42, 0.08, _OBJ_INIT_Z]),
    )

    distractor_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Distractor2",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # Green
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.42, -0.08, _OBJ_INIT_Z]),
    )

    distractor_3: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Distractor3",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),  # Yellow
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.38, 0.0, _OBJ_INIT_Z]),
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
    # Custom action class drastically reduces the RL action space by mechanically coupling the finger joints
    def __init__(self, cfg: JointPositionActionCfg, env):
        super().__init__(cfg, env)
        
        def get_idx(joint_name):
            return self._asset.find_joints(joint_name)[0][0]

        self.src_thumb_2 = get_idx("R_thumb_proximal_pitch_joint")
        self.src_idx_1 = get_idx("R_index_proximal_joint")
        self.src_mid_1 = get_idx("R_middle_proximal_joint")
        self.src_rng_1 = get_idx("R_ring_proximal_joint")
        self.src_lit_1 = get_idx("R_pinky_proximal_joint")

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
        
        # Applies URDF hardware transmission multipliers to slave joints so the policy only controls the proximal base
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
        scale=0.3, # Action scale maps policy outputs [-1, 1] to larger joint position targets
        use_default_offset=True, # Actions are relative to the ideal starting posture
    )
    
    right_hand_action = InspireMimicActionCfg(
        asset_name="robot",
        joint_names=[
            "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint",
            "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint",
        ],
        scale=0.5, # Fingers need finer control, so we use a smaller action scale
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Proprioception: Tells the policy where its body currently is
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", 
                "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
                "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint", 
                "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint",
            ])},
        )
        # Proprioception: Tells the policy how fast its joints are moving (vital for damping/stopping)
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
        # Exteroception: Tells the policy where the target object is relative to its base
        object_pos_b = ObsTerm(
            func=mdp.target_object_position_b,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("target_object")},
        )
        # Exteroception: Helps the policy catch a slipping/falling block
        object_vel = ObsTerm(
            func=mdp.object_root_velocity,
            params={"object_cfg": SceneEntityCfg("target_object")},
            clip=(-50.0, 50.0),
        )
        # Distractor positions — privileged state so teacher learns avoidance
        distractor_1_pos_b = ObsTerm(
            func=mdp.target_object_position_b,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("distractor_1")},
        )
        distractor_2_pos_b = ObsTerm(
            func=mdp.target_object_position_b,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("distractor_2")},
        )
        distractor_3_pos_b = ObsTerm(
            func=mdp.target_object_position_b,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("distractor_3")},
        )
        # Dense Spatial Observation: Explicitly feeds the distance between fingertips and the block into the neural network
        right_fingertip_pos = ObsTerm(
            func=mdp.fingertip_positions_b,
            params={"robot_cfg": SceneEntityCfg("robot", body_names=_RIGHT_HAND_BODIES)},
        )
        # History: Feeds the previous action back into the network to help it learn smoothness
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False # Set to True to add Gaussian noise for Domain Randomization later
            self.concatenate_terms = True  # Flattens all observations into a single 1D tensor for the MLP

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    # Single monolithic function calculates all rewards, heavily optimizing GPU computation time
    wuji_total = RewTerm(
        func=wuji_monolithic_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=_RIGHT_HAND_BODIES),
            "object_cfg": SceneEntityCfg("target_object"),
            "action_rate_scale": 0.005, # Heavily penalizes twitching/spasming
            "joint_vel_scale": 0.001,  # Creates a "speed limit" to stop Mach 3 movements
            "action_l2_scale": 0.0,   # Encourages the network to rest when not moving
        },
    )

    distractor_penalty = RewTerm(
        func=distractor_velocity_penalty,
        weight=-50.0,
        params={
            "distractor_1_cfg": SceneEntityCfg("distractor_1"),
            "distractor_2_cfg": SceneEntityCfg("distractor_2"),
            "distractor_3_cfg": SceneEntityCfg("distractor_3"),
        },
    )

    distractor_drop = RewTerm(
        func=distractor_drop_penalty,
        weight=-500.0,
        params={
            "minimum_height": _DROP_Z,
            "distractor_1_cfg": SceneEntityCfg("distractor_1"),
            "distractor_2_cfg": SceneEntityCfg("distractor_2"),
            "distractor_3_cfg": SceneEntityCfg("distractor_3"),
        },
    )

@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Domain Randomization: Adds slight variance to starting arm position to prevent overfitting
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

    # Domain Randomization: Adds slight variance to starting finger positions
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

    # Freezes the inactive arm perfectly still to save exploration capability for the active arm
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

    # Freezes the legs/waist to reduce the overall state-action complexity the agent must learn
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

    # Domain Randomization: Spawns the cube in a slightly different location every episode to generalize grasping
    reset_target_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.10, 0.10), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("target_object"),
        },
    )

    # Distractor reset with small jitter so they stay in their quadrants
    reset_distractor_1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("distractor_1"),
        },
    )

    reset_distractor_2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("distractor_2"),
        },
    )

    reset_distractor_3 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("distractor_3"),
        },
    )

    # apply_high_friction_to_fingers = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup", # Only runs once when the environment boots up
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", 
    #             body_names=[
    #                 "R_thumb_distal", 
    #                 "R_index_intermediate", 
    #                 "R_middle_intermediate", 
    #                 "R_ring_intermediate", 
    #                 "R_pinky_intermediate"
    #             ]
    #         ),
    #         "static_friction_range": (1.5, 1.5),   # Min and Max are the same to force an exact value
    #         "dynamic_friction_range": (1.5, 1.5),
    #         "restitution_range": (0.0, 0.0),       # Zero bounciness
    #         "num_buckets": 1,                      # All fingers share this one material
    #     },
    # )

@configclass
class TerminationsCfg:
    # Resets the episode if the agent takes too long (8.0 seconds based on episode_length_s)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # End episode on successful lift — no wasted sim time after the task is done
    target_lifted = DoneTerm(
        func=mdp.target_object_lifted,
        params={
            "success_height": _SUCCESS_Z,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )

    # Instantly resets the environment if the cube falls off the table, preventing wasted simulation time
    target_dropped = DoneTerm(
        func=mdp.target_object_dropped,
        params={
            "minimum_height": _DROP_Z,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )

    distractor_1_dropped = DoneTerm(
        func=mdp.target_object_dropped,
        params={
            "minimum_height": _DROP_Z,
            "object_cfg": SceneEntityCfg("distractor_1"),
        },
    )

    distractor_2_dropped = DoneTerm(
        func=mdp.target_object_dropped,
        params={
            "minimum_height": _DROP_Z,
            "object_cfg": SceneEntityCfg("distractor_2"),
        },
    )

    distractor_3_dropped = DoneTerm(
        func=mdp.target_object_dropped,
        params={
            "minimum_height": _DROP_Z,
            "object_cfg": SceneEntityCfg("distractor_3"),
        },
    )

@configclass
class G1RightArmLiftEnvCfg_V2(ManagerBasedRLEnvCfg):
    # Initializes 4096 robots simultaneously on the GPU for massive parallel data collection
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
            # --- BASIC SIMULATION SETTINGS ---
            self.decimation = 4             
            self.episode_length_s = 8.0     
            self.sim.dt = 1 / 120           
            self.sim.render_interval = self.decimation
            self.sim.physx.bounce_threshold_velocity = 0.2

            # --- OPTIMIZED GPU BUFFERS FOR 8192 ENVS ---
            # Dropped from 10M to 2M to save VRAM and speed up iteration loops
            self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2 * 1024 * 1024
            self.sim.physx.gpu_found_lost_pairs_capacity = 2 * 1024 * 1024
            self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 1024
            
            self.sim.physx.gpu_max_rigid_patch_count = 1024 * 1024
            self.sim.physx.gpu_max_rigid_contact_count = 2 * 1024 * 1024
            
            self.sim.physx.gpu_heap_capacity = 64 * 1024 * 1024
            self.sim.physx.gpu_temp_buffer_capacity = 16 * 1024 * 1024
            # -----------------------------------------------------------

            # --- RIGID JOINT LOCKING ---
            # Lock Left Hand
            left_hand_act = copy.deepcopy(self.scene.robot.actuators["left_hand"])
            left_hand_act.stiffness = 10000.0
            left_hand_act.damping = 1000.0
            left_hand_act.effort_limit_sim = 10000.0
            self.scene.robot.actuators["left_hand"] = left_hand_act

            # Lock Lower Body
            for part in ["legs", "feet"]:
                if part in self.scene.robot.actuators:
                    act = copy.deepcopy(self.scene.robot.actuators[part])
                    act.stiffness = 10000.0
                    act.damping = 1000.0
                    act.effort_limit_sim = 10000.0
                    self.scene.robot.actuators[part] = act

@configclass
class G1RightArmLiftEnvCfg_V2_PLAY(G1RightArmLiftEnvCfg_V2):
    def __post_init__(self):
        super().__post_init__()
        # Reduces environment count during evaluation to save resources and allow smooth visualization
        self.scene.num_envs = 64

# Aliases for registering the version in __init__.py
G1PickEnvCfg = G1RightArmLiftEnvCfg_V2
G1PickEnvCfg_PLAY = G1RightArmLiftEnvCfg_V2_PLAY