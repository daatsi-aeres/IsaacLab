from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def fingertip_proximity_reward(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    tips_w = robot.data.body_pos_w[:, robot_cfg.body_ids]
    obj_pos = obj.data.root_pos_w.unsqueeze(1)
    dists = torch.norm(tips_w - obj_pos, dim=-1)
    return (1.0 - torch.tanh(dists / std)).mean(dim=-1)


# In rewards.py
def finger_closure_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    max_closure_dist: float = 0.08,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    tips_w = robot.data.body_pos_w[:, robot_cfg.body_ids]
    obj_pos = obj.data.root_pos_w

    tip_vectors = tips_w - obj_pos.unsqueeze(1)
    tip_dirs = torch.nn.functional.normalize(tip_vectors, dim=-1)

    thumb_dir = tip_dirs[:, 0, :]
    other_dirs = tip_dirs[:, 1:, :]
    dots = (thumb_dir.unsqueeze(1) * other_dirs).sum(dim=-1)
    opposition = (1.0 - dots) / 2.0

    dists = torch.norm(tips_w - obj_pos.unsqueeze(1), dim=-1)
    close_enough = (dists < max_closure_dist).float()
    proximity_gate = close_enough[:, 0:1] * close_enough[:, 1:]

    # REMOVED the cube speed gate. Just reward the geometry!
    return (opposition * proximity_gate).mean(dim=-1)

def approach_velocity_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    
    tips_w = robot.data.body_pos_w[:, robot_cfg.body_ids] # (N, 5, 3)
    tips_vel_w = robot.data.body_lin_vel_w[:, robot_cfg.body_ids] # (N, 5, 3)
    obj_pos = obj.data.root_pos_w.unsqueeze(1) # (N, 1, 3)
    
    # Vector from tip to object
    to_obj = obj_pos - tips_w
    to_obj_dir = torch.nn.functional.normalize(to_obj, dim=-1)
    
    # Dot product of tip velocity and direction to object
    approach_vel = (tips_vel_w * to_obj_dir).sum(dim=-1)
    
    # We only care about positive approach velocity, not moving away
    return approach_vel.clamp(min=0.0).mean(dim=-1)

def _get_proximity_gate(env, robot_cfg, object_cfg, gate_std):
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    tips_w = robot.data.body_pos_w[:, robot_cfg.body_ids]
    obj_pos = obj.data.root_pos_w.unsqueeze(1)
    dists = torch.norm(tips_w - obj_pos, dim=-1)
    mean_prox = (1.0 - torch.tanh(dists / gate_std)).mean(dim=-1)
    return torch.sigmoid((mean_prox - 0.5) * 10.0)


def upward_velocity_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    gate_std: float = 0.13,
) -> torch.Tensor:
    """Reward cube moving upward RIGHT NOW — frozen hand earns zero."""
    obj: RigidObject = env.scene[object_cfg.name]
    upward_vel = obj.data.root_lin_vel_w[:, 2].clamp(0.0, 2.0)
    gate = _get_proximity_gate(env, robot_cfg, object_cfg, gate_std)
    return gate * upward_vel


def height_progress_reward(
    env, object_cfg=SceneEntityCfg("target_object"),
    resting_height=0.850,
    min_progress_threshold=0.005,  # ignore sub-5mm progress — filters physics jitter
) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    if not hasattr(env, "_max_cube_height"):
        env._max_cube_height = torch.full((env.num_envs,), resting_height, device=env.device)

    raw_progress = (obj_z - env._max_cube_height).clamp(0.0, 1.0)
    progress = (raw_progress - min_progress_threshold).clamp(0.0, 1.0)
    env._max_cube_height = torch.max(env._max_cube_height, obj_z)

    reset_ids = (env.episode_length_buf == 1).nonzero(as_tuple=False).flatten()
    if len(reset_ids) > 0:
        env._max_cube_height[reset_ids] = resting_height

    return progress


def hold_height_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    target_height: float = 0.950,
    min_height: float = 0.870,   # must be above this before hold reward activates
    std: float = 0.05,
    gate_std: float = 0.13,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    
    # Zero reward below min_height — resting position gets nothing
    above_threshold = (obj_z > min_height).float()
    height_reward = (1.0 - torch.tanh(torch.abs(obj_z - target_height) / std))
    gate = _get_proximity_gate(env, robot_cfg, object_cfg, gate_std)
    return gate * height_reward * above_threshold


def success_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    success_height: float = 0.880,
    gate_std: float = 0.13,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    is_lifted = (obj_z > success_height).float()
    gate = _get_proximity_gate(env, robot_cfg, object_cfg, gate_std)
    return gate * is_lifted


def joint_velocity_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    scale: float = 0.01,
) -> torch.Tensor:
    """Small reward for joint movement — prevents policy freezing."""
    robot: Articulation = env.scene[robot_cfg.name]
    joint_vel = robot.data.joint_vel[:, robot_cfg.joint_ids]
    return torch.norm(joint_vel, dim=-1).clamp(0.0, 5.0) * scale


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    ).clamp(0.0, 100.0)

def joint_pos_limit_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    
    lower_violation = (limits[..., 0] - joint_pos).clamp(0.0, 1.0)
    upper_violation = (joint_pos - limits[..., 1]).clamp(0.0, 1.0)
    
    total_violation = (lower_violation + upper_violation).sum(dim=-1)
    
    # Scale up so small violations are still meaningful
    return (total_violation)

def object_displacement_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    gate_std: float = 0.13,
) -> torch.Tensor:
    """
    Reward ANY movement of the cube when fingers are close.
    This acts as a contact proxy — if the cube moves, fingers are touching it.
    Direction doesn't matter yet — just make contact and disturb the cube.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    speed = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    gate = _get_proximity_gate(env, robot_cfg, object_cfg, gate_std)
    return gate * speed.clamp(0.0, 1.0)

def lift_height_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    resting_height: float = 0.850,
    max_height: float = 0.950,
    gate_std: float = 0.13,
) -> torch.Tensor:
    """Convex reward for lifting the object. Kills micro-bounce farming by squaring the progress."""
    obj = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    
    # 1. Calculate how far it has lifted
    lift_dist = (obj_z - resting_height).clamp(min=0.0)
    
    # 2. Normalize to a 0.0 to 1.0 scale
    normalized_height = (lift_dist / (max_height - resting_height)).clamp(max=1.0)
    
    # 3. Apply the Convex "Snowball" Curve (Square the normalized height)
    convex_reward = torch.pow(normalized_height, 2.0)
    
    # 4. Gate it by finger proximity so the robot must actually be holding it
    gate = _get_proximity_gate(env, robot_cfg, object_cfg, gate_std)
    
    return gate * convex_reward

def contact_detection_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    contact_dist: float = 0.07,
) -> torch.Tensor:
    """
    Smooth contact reward — exponential proximity weight × cube speed.
    Closer finger + moving cube = higher reward.
    No cliff edge, continuous gradient from any distance down to contact.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    
    tips_w = robot.data.body_pos_w[:, robot_cfg.body_ids]
    obj_pos = obj.data.root_pos_w
    
    dists = torch.norm(tips_w - obj_pos.unsqueeze(1), dim=-1)  # (N, 5)
    min_dist = dists.min(dim=-1).values  # closest finger distance to cube center
    
    # Smooth exponential proximity — peaks at 1.0 when finger at cube center
    # std=0.05 means at 5cm distance reward is ~0.37, at 2.5cm (surface) ~0.61
    proximity_weight = torch.exp(-min_dist / 0.05)
    
    # Cube moving in any direction — real contact signal
    cube_speed = torch.norm(obj.data.root_lin_vel_w, dim=-1).clamp(0.0, 1.0)
    
    return proximity_weight * cube_speed