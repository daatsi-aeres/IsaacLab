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

    return (opposition * proximity_gate).mean(dim=-1)


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
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    resting_height: float = 0.850,
) -> torch.Tensor:
    """Reward making NEW height progress this episode — freezing earns zero."""
    obj: RigidObject = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    if not hasattr(env, "_max_cube_height"):
        env._max_cube_height = torch.full(
            (env.num_envs,), resting_height, device=env.device
        )

    progress = (obj_z - env._max_cube_height).clamp(0.0, 1.0)
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
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    resting_height: float = 0.850,
    max_height: float = 0.950,
) -> torch.Tensor:
    """Reward proportional to how high cube is above resting — every timestep."""
    obj: RigidObject = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    
    # Normalize height above resting to [0, 1]
    height_above_resting = (obj_z - resting_height).clamp(0.0, max_height - resting_height)
    normalized = height_above_resting / (max_height - resting_height)
    
    return normalized