# rewards.py  — fresh, right-arm-only, no contact sensors

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
    """
    Stage 1: Pull ALL right fingertips toward the cube.
    Uses MEAN across tips (not min) — forces full encirclement, not one-finger poke.
    Returns values in [0, 1].
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    tips_w = robot.data.body_pos_w[:, robot_cfg.body_ids]       # (N, 5, 3)
    obj_pos = obj.data.root_pos_w.unsqueeze(1)                   # (N, 1, 3)
    dists = torch.norm(tips_w - obj_pos, dim=-1)                 # (N, 5)
    return (1.0 - torch.tanh(dists / std)).mean(dim=-1)          # (N,)


def _get_proximity_gate(env, robot_cfg, object_cfg, gate_std):
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    tips_w = robot.data.body_pos_w[:, robot_cfg.body_ids]
    obj_pos = obj.data.root_pos_w.unsqueeze(1)
    dists = torch.norm(tips_w - obj_pos, dim=-1)
    mean_prox = (1.0 - torch.tanh(dists / gate_std)).mean(dim=-1)
    return torch.sigmoid((mean_prox - 0.5) * 10.0)


def lift_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    resting_height: float = 0.850,
    lift_scale: float = 0.05,
    gate_std: float = 0.08,
) -> torch.Tensor:
    """
    Stage 2: Reward the cube rising — BUT ONLY when fingers are in a grasp posture.
    This is the critical gap-filler. The gate ensures the policy can't get lift
    reward by hitting the cube upward with one finger.

    reward = gate(fingertip_proximity) * clamp((z - resting_height) / lift_scale, 0, 1)

    With lift_scale=0.05, the reward saturates at 5 cm of lift. This keeps the
    gradient steep early on (every mm of lift matters) rather than using tanh
    which is too flat near zero.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    lift = ((obj_z - resting_height) / lift_scale).clamp(0.0, 1.0)

    gate = _get_proximity_gate(env, robot_cfg, object_cfg, gate_std)
    return gate * lift


def hold_height_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    target_height: float = 0.950,
    std: float = 0.05,
    gate_std: float = 0.08,
) -> torch.Tensor:
    """
    Stage 3: Reward holding the cube near a target height.
    Also gated — no free reward for cube sitting on table near target z.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    height_reward = 1.0 - torch.tanh(torch.abs(obj_z - target_height) / std)

    gate = _get_proximity_gate(env, robot_cfg, object_cfg, gate_std)
    return gate * height_reward


def success_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    success_height: float = 0.920,
    gate_std: float = 0.08,
) -> torch.Tensor:
    """
    Sparse +1 when cube is above success_height AND fingers are in grasp posture.
    Kept small (weight=5) — the dense lift_reward does the heavy lifting.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    is_lifted = (obj_z > success_height).float()

    gate = _get_proximity_gate(env, robot_cfg, object_cfg, gate_std)
    return gate * is_lifted


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalise jerky actions. Small weight — don't let this dominate."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    ).clamp(0.0, 100.0)

def finger_closure_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    max_closure_dist: float = 0.08,
) -> torch.Tensor:
    """
    Reward fingers for being on OPPOSITE SIDES of the cube.
    This directly rewards grasp geometry rather than proximity.
    
    A hand hovering above the cube gets zero.
    A hand wrapping around the cube gets high reward.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    
    tips_w = robot.data.body_pos_w[:, robot_cfg.body_ids]  # (N, 5, 3)
    obj_pos = obj.data.root_pos_w                           # (N, 3)
    
    # Vector from object to each fingertip
    tip_vectors = tips_w - obj_pos.unsqueeze(1)             # (N, 5, 3)
    
    # Normalize to get directions
    tip_dirs = torch.nn.functional.normalize(tip_vectors, dim=-1)  # (N, 5, 3)
    
    # For each pair of fingers, reward if they point in OPPOSITE directions
    # (dot product negative = fingers on opposite sides of cube)
    # Check thumb vs each other finger — most important opposition
    thumb_dir = tip_dirs[:, 0, :]                          # (N, 3)
    other_dirs = tip_dirs[:, 1:, :]                        # (N, 4, 3)
    
    # Dot product thumb vs each finger: -1 = perfect opposition, +1 = same side
    dots = (thumb_dir.unsqueeze(1) * other_dirs).sum(dim=-1)  # (N, 4)
    
    # Reward opposition: map [-1, 1] → [1, 0]
    opposition = (1.0 - dots) / 2.0                        # (N, 4), 1=opposite, 0=same
    
    # Only count opposition when fingers are actually close to cube
    dists = torch.norm(tips_w - obj_pos.unsqueeze(1), dim=-1)  # (N, 5)
    close_enough = (dists < max_closure_dist).float()
    thumb_close = close_enough[:, 0:1]                     # (N, 1)
    others_close = close_enough[:, 1:]                     # (N, 4)
    proximity_gate = thumb_close * others_close            # (N, 4)
    
    return (opposition * proximity_gate).mean(dim=-1)      # (N,)