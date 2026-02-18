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
