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
    """Reward ALL right-hand fingertips for moving close to the target object.

    Uses the MEAN across all fingertips rather than the minimum.  This forces
    every finger to converge on the object, creating the wrapping/encircling
    posture needed for a stable grasp.  A min() reward is satisfied by one
    fingertip poking the object — mean() is only maximised when all fingers
    surround it.

    Returns:
        Tensor (num_envs,): mean per-fingertip reward, values in [0, 1].
    """
    robot: Articulation = env.scene[robot_cfg.name]
    target_object: RigidObject = env.scene[object_cfg.name]

    fingertip_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids]   # (N, 5, 3)
    object_pos_w = target_object.data.root_pos_w.unsqueeze(1)         # (N, 1, 3)

    distances = torch.norm(fingertip_pos_w - object_pos_w, dim=-1)    # (N, 5)
    per_tip = 1.0 - torch.tanh(distances / std)                        # (N, 5)
    return per_tip.mean(dim=-1)                                        # (N,)


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
    scale: float = 0.08,
    robot_cfg: SceneEntityCfg | None = None,
    proximity_std: float | None = None,
) -> torch.Tensor:
    """Continuous lift reward, optionally coupled with fingertip proximity.

    Args:
        minimal_height: Baseline height [m above env origin] — set to the object's
            resting height so the reward is non-zero from the first millimetre of lift.
        scale: tanh denominator [m] — how quickly the height component saturates.
        robot_cfg: If provided (with right-hand fingertip body_ids), the reward is
            multiplied by a proximity term so that lift reward is only earned when
            the hand is simultaneously near the object.  This is the key coupling
            that prevents the robot ignoring the hand while getting lift reward
            from accidental object movement.
        proximity_std: tanh length scale [m] for the proximity multiplier.
            Typical value: 0.10 m.  Required when robot_cfg is provided.

    Returns:
        Tensor (num_envs,): values in [0, 1].
    """
    target_object: RigidObject = env.scene[object_cfg.name]
    object_height = target_object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    # lift = torch.tanh((object_height - minimal_height) / scale).clamp(0.0, 1.0)
    lift = ((object_height - minimal_height) / scale).clamp(0.0, 1.0)

    if robot_cfg is not None and proximity_std is not None:
        robot: Articulation = env.scene[robot_cfg.name]
        fingertip_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids]       # (N, 5, 3)
        object_pos_w = target_object.data.root_pos_w.unsqueeze(1)             # (N, 1, 3)
        min_dist = torch.norm(fingertip_pos_w - object_pos_w, dim=-1).min(dim=-1).values
        proximity = 1.0 - torch.tanh(min_dist / proximity_std)               # (N,)
        return lift * proximity

    return lift


def object_upward_velocity_reward(
    env: ManagerBasedRLEnv,
    scale: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Reward upward velocity of the target object (kept for reference, disable via weight=0)."""
    target_object: RigidObject = env.scene[object_cfg.name]
    vel_z = target_object.data.root_lin_vel_w[:, 2]
    return torch.tanh(vel_z / scale).clamp(0.0, 1.0)


def object_velocity_penalty(
    env: ManagerBasedRLEnv,
    scale: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Penalise total object speed (all axes) to discourage shaking/vibration.

    When the robot shakes the object to exploit a velocity reward, the total
    speed is high even though the net height gain is zero.  This penalty makes
    shaking strictly costly while still allowing deliberate slow lifting
    (low speed, sustained height gain).

    Args:
        scale: tanh denominator [m/s].  0.1 m/s → penalty ≈ 0.46 at that speed.
            Set lower (e.g. 0.05) to punish even small vibrations more harshly.

    Returns:
        Tensor (num_envs,): values in [-1, 0].
    """
    target_object: RigidObject = env.scene[object_cfg.name]
    speed = torch.norm(target_object.data.root_lin_vel_w, dim=-1)   # (N,) total speed
    return -torch.tanh(speed / scale)


def left_hand_declutter_reward(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distractor_names: list[str] | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    min_active_height: float = 0.65,
) -> torch.Tensor:
    """Reward the LEFT hand for moving distractors away from the target object.

    Two equally-weighted components:
      1. Near reward  – LEFT fingertips close to the nearest active distractor.
      2. Spread reward – mean distance of active distractors from the target.

    When no distractors are on the tray the function returns 0.

    Args:
        std: Length scale [m] for the near-reward tanh kernel.
        robot_cfg: Must contain the LEFT hand fingertip body_ids.
        distractor_names: List of scene entity names for distractor rigid objects.
        target_cfg: Scene entity for the target object.
        min_active_height: Minimum height above env origin [m] for a distractor to
            be considered "on the tray". Default 0.65 m — well above the ground
            plane (0 m) but below the tray surface (0.82 m). This prevents hidden
            distractors that were pushed up to the ground plane by physics from
            being mistakenly counted as active.

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
    env_origin_z = env.scene.env_origins[:, 2]                 # (N,)

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
        # Active = distractor is on/above the tray, not hiding below the table.
        # We check height relative to env origin so this works for all env grid
        # positions.  min_active_height=0.65 m is above the ground plane (0 m)
        # but below the tray surface (0.82 m), so a distractor resting on the
        # ground after physics pushed it up from z=-5 is NOT counted as active.
        obj_z_local = obj_pos[:, 2] - env_origin_z
        active = (obj_z_local > min_active_height).float()

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
