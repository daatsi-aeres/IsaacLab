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
