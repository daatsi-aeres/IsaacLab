# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for G1 picking environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_object_position_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Target object position in the robot's root frame (privileged information).

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame).
        object_cfg: Scene entity for the target object.

    Returns:
        Tensor of shape (num_envs, 3): target object position [x, y, z] in robot root frame.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    target_object: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(
        robot.data.root_quat_w, target_object.data.root_pos_w - robot.data.root_pos_w
    )


def fingertip_positions_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fingertip positions in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot with fingertip bodies.

    Returns:
        Tensor of shape (num_envs, num_fingertips * 3): fingertip positions concatenated.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    # Get fingertip positions in world frame
    fingertip_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids]  # (num_envs, num_tips, 3)
    
    # Transform to robot base frame
    robot_pos_w = robot.data.root_pos_w.unsqueeze(1)  # (num_envs, 1, 3)
    robot_quat_w = robot.data.root_quat_w.unsqueeze(1)  # (num_envs, 1, 4)
    
    fingertip_pos_b = quat_apply_inverse(
        robot_quat_w.expand(-1, fingertip_pos_w.shape[1], -1).reshape(-1, 4),
        (fingertip_pos_w - robot_pos_w.expand(-1, fingertip_pos_w.shape[1], -1)).reshape(-1, 3)
    )
    
    return fingertip_pos_b.view(env.num_envs, -1)


def target_object_id(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """One-hot encoding of the target object ID (privileged information).

    This function assumes the environment stores target_object_ids as an attribute.

    Args:
        env: The environment.

    Returns:
        Tensor of shape (num_envs, max_objects): one-hot encoding of target object.
    """
    # This will be populated by the environment's reset logic
    if hasattr(env, "target_object_ids"):
        max_objects = 10  # Maximum number of objects in scene
        one_hot = torch.zeros(env.num_envs, max_objects, device=env.device)
        one_hot.scatter_(1, env.target_object_ids.unsqueeze(1), 1.0)
        return one_hot
    else:
        # Return zeros if not available (during initialization)
        return torch.zeros(env.num_envs, 10, device=env.device)
