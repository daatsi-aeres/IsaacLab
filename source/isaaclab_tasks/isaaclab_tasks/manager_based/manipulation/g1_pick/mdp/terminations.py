# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom termination functions for G1 picking environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_object_dropped(
    env: ManagerBasedRLEnv,
    minimum_height: float = -0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Termination condition when target object falls below minimum height.

    Args:
        env: The environment.
        minimum_height: Minimum height threshold (relative to environment origin).
        object_cfg: Scene entity for the target object.

    Returns:
        Tensor of shape (num_envs,): boolean termination flags.
    """
    target_object: RigidObject = env.scene[object_cfg.name]
    object_height = target_object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return object_height < minimum_height


def target_object_lifted(
    env: ManagerBasedRLEnv,
    success_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Termination condition when target object is lifted above success height."""
    target_object: RigidObject = env.scene[object_cfg.name]
    object_height = target_object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return object_height > success_height


def any_distractor_dropped(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    distractor_names: list[str],
) -> torch.Tensor:
    """Termination if ANY distractor in the list falls below minimum_height.
    Aggregates multiple distractors into a single termination signal for clean logging."""
    any_dropped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in distractor_names:
        obj: RigidObject = env.scene[name]
        any_dropped |= obj.data.root_pos_w[:, 2] < minimum_height
    return any_dropped
