# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
G1 Object Picking Environment with Curriculum Learning.

This environment trains a G1 humanoid robot to pick target objects from a cluttered tray.
The robot stands with frozen lower body and uses its arms and dexterous hands to manipulate objects.
"""

import gymnasium as gym

from . import agents
from .g1_pick_env_cfg import G1PickEnvCfg, G1PickEnvCfg_PLAY

##
# Register Gym environments.
##

gym.register(
    id="Isaac-G1-Pick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1PickEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PickPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Pick-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1PickEnvCfg_PLAY,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PickPPORunnerCfg",
    },
)
