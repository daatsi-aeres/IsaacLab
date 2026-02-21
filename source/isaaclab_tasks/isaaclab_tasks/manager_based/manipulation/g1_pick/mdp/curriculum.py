# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum learning functions for G1 picking environment."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PickingCurriculumScheduler(ManagerTermBase):
    """Adaptive difficulty scheduler with three-phase reward gating.

    Phases
    ------
    Phase 0 – Reaching only.
        The agent learns to move the RIGHT hand toward the target.
        Grasping / lifting / declutter / pick-success weights are all 0.

    Phase 1 – Grasping + Lifting + Declutter unlocked.
        Triggered when the rolling mean *reaching* reward per step exceeds
        ``phase1_reaching_threshold``.  The reward manager weights for
        "grasping_target", "lifting_target", and "declutter" are set to their
        configured values.

    Phase 2 – Pick-success unlocked.
        Triggered when the rolling mean *lifting* reward per step exceeds
        ``phase2_lifting_threshold``.  The "pick_success" weight is enabled.

    Difficulty tracks clutter count independently of phase and continues to
    ramp up throughout training.

    Curriculum params (all optional, tunable via CurriculumCfg)
    -----------------------------------------------------------
    init_difficulty          : int   = 0
    min_difficulty           : int   = 0
    max_difficulty           : int   = 60
    promotion_only           : bool  = False
    history_size             : int   = 500   # rolling episode window
    phase1_reaching_threshold: float = 0.5   # weighted reward/step > this → phase 1
    phase2_lifting_threshold: float = 0.75  # weighted reward/step > this → phase 2
    phase1_terms             : dict  = {"lifting_target": 5.0,
                                        "declutter": 2.0}
    phase2_terms             : dict  = {"pick_success": 1.0}
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        # ── Difficulty (clutter) ──────────────────────────────────────────────
        init_difficulty = cfg.params.get("init_difficulty", 0)
        self.current_difficulties = torch.ones(env.num_envs, device=env.device) * init_difficulty
        self.difficulty_frac = 0.0

        # ── Phase management ─────────────────────────────────────────────────
        self._phase = 0
        history_size = cfg.params.get("history_size", 500)
        self._reaching_history: deque[float] = deque(maxlen=history_size)
        self._lifting_history: deque[float] = deque(maxlen=history_size)

        self._phase1_threshold = cfg.params.get("phase1_reaching_threshold", 0.4)
        self._phase2_threshold = cfg.params.get("phase2_lifting_threshold", 0.75)

        self._phase1_terms: dict[str, float] = cfg.params.get(
            "phase1_terms",
            {"lifting_target": 5.0, "declutter": 2.0},
        )
        self._phase2_terms: dict[str, float] = cfg.params.get(
            "phase2_terms",
            {"pick_success": 1.0},
        )

        # Reward-manager term indices resolved lazily on first __call__
        self._rm_initialized = False
        self._reaching_key: str | None = None
        self._lifting_key: str | None = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _init_rm(self, env: ManagerBasedRLEnv) -> None:
        """Resolve reward term names → indices once the reward manager is ready."""
        rm = env.reward_manager
        names = list(rm._term_names)

        # Keys for reading episode sums
        self._reaching_key = "reaching_target" if "reaching_target" in names else None
        self._lifting_key = "lifting_target" if "lifting_target" in names else None

        # Indices for modifying weights
        self._phase1_indices: dict[str, int] = {
            n: names.index(n) for n in self._phase1_terms if n in names
        }
        self._phase2_indices: dict[str, int] = {
            n: names.index(n) for n in self._phase2_terms if n in names
        }
        self._rm_initialized = True

    def _set_phase_weights(self, env: ManagerBasedRLEnv, phase: int) -> None:
        """Write reward weights for the given phase into the reward manager config."""
        rm = env.reward_manager
        if phase >= 1:
            for name, idx in self._phase1_indices.items():
                rm._term_cfgs[idx].weight = self._phase1_terms[name]
                print(f"[PickingCurriculum] Phase 1 → enabled reward '{name}' "
                      f"(weight={self._phase1_terms[name]})")
        if phase >= 2:
            for name, idx in self._phase2_indices.items():
                rm._term_cfgs[idx].weight = self._phase2_terms[name]
                print(f"[PickingCurriculum] Phase 2 → enabled reward '{name}' "
                      f"(weight={self._phase2_terms[name]})")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_state(self) -> torch.Tensor:
        return self.current_difficulties

    def set_state(self, state: torch.Tensor) -> None:
        self.current_difficulties = state.clone().to(self._env.device)

    @property
    def phase(self) -> int:
        return self._phase

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
        lift_height_threshold: float = 0.15,
        init_difficulty: int = 0,
        min_difficulty: int = 0,
        max_difficulty: int = 60,
        promotion_only: bool = False,
        # The remaining params are read from cfg.params in __init__ instead:
        history_size: int = 500,
        phase1_reaching_threshold: float = 0.5,
        phase2_lifting_threshold: float = 0.75,
        phase1_terms: dict | None = None,
        phase2_terms: dict | None = None,
    ):
        """Update difficulty and advance the training phase when ready.

        Args:
            env: The environment.
            env_ids: Environments that are resetting this step.
            object_cfg: Scene entity for the target object.
            lift_height_threshold: Height above env origin to count as a pick.
            min/max_difficulty: Clamp range for clutter difficulty.
            promotion_only: If True, difficulty never decreases.
            (remaining args) : Declared so IsaacLab doesn't treat them as unknown;
                               their values are consumed in __init__ instead.
        """
        # ── Lazy reward-manager init ──────────────────────────────────────────
        if not self._rm_initialized:
            try:
                self._init_rm(env)
            except Exception:
                pass  # reward manager not ready yet; retry next call

        # ── Phase tracking ────────────────────────────────────────────────────
        if self._rm_initialized and len(env_ids) > 0:
            rm = env.reward_manager
            max_ep_len = max(env.max_episode_length, 1)

            # Reaching: always track
            if self._reaching_key and self._reaching_key in rm._episode_sums:
                ep_sums = rm._episode_sums[self._reaching_key][env_ids]
                for v in ep_sums.tolist():
                    self._reaching_history.append(float(v))

            # Lifting: only meaningful after phase 1 (weight > 0)
            if self._phase >= 1 and self._lifting_key and self._lifting_key in rm._episode_sums:
                ep_sums = rm._episode_sums[self._lifting_key][env_ids]
                for v in ep_sums.tolist():
                    self._lifting_history.append(float(v))

            # ── Phase advancement ─────────────────────────────────────────────
            MIN_HISTORY = 50  # require at least this many completed episodes

            if self._phase == 0 and len(self._reaching_history) >= MIN_HISTORY:
                mean_reaching = sum(self._reaching_history) / len(self._reaching_history)
                if mean_reaching >= self._phase1_threshold:
                    self._phase = 1
                    self._set_phase_weights(env, 1)
                    print(f"[PickingCurriculum] *** PHASE 0→1 *** "
                          f"(mean reaching sum={mean_reaching:.3f} ≥ {self._phase1_threshold})")

            elif self._phase == 1 and len(self._lifting_history) >= MIN_HISTORY:
                mean_lifting = sum(self._lifting_history) / len(self._lifting_history)
                if mean_lifting >= self._phase2_threshold:
                    self._phase = 2
                    self._set_phase_weights(env, 2)
                    print(f"[PickingCurriculum] *** PHASE 1→2 *** "
                          f"(mean lifting sum={mean_lifting:.3f} ≥ {self._phase2_threshold})")

        # ── Clutter difficulty ────────────────────────────────────────────────
        target_object: RigidObject = env.scene[object_cfg.name]
        object_height = target_object.data.root_pos_w[env_ids, 2] - env.scene.env_origins[env_ids, 2]
        success = object_height > lift_height_threshold

        demote = self.current_difficulties[env_ids] if promotion_only else (self.current_difficulties[env_ids] - 1)
        self.current_difficulties[env_ids] = torch.where(
            success,
            self.current_difficulties[env_ids] + 1,
            demote,
        ).clamp(min=min_difficulty, max=max_difficulty)

        self.difficulty_frac = torch.mean(self.current_difficulties) / max(max_difficulty, 1)
        return self.difficulty_frac

    def get_num_clutter_objects(self, env_id: int) -> int:
        """Return the number of clutter objects for the given environment."""
        difficulty = self.current_difficulties[env_id].item()
        if difficulty < 30:
            return 0
        elif difficulty < 40:
            return int(torch.randint(1, 3, (1,)).item())
        elif difficulty < 50:
            return int(torch.randint(3, 6, (1,)).item())
        else:
            return int(torch.randint(5, 8, (1,)).item())
