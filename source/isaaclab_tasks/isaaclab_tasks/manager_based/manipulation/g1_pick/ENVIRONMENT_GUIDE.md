# G1 Pick — Complete Environment Guide

> **Audience**: engineers reading or modifying this environment for the first time.
> **Goal**: understand every design decision, follow a training run from first gradient to pick success.

---

## 1. What this environment is

A G1 humanoid robot stands in front of a table with a tray on it. A brightly coloured target object (red sphere, cube, or capsule) is placed somewhere on the tray. Optionally, up to seven distractor objects (blue, green, yellow cubes) are placed around it. The robot must reach the target with its **right hand**, grasp it, and lift it cleanly off the tray — while using its **left hand** to sweep distractors out of the way.

The lower body is pinned in place (`fix_base=True` on the pelvis; legs driven at stiffness 10000 N·m/rad). Only the arms and hands move. This lets the training focus entirely on dexterous manipulation rather than balancing.

The episode runs for **10 seconds** at 120 Hz physics with a 2× decimation, giving **600 control steps** per episode at 60 Hz.

---

## 2. Physical scene layout

```
                      ┌───────────────────────────┐  ← tray top   z = 0.820 m
                      │  [distractor][TARGET][dist]│
                      │         (40×60 cm tray)    │
         ─────────────┴───────────────────────────-┘  ← table top  z = 0.800 m
         ─────────────────────────────────────────────  (solid 60×120×80 cm block)
         ─────────────────────────────────────────────  ← floor     z = 0.000 m

 Robot pelvis  x = -0.10 m  (pinned to world, fix_base = True)
 Table centre  x =  0.40 m  (front edge at x = 0.10 m, 20 cm from pelvis)
 Tray centre   x =  0.40 m  (same as table, sits on top)

Object spawn region on tray:  x ∈ [0.26, 0.54]   (±14 cm from centre)
                               y ∈ [-0.17, 0.17]   (±17 cm)
```

**Why solid table?** A solid block means no gap between the legs; the scene looks realistic and objects can't fall through the support structure.

---

## 3. Observation space (174 dimensions)

The policy receives a single flat vector every control step.

| Slice | Dims | Source | Why the policy needs it |
|-------|------|--------|-------------------------|
| `joint_pos` | 53 | All robot joints, relative to default | Full proprioception — where every arm, hand, leg joint sits right now |
| `joint_vel` | 53 | All robot joints | Rate of change — lets the policy damp oscillations and predict contact events |
| `target_object_position` | 3 | Target in robot root frame | The **goal** — right hand must navigate toward this point |
| `distractor_0_position` | 3 | Distractor 0 in robot root frame | Where the left hand's first declutter target is |
| `distractor_1_position` | 3 | Distractor 1 in robot root frame | When this is at z ≈ −5 m in robot frame it signals "hidden" |
| `distractor_2_position` | 3 | Distractor 2 in robot root frame | Same; left hand ignores hidden objects (reward also zeros out for hidden) |
| `left_fingertip_positions` | 15 | 5 left fingertips in robot frame (×3) | Left hand spatial state — needed for declutter gradients |
| `right_fingertip_positions` | 15 | 5 right fingertips in robot frame (×3) | Right hand spatial state — primary reaching / grasping hand |
| `actions` | 26 | Last control action sent | Gives the policy memory of what it just did — helps with smooth, stable motions |

**Total: 174**

### A note on "robot frame"
Positions are expressed in the robot's root frame (pelvis). Because the pelvis is fixed in world space the robot frame is effectively identical to world frame in this environment, but expressing observations in body frame keeps the policy robust if you ever un-fix the base later.

### Why leg joints are included even though they don't move
The full 53-DOF state is observed without masking. The leg joints are constant (locked by their actuator stiffness), so after a few early episodes the network simply learns to ignore them. Excluding them would require a custom joint-name filter that adds maintenance overhead.

---

## 4. Action space (26 dimensions)

`JointPositionActionCfg` with `scale=0.5` and `use_default_offset=True`.

| Group | Joints | Count |
|-------|--------|-------|
| Left arm | shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw | 7 |
| Right arm | same | 7 |
| Left hand leader joints | thumb 1&2, index/middle/ring/little 1 | 6 |
| Right hand leader joints | same | 6 |
| **Total** | | **26** |

Each action is a **target joint position offset** from the default pose. Scale 0.5 means the policy outputs values in roughly [−1, 1] and they map to ±0.5 rad offsets. This keeps initial actions from immediately hitting joint limits.

**Mimic joints are excluded.** The Inspire hand has passive mimic joints (`thumb_3/4`, `{finger}_2`) that follow leader joints mechanically. These are frozen at zero by the event manager and are not in the action space — controlling them would be redundant.

---

## 5. The managers

### 5.1 Event Manager — "what happens at every reset"

When an episode ends (timeout or object dropped) all five event terms fire **in order**:

```
reset_all  →  reset_robot_joints  →  freeze_lower_body  →  freeze_mimic_joints  →  reset_target_object  →  reset_clutter
```

**`reset_all`** resets every scene entity to its USD-defined default pose. This is the baseline — everything else refines it.

**`reset_robot_joints`** adds a small uniform random offset (±0.1 rad) to every arm and hand leader joint. This breaks symmetry and prevents the policy from memorising a single start posture. It is the primary source of **domain randomisation** for the robot state.

**`freeze_lower_body`** sets all leg / waist joints back to their default with zero offset and zero velocity. This undoes any drift in the lower body from `reset_all` and enforces the "frozen legs" contract.

**`freeze_mimic_joints`** zeros out the thumb/finger mimic joints. Because mimic joints are not in the action space the actuator may drift them slightly; this event snaps them back.

**`reset_target_object`** places the target object at a random position on the tray: `x ∈ [-0.14, 0.14]`, `y ∈ [-0.18, 0.18]` from tray centre, always at the same height (resting on tray surface). The random XY spread ensures the right hand must generalise its reaching strategy across the full reachable workspace.

**`reset_clutter`** reads the current difficulty from `PickingCurriculumScheduler` and decides how many distractors to place and where. At difficulty < 30 all three distractors are hidden below the table (z = −5 m). As difficulty grows, 1–7 distractors are scattered randomly on the tray near the target. This is the mechanism by which the scene becomes progressively harder.

---

### 5.2 Reward Manager — "what the policy is trying to maximise"

All rewards are computed every control step and summed to produce the scalar reward used for PPO updates.

#### Phase 0 rewards (always active)

**`reaching_target`** (weight 1.0)

```
reward = 1 − tanh(min_distance_right_fingertips_to_target / 0.1)
```

Range: [0, 1] per step. The `tanh` kernel gives a smooth gradient. At 10 cm away reward ≈ 0.24; at 2 cm away reward ≈ 0.82; touching reward ≈ 1.0. Only the **right hand's** five fingertips are measured. This is the first signal the policy ever receives and it shapes the entire reaching behaviour.

**`left_penalty`** (weight 0.3)

```
penalty = −(1 − tanh(min_distance_left_fingertips_to_target / 0.15))
```

Range: [−1, 0] per step. A mirror image of the reaching reward but for the left hand and negative. At 0.3 weight it is too small to strongly constrain the left hand early on, but it is enough to prevent the left hand from defaulting to "also grab the target" once grasping gradients appear. It gently steers the left hand away from the target's vicinity from the very first episode.

#### Phase 1 rewards (zero-weighted until curriculum unlocks)

**`grasping_target`** (unlocked at weight 3.0)

```
forces = |net_forces_w| per right-hand fingertip body
reward = 1.0   if  any finger force > 0.5 N
         0.0   otherwise
```

Binary. The right-hand contact sensor tracks five fingertip prims. The moment any one of them presses against anything with more than 0.5 N — likely the target after the reaching stage has been mastered — this fires and gives a reward three times larger than the reaching signal. This strong relative weight is intentional: grasping is much harder to discover than reaching, so it needs a proportionally larger incentive.

**`lifting_target`** (unlocked at weight 5.0)

```
reward = tanh((object_height − _LIFT_Z) / 0.02).clamp(0, 1)
```

Range: [0, 1]. `_LIFT_Z = 0.900 m` (10 cm above table top). The narrow tanh scale of 0.02 m means the reward rises steeply in the 2 cm band around the lift threshold. Below the threshold it is essentially zero; above it saturates at 1.0. Weight 5.0 makes this the dominant reward term, ensuring the policy eventually transitions from "grasp and hold" to "grasp and lift".

**`declutter`** (unlocked at weight 2.0)

Two equally weighted components:

```
near_reward   = max over active distractors:
                  (1 − tanh(left_fingertip_to_distractor / 0.12)) × active_flag

spread_reward = tanh(mean_distractor_to_target_distance / 0.15) × any_active_flag

reward = 0.5 × near_reward + 0.5 × spread_reward
```

When no distractors are on the table (`active_flag = 0`) both components are zero — this reward is silent during easy difficulty levels so it doesn't confuse the left hand. When distractors appear:
- `near_reward` pulls the left hand toward the nearest distractor (same kernel shape as reaching).
- `spread_reward` rewards the *outcome* — distractors being far from the target. The policy is rewarded for achieving separation, not just touching.

The combination means the left hand learns a two-part skill: approach a distractor, then push it far enough that spread reward increases.

#### Phase 2 rewards (zero-weighted until curriculum unlocks)

**`pick_success`** (unlocked at weight 1.0)

```
hold_time += step_dt   if object is above _LIFT_Z
hold_time  = 0         otherwise
reward = 10.0          if hold_time ≥ 1.0 s
         0.0           otherwise
```

Sparse 10× bonus. This fires once the robot has lifted the target and held it for a full second. It provides a clear terminal success signal that the PPO value function can propagate backward into the holding behaviour. Weight 1.0 may seem small but the ×10 multiplier makes each success worth 10 steps of maximum lifting reward.

#### Always-active penalties

**`action_rate`** (weight −0.0001): penalises `||a_t − a_{t−1}||²`. Prevents jittery bang-bang control. Small enough not to dominate but present from step 0.

**`joint_vel`** (weight −0.0001): penalises `||q̇||²`. Encourages smooth, slow motions near the object rather than flailing.

---

### 5.3 Curriculum Manager — "how the environment gets harder over time"

`PickingCurriculumScheduler` runs at every episode reset and does two independent things: **phase gating** and **clutter difficulty**.

#### Phase gating

The curriculum tracks a rolling history of episode-level reward sums, divided by episode length to give a "per step" metric.

```
Phase 0  (start)
   │  reaching_target / episode_length  ──rolling mean──►  ≥ 0.5
   ▼
Phase 1  (grasping + lifting + declutter unlocked)
   │  grasping_target / episode_length  ──rolling mean──►  ≥ 0.75
   ▼
Phase 2  (pick_success unlocked)
```

The phase is **global** (all environments share the same phase). A minimum of 50 completed episodes are required before a phase check fires, so the history is never noisy. When a phase advances, `PickingCurriculumScheduler._set_phase_weights()` directly writes into `env.reward_manager._term_cfgs[i].weight` — no restart required, the weights change live during training.

A console line is printed at every transition:
```
[PickingCurriculum] *** PHASE 0→1 *** (mean reaching/step=0.512 ≥ 0.5)
[PickingCurriculum] Phase 1 → enabled reward 'grasping_target' (weight=3.0)
...
```
This makes it easy to see in the training log exactly when each phase unlocked.

#### Clutter difficulty

Independently of phase, each environment has its own `difficulty` value (0–60). At every episode reset:
- If the object was above `_LIFT_Z` at reset time → difficulty += 1 (success)
- Otherwise → difficulty −= 1 (failure)

`reset_clutter` reads this difficulty per-environment and places distractors accordingly:

| difficulty | clutter objects on tray |
|------------|------------------------|
| 0–29 | 0 (clean tray) |
| 30–39 | 1–2 random |
| 40–49 | 3–5 random |
| 50–60 | 5–7 random |

Each environment independently ramps its difficulty. Environments that master the task first start getting harder scenes while struggling environments keep a clear tray. This **per-environment pacing** keeps the training distribution well populated at all difficulty levels throughout training.

---

### 5.4 Termination Manager — "when an episode ends"

| Term | Condition | Notes |
|------|-----------|-------|
| `time_out` | step count ≥ 600 | Normal episode end; counted as non-terminal for value bootstrap |
| `target_dropped` | object height < 0.5 m | Object fell off the table; terminates immediately. Prevents wasted time after an irrecoverable state |

The drop threshold (0.5 m) is comfortably below the tray surface (0.82 m), so a resting object never triggers it. It only fires when the object falls completely off the table or is knocked far enough below the tray to be unrecoverable.

---

## 6. The ideal training run — a narrative walkthrough

### Episode 1–50: the policy is random, but reaching gradients form

On episode 1 all weights are random; the robot flails its arms. The **only** non-zero reward signals are `reaching_target` and `left_penalty`. The right hand produces a continuous dense gradient: every millimetre closer to the target earns more reward. Within a few hundred episodes the policy begins developing a crude reaching behaviour — the right wrist swings toward the object's XY position and the elbow flexes to bring the hand down to table height.

The `left_penalty` quietly discourages the left hand from drifting toward the target. At weight 0.3 it is not strong enough to actively steer the left hand, but it breaks the symmetry that would otherwise cause both hands to converge on the same point.

Because the object is spawned at a random tray position each episode, the policy is already learning a generalised "move right hand toward [x, y, z]" skill, not memorising a fixed motion.

### Episode 50–200: reaching stabilises, phase 0→1 transition fires

By episode 50 the rolling history has enough samples for a phase check. Once the mean normalised reaching reward exceeds 0.5 (right fingertips within ~5 cm of target for more than half of each episode) the curriculum fires the phase 1 unlock.

Three things change instantly:
1. `grasping_target` weight becomes 3.0.
2. `lifting_target` weight becomes 5.0.
3. `declutter` weight becomes 2.0.

The policy now has a strong incentive to not just *approach* the target but to *press against* it. The contact sensor fires the moment any right fingertip pushes with ≥ 0.5 N, which typically happens when the hand is already near the object from the reaching behaviour. The new grasping reward is 3× the reaching reward, so the value function immediately starts attributing much higher value to states where contact is occurring.

Simultaneously `declutter` begins accumulating. At phase 1 entry there are still no distractors (difficulty < 30 for most envs), so `declutter` returns 0 and has no effect yet — but the term is live and ready.

### Episode 200–600: grasping consolidates, wrist orientation adapts

The policy finds that holding the object firmly (constant contact) while keeping the wrist angled to wrap around it produces more sustained grasping reward than repeated tap-and-release. The wrist pitch and yaw joints begin specialising: the policy learns to pronate the wrist so the palm faces down, then close the fingers.

`lifting_target` starts contributing as soon as the hand lifts the object even slightly off the tray. The tanh kernel creates a strong gradient in the 2 cm zone around `_LIFT_Z`. The policy discovers that lifting further is strictly better than just holding, and the lifting weight of 5.0 — the largest single reward term — pushes the elbow extension and shoulder pitch combination that actually raises the object.

### Episode 600–1500: phase 1→2 transition, pick-success unlocked

As grasping and lifting consolidate, the rolling grasping reward climbs. When the mean exceeds 0.75 per step (right hand in contact for ~25% of steps, at weight 3.0) the phase 2 unlock fires. `pick_success` weight becomes 1.0.

Now the policy is rewarded for holding the object above `_LIFT_Z` for a sustained second. The 10× multiplier makes this a large reward relative to all other terms combined. The value function starts assigning very high value to "object aloft, hand wrapped around it, no wrist wobble" states. The policy stabilises its grip, learns to resist the object's weight, and develops a quiet hold posture rather than continuing to move.

### Episode 1500+: clutter appears, left hand becomes active

By this point most environments have reached difficulty 30–40. The event manager starts placing 1–2 distractors on the tray. Now `declutter` reward becomes non-zero.

At first the policy ignores distractors — the reaching and grasping signals dominate and the left hand was already penalised away from the target. But the `near_reward` component of `declutter` creates a gradient pulling the left hand toward distractor objects. The policy begins to notice that reaching distractors while grasping with the right hand earns extra reward.

The `spread_reward` component then teaches *what to do* once the left hand touches a distractor: push it away from the target. This requires planning — the left hand must make contact, then exert force in the correct direction. The policy develops a sweeping motion: left arm extends, palm contacts a distractor, elbow extends further to push laterally.

As difficulty climbs to 40–50 and then 50–60, the tray becomes increasingly crowded. The left hand learns to prioritise the distractor closest to the target, push it to the edge of the tray, and then re-engage for the next one. The right hand has learned to hold its position once the target is grasped, waiting while the left hand clears space.

### Fully trained behaviour

In a successfully trained policy you would observe:

1. **Reset**: arms begin near the default pose (slightly above table, elbows bent forward).
2. **Right arm descend**: shoulder pitch adjusts, elbow extends toward table, wrist pronates toward target XY.
3. **Finger close**: as the hand nears the object, finger joints close to wrap around it.
4. **Contact and grip**: fingertips register force. Grasping reward fires continuously.
5. **Lift**: elbow extends upward, shoulder rolls back, object clears the tray surface.
6. **Hold**: wrist stabilises, object remains aloft. After 1 second the pick-success bonus fires.
7. **Left hand declutter** (high difficulty): while the right hand executes steps 2–6, the left arm sweeps distractors away, creating a clear path for the right hand's approach.

---

## 7. Key numbers at a glance

| Parameter | Value |
|-----------|-------|
| Observation dim | 174 |
| Action dim | 26 |
| Control frequency | 60 Hz (120 Hz physics, 2× decimation) |
| Episode length | 10 s = 600 steps |
| Lift threshold (`_LIFT_Z`) | 0.900 m above world origin |
| Drop threshold (`_DROP_Z`) | 0.500 m |
| Phase 0→1 trigger | rolling mean reaching/step ≥ 0.50 |
| Phase 1→2 trigger | rolling mean grasping/step ≥ 0.75 |
| History window | 500 completed episodes |
| Max clutter objects | 7 |
| Difficulty range | 0–60 |

---

## 8. TensorBoard — what to look for

```bash
./isaaclab.sh -p -m tensorboard.main --logdir logs/rl_games/G1Pick
```

| Signal | Healthy trajectory |
|--------|--------------------|
| `Episode/rewards/step` | Climbs in 3 visible steps corresponding to phase unlocks |
| `Episode/episode_lengths` | Rises from ~32 toward 600 as terminations become rarer |
| `losses/actor_loss` | Decreases, plateaus; spikes briefly at phase transitions (normal) |
| `info/kl` | Stays near 0.016 (our threshold); if it spikes consistently, reduce learning rate |
| Phase unlock lines in stdout | Grep for `PHASE` to find transition epochs |

---

## 9. Tuning guide

| If you see... | Try... |
|---------------|--------|
| Phase 0 never advances | Lower `phase1_reaching_threshold` (default 0.5) |
| Phase 1 never advances | Lower `phase2_grasping_threshold` (default 0.75) |
| Phase 0 advances too early | Raise `phase1_reaching_threshold` |
| Left hand ignores distractors | Raise `declutter` weight (currently 2.0) |
| Left hand constantly touches target | Raise `left_penalty` weight (currently 0.3) |
| Object immediately falls (OOM recovery) | Lower `--num_envs`; `replicate_physics=False` uses more VRAM per env |
| Grasping never fires (reward stuck at 0) | Check right-hand fingertip prim paths in contact sensor config |
