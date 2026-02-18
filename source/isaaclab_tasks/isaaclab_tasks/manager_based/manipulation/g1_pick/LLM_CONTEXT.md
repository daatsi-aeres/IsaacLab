# G1 Pick — Full Codebase Context for LLM

## Goal
Train a G1 humanoid robot (with Inspire FTP dexterous hands) to pick a target
object off a tray, using IsaacLab ManagerBasedRLEnv + PPO (rl_games).
Lower body is fixed (`fix_base=True`); only arms + hands are controlled.
Curriculum progressively adds clutter objects as picking success improves.

---

## Framework: IsaacLab ManagerBasedRLEnv

Key patterns used throughout:
- `@configclass` dataclasses wired through `scene / actions / observations / rewards / events / terminations / curriculum` managers.
- `SceneEntityCfg(name, body_names=[...])` — resolved at runtime; `body_ids` populated by manager.
- `reset_root_state_uniform`: new_pos = `default_root_state[:3] + env_origin + rand_sample` (additive).
- `ImplicitActuatorCfg` — PhysX PD; `effort_limit_sim` caps max torque.
- `ContactSensorCfg.prim_path` uses full Python regex via `find_matching_prims` (re module).
- `ContactSensor.data.net_forces_w` shape `(N_envs, N_bodies, 3)` — always populated.
  `force_matrix_w` is only populated when `filter_prim_paths_expr` is non-empty.
- `ManagerTermBase` subclass as curriculum term: manager calls `__call__(env, env_ids, **params)`;
  after init `term_cfg.func` holds the instance (not the class).

---

## Directory layout

```
g1_pick/
├── __init__.py              # gym registration
├── robot_cfg.py             # G1_INSPIRE_CFG (ArticulationCfg from URDF)
├── g1_pick_env_cfg.py       # Main env config (scene + all MDP managers)
├── agents/
│   └── rl_games_ppo_cfg.yaml
└── mdp/
    ├── __init__.py          # from isaaclab.envs.mdp import *; from .* import *
    ├── curriculum.py        # PickingCurriculumScheduler(ManagerTermBase)
    ├── events.py            # reset_clutter_based_on_difficulty
    ├── observations.py      # target_object_position_b, fingertip_positions_b
    ├── rewards.py           # reaching/grasping/lifting/success/penalty
    └── terminations.py      # target_object_dropped
```

---

## FILE: robot_cfg.py

```python
# URDF: G1 29-DOF body + Inspire FTP hands (24 hand DOF, 12 per hand)
# Mimic joints in URDF (thumb_3/4, *_2 fingers) — PhysX ignores <mimic>;
# treated as independent DOFs, excluded from action space, held at 0 via event.
# fix_base=True: pelvis pinned, lower body cosmetic only.

_URDF_PATH = "/home/daatsi-aeres/ARCLab_ws/unitree_ros/robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"

G1_INSPIRE_CFG = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path=_URDF_PATH,
        activate_contact_sensors=True,  # PhysxContactReporter on all bodies
        fix_base=True,
        merge_fixed_joints=True,        # collapses sensor fixed-links into finger links
        self_collision=False,
        make_instanceable=True,
        joint_drive=None,               # use URDF values for USD conversion; overridden by actuators
        rigid_props=RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0, angular_damping=0,
                                           max_linear_velocity=1000, max_angular_velocity=1000,
                                           max_depenetration_velocity=1.0, retain_accelerations=False),
        articulation_props=ArticulationRootPropertiesCfg(enabled_self_collisions=False,
                                                          solver_position_iteration_count=8,
                                                          solver_velocity_iteration_count=4),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.1, 0.0, 0.74),          # pelvis; table front edge at x=0.1 → 0.2m clearance
        joint_pos={
            # Legs (cosmetic, fix_base=True)
            ".*_hip_pitch_joint": -0.20, ".*_knee_joint": 0.42, ".*_ankle_pitch_joint": -0.23,
            # Waist
            "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
            # Arms pre-positioned over tray
            # G1 elbow convention: NEGATIVE=forward(table), POSITIVE=backward; range -1.05..+2.09
            ".*_shoulder_pitch_joint": -0.5,    # updated by user
            "left_shoulder_roll_joint": -0.1,
            "right_shoulder_roll_joint": 0.1,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.3,              # user-tuned; negative=forward toward table
            ".*_wrist_roll_joint": 0.0,
            ".*_wrist_pitch_joint": -0.8,       # palms downward
            ".*_wrist_yaw_joint": 0.0,
            # Hands: fully open (mimic joints default to 0)
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Frozen lower body (high stiffness via __post_init__ deepcopy)
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint",".*_hip_roll_joint",".*_hip_pitch_joint",
                               ".*_knee_joint","waist_yaw_joint","waist_roll_joint","waist_pitch_joint"],
            effort_limit_sim=300, stiffness={".*_hip_yaw_joint":150,".*_hip_roll_joint":150,
                ".*_hip_pitch_joint":200,".*_knee_joint":200,"waist_.*_joint":200},
            damping={".*_hip_yaw_joint":5,".*_hip_roll_joint":5,".*_hip_pitch_joint":5,
                     ".*_knee_joint":5,"waist_.*_joint":5}),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint",".*_ankle_roll_joint"],
            effort_limit_sim=20, stiffness=20.0, damping=2.0),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint",".*_shoulder_roll_joint",".*_shoulder_yaw_joint",
                               ".*_elbow_joint",".*_wrist_roll_joint",".*_wrist_pitch_joint",".*_wrist_yaw_joint"],
            effort_limit_sim=25, stiffness=40.0, damping=10.0),
        # All 12 hand joints per hand in actuator group.
        # Policy controls only 6 leader joints (thumb_1/2, index/middle/ring/little_1).
        # 6 mimic joints (thumb_3/4, *_2) excluded from action space; held at 0 by stiffness+event.
        "left_hand": ImplicitActuatorCfg(
            joint_names_expr=["left_thumb_1_joint","left_thumb_2_joint","left_thumb_3_joint","left_thumb_4_joint",
                               "left_index_1_joint","left_index_2_joint","left_middle_1_joint","left_middle_2_joint",
                               "left_ring_1_joint","left_ring_2_joint","left_little_1_joint","left_little_2_joint"],
            effort_limit_sim=10, stiffness=20.0, damping=2.0),
        "right_hand": ImplicitActuatorCfg(  # same pattern, right_ prefix
            joint_names_expr=["right_thumb_1_joint","right_thumb_2_joint","right_thumb_3_joint","right_thumb_4_joint",
                               "right_index_1_joint","right_index_2_joint","right_middle_1_joint","right_middle_2_joint",
                               "right_ring_1_joint","right_ring_2_joint","right_little_1_joint","right_little_2_joint"],
            effort_limit_sim=10, stiffness=20.0, damping=2.0),
    },
)
```

---

## FILE: g1_pick_env_cfg.py

### Scene geometry
```
table  : size=(0.6x1.2x0.80), centre=(0.4, 0, 0.40) → top=0.800m, front edge x=0.10
tray   : size=(0.4x0.6x0.02), centre=(0.4, 0, 0.810) → top=0.820m  (grey)
robot  : pelvis at (-0.1, 0, 0.74), fix_base=True
target : MultiAssetSpawner [red cube(0.05³), red sphere(r=0.03), red capsule(r=0.02,h=0.06)]
         random_choice=True → different shape per env; init_pos=(0.4, 0, 0.850)
distractor_0: blue  cube (0.05³), hidden at z=-5 until difficulty≥30
distractor_1: green cube (0.05×0.07×0.04), hidden until difficulty≥40
distractor_2: yellow cube (0.04×0.04×0.07), hidden until difficulty≥50
contact sensors: left_contact_sensor  prim={ENV}/Robot/left_(thumb_4|index_2|middle_2|ring_2|little_2)
                 right_contact_sensor prim={ENV}/Robot/right_(thumb_4|index_2|middle_2|ring_2|little_2)
                 history_length=3, debug_vis=True
```

### Height constants
```python
_OBJ_INIT_Z = 0.850  # tray_top(0.820) + half-size(0.025) + gap(0.005)
_LIFT_Z     = 0.900  # 10cm above table → counts as lifted
_DROP_Z     = 0.500  # below → terminate (object fell off table)
```

### ActionsCfg — 26 DOF total
```python
# arm_action: JointPositionActionCfg, scale=0.5, use_default_offset=True
joint_names = [
    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint", ".*_wrist_pitch_joint", ".*_wrist_yaw_joint",   # 7×2 = 14 arm DOF
    ".*_thumb_1_joint", ".*_thumb_2_joint",
    ".*_index_1_joint", ".*_middle_1_joint", ".*_ring_1_joint", ".*_little_1_joint",  # 6×2 = 12 hand DOF
]
```

### ObservationsCfg (policy group, concatenated)
```
joint_pos          : joint_pos_rel()
joint_vel          : joint_vel_rel()
target_object_pos  : target_object_position_b(robot, target_object) → (N,3) in robot frame
left_fingertip_pos : fingertip_positions_b(robot, bodies=[left_thumb_4,left_index_2,...]) → (N,15)
right_fingertip_pos: fingertip_positions_b(robot, bodies=[right_thumb_4,...]) → (N,15)
actions            : last_action()
```

### EventCfg (all mode="reset")
```
reset_all          : reset_scene_to_default
reset_robot_joints : reset_joints_by_offset ±0.1 rad on arms + hand leaders
freeze_lower_body  : reset_joints_by_offset ±0.0 on hip/knee/ankle/waist
freeze_mimic_joints: reset_joints_by_offset ±0.0 on thumb_3/4, *_2 finger joints
reset_target_object: reset_root_state_uniform, pose_range x:±0.14, y:±0.18, z:(0.01,0.01)
reset_clutter      : reset_clutter_based_on_difficulty(distractor_names, table_center_x=0.4, ...)
```

### RewardsCfg
```
reaching_target  w=1.0  : 1-tanh(min_fingertip_dist/0.1)  — all 10 fingertips vs target
grasping_target  w=3.0  : 1.0 if both left_contact_sensor AND right_contact_sensor force>0.5N
lifting_target   w=5.0  : tanh((obj_z-env_origin_z - _LIFT_Z)/0.02).clamp(0,1)
pick_success     w=1.0  : 10.0 bonus if object held > _LIFT_Z for ≥1.0 s continuous
action_rate      w=-1e-4: action_rate_l2
joint_vel        w=-1e-4: joint_vel_l2 on robot
```

### TerminationsCfg
```
time_out      : episode_length_s=10.0
target_dropped: obj_z - env_origin_z < _DROP_Z (0.5m)
```

### CurriculumCfg
```
picking_curriculum: PickingCurriculumScheduler
  params: object_cfg=target_object, lift_height_threshold=_LIFT_Z,
          init_difficulty=0, min=0, max=60, promotion_only=False
```

### G1PickEnvCfg.__post_init__
```python
decimation=2, episode_length_s=10.0, sim.dt=1/120 (120Hz physics, 60Hz policy)
sim.physx.bounce_threshold_velocity=0.2
sim.physx.gpu_max_rigid_patch_count = 4*5*2**15

# Freeze lower body: deepcopy actuators to avoid mutating global G1_INSPIRE_CFG
# (shallow .replace() would share dict values)
for group in ["legs", "feet"]:
    act = copy.deepcopy(self.scene.robot.actuators[group])
    act.stiffness = 10000.0; act.damping = 1000.0; act.effort_limit_sim = 10000.0
    self.scene.robot.actuators[group] = act
```

### G1PickEnvCfg_PLAY
```python
scene.num_envs = 64
curriculum.picking_curriculum.params["init_difficulty"] = 60
```

---

## FILE: mdp/curriculum.py

```python
class PickingCurriculumScheduler(ManagerTermBase):
    # Stores per-env difficulty tensor (float, 0–60).
    # __init__: self.current_difficulties = ones(num_envs) * init_difficulty
    # __call__(env, env_ids, ...):
    #   success = object_height[env_ids] > lift_height_threshold
    #   difficulty += 1 if success else -1 (or 0 if promotion_only)
    #   clamp(min, max)
    #   return mean(difficulties)/max_difficulty  (difficulty_frac float)
    # Clutter thresholds: 0=no clutter(diff<30), 1(diff≥30), 2(diff≥40), 3(diff≥50)
    # Accessed in events.py via: cm._term_names.index("picking_curriculum") → cm._term_cfgs[idx].func
```

---

## FILE: mdp/events.py

```python
def reset_clutter_based_on_difficulty(env, env_ids, distractor_names, tray_surface_height,
                                       hidden_height=-5.0, tray_x_half=0.14, tray_y_half=0.17,
                                       table_center_x=0.4):
    # Read per-env difficulty via cm._term_cfgs[idx].func.current_difficulties (try/except)
    # For each distractor_i: activation_threshold = 30 + i*10
    # is_active[env] = difficulty[env] >= threshold
    # active → random pos on tray; inactive → hidden at z=env_origin_z + hidden_height
    # Writes via obj.write_root_pose_to_sim / write_root_velocity_to_sim
```

---

## FILE: mdp/observations.py

```python
def target_object_position_b(env, robot_cfg, object_cfg) -> (N,3):
    # quat_apply_inverse(robot.root_quat_w, obj.root_pos_w - robot.root_pos_w)

def fingertip_positions_b(env, robot_cfg) -> (N, num_tips*3):
    # fingertip_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids]
    # transform to robot root frame; return flattened
```

---

## FILE: mdp/rewards.py

```python
def target_object_reaching_reward(env, std, robot_cfg, object_cfg):
    # fingertip_pos_w = body_pos_w[:, body_ids]  shape (N, 10, 3)
    # distances = norm(fingertips - obj_pos.unsqueeze(1), dim=-1)  → (N, 10)
    # return 1 - tanh(min_distance / std)

def target_object_grasping_reward(env, threshold, contact_sensor_names):
    # For each sensor: force = norm(net_forces_w, dim=-1).max(dim=-1).values  → (N,)
    # stack → (N, 2); count sensors > threshold; return (count >= 2).float()
    # NOTE: net_forces_w used (not force_matrix_w which requires filter_prim_paths_expr)

def target_object_lift_reward(env, minimal_height, object_cfg):
    # height = obj.root_pos_w[:,2] - env_origins[:,2]
    # return tanh((height - minimal_height) / 0.02).clamp(0, 1)

def non_target_penalty(env, threshold, contact_sensor_names):
    # sum of (max_finger_force > threshold).float() per sensor, negated

def pick_success_reward(env, minimal_height, hold_time_threshold, object_cfg):
    # Tracks env.object_hold_time (created lazily as zeros on first call)
    # Accumulates step_dt while lifted; resets to 0 when not lifted
    # Returns 10.0 * (hold_time >= threshold).float()
```

---

## FILE: mdp/terminations.py

```python
def target_object_dropped(env, minimum_height=_DROP_Z, object_cfg):
    return (obj.root_pos_w[:,2] - env_origins[:,2]) < minimum_height
```

---

## FILE: agents/rl_games_ppo_cfg.yaml (key settings)

```yaml
algo: a2c_continuous (PPO)
network: actor_critic MLP [512, 256, 128] ELU, fixed_sigma=True
gamma=0.99, tau=0.95, lr=3e-4 (adaptive KL, kl_threshold=0.016)
horizon_length=32, minibatch_size=32768, mini_epochs=8
normalize_input=True, normalize_value=True, normalize_advantage=True
reward_shaper scale=0.01, entropy_coef=0.0, e_clip=0.2
```

---

## Known design decisions & gotchas

| Topic | Detail |
|---|---|
| URDF mimic joints | PhysX ignores `<mimic>` tags. thumb_3/4 and `*_2` finger joints become free DOFs. Excluded from action space; freeze_mimic_joints event resets them to 0 each episode. |
| `force_matrix_w` vs `net_forces_w` | Only `net_forces_w` is populated without `filter_prim_paths_expr`. Use `net_forces_w` in rewards. |
| Lower body freeze | Must `copy.deepcopy()` actuator configs before mutation; shallow `.replace()` shares the dict and mutates the global singleton. |
| `joint_drive=None` | Required in UrdfFileCfg to avoid `MISSING` validation error; gains.stiffness has no default. |
| Clutter access | `env.curriculum_manager._term_cfgs[idx].func` holds the scheduler instance after manager init; no public `get_term()` API. |
| Grasping reward is non-specific | Contact sensors detect contact with anything (table, tray, object). Lifting reward is the gating signal that enforces actual pick success. |
| G1 elbow sign | Negative elbow value = forward reach toward table. Joint range: -1.05 to +2.09 rad. |
| Default object shape | `MultiAssetSpawnerCfg` with `random_choice=True` (default) → cube/sphere/capsule sampled independently per env each episode. |

Done. Added replicate_physics: bool = False to G1PickSceneCfg in g1_pick_env_cfg.py.

Why this fixes it: InteractiveSceneCfg defaults replicate_physics=True, which tells PhysX to clone env_0's mesh/physics to every other environment — so whatever shape spawn_multi_asset randomly picked for env_0 (the sphere) gets stamped on all other envs. Setting it to False forces each env's physics to be created independently, so the per-env random shape selection from MultiAssetSpawnerCfg actually takes effect.

