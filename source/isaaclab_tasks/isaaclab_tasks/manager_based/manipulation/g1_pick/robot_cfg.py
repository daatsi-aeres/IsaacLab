# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 29-DOF humanoid with Inspire FTP dexterous hands.

Robot highlights:
- 12 leg DOF (hip yaw/roll/pitch, knee, ankle pitch/roll × 2)
- 3 waist DOF (yaw, roll, pitch)
- 14 arm DOF (shoulder 3DOF + elbow 1DOF + wrist 3DOF × 2)
- 24 hand DOF (Inspire FTP: thumb ×4, index/middle/ring/little ×2 each × 2 hands)

URDF issues noted:
- Mimic joints (thumb_3/4 mimic thumb_2/3; finger _2 joints mimic _1): PhysX does NOT
  support <mimic> tags; all joints are treated as independent DOFs in simulation.
  In real deployment, the hardware's mechanical coupling enforces the mimic constraint.
- Mesh files must exist at
  /home/daatsi-aeres/ARCLab_ws/unitree_ros/robots/g1_description/meshes/ (160 STL files).
  The URDF is loaded from that directory so relative mesh paths resolve correctly.
- Four sensor-only links (imu_in_torso, imu_in_pelvis, d435_link, mid360_link) have no
  inertial definition — this is intentional (massless sensor frames).
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg

import os
# URDF must live next to its meshes/ folder so relative paths resolve
_URDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "g1_description",
    "g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"
)

G1_INSPIRE_CFG = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path=_URDF_PATH,
        activate_contact_sensors=True,
        fix_base=True,
        merge_fixed_joints=True,   # merges force-sensor fixed links → fewer bodies, faster sim
        self_collision=False,
        make_instanceable=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        # joint_drive=None: use URDF-parsed PD values for USD conversion.
        # IsaacLab ImplicitActuatorCfg overrides these at runtime anyway.
        joint_drive=None,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Pelvis fixed at x=-0.1 (fix_base=True).
        # Table front edge at x=0.1 (table centre 0.4, half-depth 0.3) → 0.2 m clearance.
        # Tray centre at x=0.4 → arms need ~0.5 m forward reach, achieved with high
        # shoulder_pitch + bent elbow so hands hover just above the tray surface.
        pos=(-0.1, 0.0, 0.74),
        joint_pos={
            # ── Legs: standard crouched-standing pose (cosmetic, base is fixed) ─
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            # ── Waist: neutral ─────────────────────────────────────────────────
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            # ── Arms: hands pre-positioned over tray ──────────────────────────
            # G1 elbow convention: NEGATIVE = forward (toward table), POSITIVE = backward.
            # Joint range: -1.05 to +2.09 rad.
            # shoulder_pitch =  1.0 rad → arms swing ~57° forward
            # shoulder_roll  = ±0.1 rad → slight inward to center over tray
            # elbow          = -0.8 rad → forward reach toward table; 0.25 rad from limit
            # wrist_pitch    = -0.8 rad → palms face down toward objects
            ".*_shoulder_pitch_joint": -0.5,
            "left_shoulder_roll_joint": -0.1,
            "right_shoulder_roll_joint": 0.1,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.3,
            ".*_wrist_roll_joint": 0.0,
            ".*_wrist_pitch_joint": -0.8,
            ".*_wrist_yaw_joint": 0.0,
            # ── Inspire hands: fully open ──────────────────────────────────────
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # ── Lower body (frozen during manipulation) ────────────────────────────
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=300,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "waist_.*_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "waist_.*_joint": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=20,
            stiffness=20.0,
            damping=2.0,
        ),
        # ── Arms (controlled) ─────────────────────────────────────────────────
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=25,
            stiffness=40.0,
            damping=10.0,
        ),
        # ── Inspire hands (controlled) ────────────────────────────────────────
        # All 12 joints per hand are in the actuator group.
        # The policy controls only the 6 leader joints per hand (thumb_1/2, index/middle/ring/little_1).
        # The 6 mimic joints (thumb_3/4, *_2 fingers) are excluded from the action space but
        # are still actuated; they are reset to 0 each episode and held there by stiffness.
        "left_hand": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_thumb_1_joint", "left_thumb_2_joint",
                "left_thumb_3_joint", "left_thumb_4_joint",
                "left_index_1_joint", "left_index_2_joint",
                "left_middle_1_joint", "left_middle_2_joint",
                "left_ring_1_joint", "left_ring_2_joint",
                "left_little_1_joint", "left_little_2_joint",
            ],
            effort_limit_sim=10,
            stiffness=20.0,
            damping=2.0,
        ),
        "right_hand": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_thumb_1_joint", "right_thumb_2_joint",
                "right_thumb_3_joint", "right_thumb_4_joint",
                "right_index_1_joint", "right_index_2_joint",
                "right_middle_1_joint", "right_middle_2_joint",
                "right_ring_1_joint", "right_ring_2_joint",
                "right_little_1_joint", "right_little_2_joint",
            ],
            effort_limit_sim=10,
            stiffness=20.0,
            damping=2.0,
        ),
    },
)
"""G1 29-DOF humanoid with Inspire FTP dexterous hands (loaded from URDF)."""
