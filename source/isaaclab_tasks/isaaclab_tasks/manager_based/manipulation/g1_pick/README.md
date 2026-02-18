# G1 Object Picking Environment

A curriculum-based reinforcement learning environment where a G1 humanoid robot learns to pick target objects from a cluttered tray using its dexterous hands.

## Overview

This environment implements a progressive curriculum learning approach for training a G1 robot to:
1. Touch target objects
2. Grasp target objects
3. Lift target objects
4. Pick target objects from increasing clutter (1-2, 3-5, 5+ non-target objects)

The robot stands with a frozen lower body (legs locked) and uses both arms and hands to manipulate objects on a table.

## Features

- **Curriculum Learning**: 6 progressive stages from simple touch to complex picking with heavy clutter
- **Frozen Lower Body**: Legs, pelvis, and torso are frozen to focus learning on manipulation
- **Dual-Arm Capability**: Both arms are active and can be used for picking or clearing clutter
- **Privileged Information**: Target object identification provided for teacher model training
- **Multi-Shape Objects**: Cuboids, spheres, and capsules with randomized properties
- **Adaptive Difficulty**: Automatic progression based on success rate

## Environment Registration

- **Training**: `Isaac-G1-Pick-v0` (4096 parallel environments)
- **Evaluation**: `Isaac-G1-Pick-Play-v0` (64 environments, max difficulty)

## Usage

### Testing the Environment

```bash
# IMPORTANT: First activate the Isaac Lab conda environment
conda activate env_isaaclab
cd /home/daatsi-aeres/IsaacLab

# EASIEST METHOD: Use the built-in random agent script
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-G1-Pick-v0 --num_envs 1

# This will:
# - Create the environment with proper Isaac Sim initialization
# - Run a random policy to test the environment
# - Show the robot and scene in Isaac Sim viewer
# - Print episode statistics

# Alternative: Run custom test script
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick/test_env.py --num_envs 4 --num_steps 100
```

### Training

```bash
# IMPORTANT: Activate conda environment first
conda activate env_isaaclab

# Train with RL Games PPO (full scale)
cd /home/daatsi-aeres/IsaacLab
./isaaclab.sh -p scripts/rl_games/train.py --task Isaac-G1-Pick-v0 --num_envs 4096 --headless

# Train with fewer environments for debugging
./isaaclab.sh -p scripts/rl_games/train.py --task Isaac-G1-Pick-v0 --num_envs 512
```

### Evaluation

```bash
# Activate conda environment
conda activate env_isaaclab

# Run trained policy
cd /home/daatsi-aeres/IsaacLab
./isaaclab.sh -p scripts/rl_games/play.py --task Isaac-G1-Pick-Play-v0 --num_envs 64 --checkpoint /path/to/checkpoint.pth
```

## Curriculum Stages

| Stage | Difficulty | Task | Clutter Objects |
|-------|-----------|------|-----------------|
| 1 | 0-10 | Touch target object | 0 |
| 2 | 10-20 | Grasp target object | 0 |
| 3 | 20-30 | Lift target object | 0 |
| 4 | 30-40 | Pick with light clutter | 1-2 |
| 5 | 40-50 | Pick with medium clutter | 3-5 |
| 6 | 50+ | Pick with heavy clutter | 5-7 |

Difficulty automatically increases when the robot successfully lifts the target object above the threshold height (0.55m).

## File Structure

```
g1_pick/
├── __init__.py                 # Environment registration
├── g1_pick_env_cfg.py         # Main environment configuration
├── test_env.py                # Test script
├── agents/
│   ├── __init__.py
│   └── rl_games_ppo_cfg.yaml  # PPO training configuration
└── mdp/
    ├── __init__.py
    ├── observations.py        # Custom observation functions
    ├── rewards.py             # Custom reward functions
    ├── terminations.py        # Custom termination functions
    ├── events.py              # Custom event functions
    └── curriculum.py          # Curriculum scheduler
```

## Notes

- Lower body joints are frozen by setting very high stiffness (10000.0) and damping (1000.0)
- Episode length is 10 seconds (1200 steps at 120Hz simulation, 2x decimation = 600 control steps)
- The environment uses privileged information (target object position) suitable for teacher-student training


...

./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-G1-Pick-v0 \
  --headless \
  --num_envs 1024 \
  --video --video_length 600 --video_interval 2000