#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test script for G1 picking environment."""

import argparse
import torch

def main():
    """Test the G1 picking environment with random actions."""
    import gymnasium as gym
    
    # Import to register the environment
    import isaaclab_tasks.manager_based.manipulation.g1_pick  # noqa: F401
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test G1 picking environment")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to run")
    args = parser.parse_args()
    
    # Create environment
    print(f"Creating environment with {args.num_envs} parallel environments...")
    env = gym.make("Isaac-G1-Pick-v0", num_envs=args.num_envs)
    
    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset environment
    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"Policy observation shape: {obs['policy'].shape}")
    
    # Run random policy
    print(f"\nRunning random policy for {args.num_steps} steps...")
    episode_rewards = torch.zeros(args.num_envs, device=env.unwrapped.device)
    episode_lengths = torch.zeros(args.num_envs, device=env.unwrapped.device)
    
    for step in range(args.num_steps):
        # Sample random action
        action = 2.0 * torch.rand(args.num_envs, env.action_space.shape[0], device=env.unwrapped.device) - 1.0
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Track statistics
        episode_rewards += reward
        episode_lengths += 1
        
        # Reset on done
        done = terminated | truncated
        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            print(f"Step {step}: {len(done_ids)} episodes finished")
            print(f"  Mean reward: {episode_rewards[done_ids].mean().item():.2f}")
            print(f"  Mean length: {episode_lengths[done_ids].mean().item():.1f}")
            episode_rewards[done_ids] = 0
            episode_lengths[done_ids] = 0
        
        # Print progress
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{args.num_steps}")
    
    print("\nTest completed successfully!")
    env.close()


if __name__ == "__main__":
    main()
