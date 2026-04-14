"""
Pad a 66-dim obs checkpoint to work with a 75-dim obs model.

The 9 new dims (3 distractor positions) are inserted at indices 22-30
(after object_vel at index 21, before right_fingertip_pos).

Old layout (66 dims):
  [0:13]  joint_pos
  [13:26] joint_vel
  [26:29] object_pos_b
  [29:35] object_vel
  [35:53] right_fingertip_pos   <-- these shift by +9
  [53:66] actions               <-- these shift by +9

New layout (75 dims):
  [0:13]  joint_pos
  [13:26] joint_vel
  [26:29] object_pos_b
  [29:35] object_vel
  [35:38] distractor_1_pos_b    <-- NEW
  [38:41] distractor_2_pos_b    <-- NEW
  [41:44] distractor_3_pos_b    <-- NEW
  [44:62] right_fingertip_pos
  [62:75] actions
"""

import torch
import sys
import os

OLD_DIM = 66
NEW_DIM = 75
INSERT_AT = 35  # after object_vel (indices 29-34), before fingertip_pos
NUM_NEW = 9


def pad_checkpoint(input_path, output_path):
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    padded = {}
    for key, tensor in state_dict.items():
        if key in ("actor.0.weight", "critic.0.weight"):
            # Shape: [out_features, 66] -> [out_features, 75]
            assert tensor.shape[1] == OLD_DIM, f"{key} has shape {tensor.shape}, expected [:,{OLD_DIM}]"
            left = tensor[:, :INSERT_AT]
            right = tensor[:, INSERT_AT:]
            zeros = torch.zeros(tensor.shape[0], NUM_NEW, dtype=tensor.dtype)
            padded[key] = torch.cat([left, zeros, right], dim=1)
            print(f"  {key}: {tensor.shape} -> {padded[key].shape}")

        elif key in (
            "actor_obs_normalizer._mean", "critic_obs_normalizer._mean",
            "actor_obs_normalizer._var", "critic_obs_normalizer._var",
        ):
            # Shape: [1, 66] -> [1, 75]
            assert tensor.shape[1] == OLD_DIM, f"{key} has shape {tensor.shape}"
            left = tensor[:, :INSERT_AT]
            right = tensor[:, INSERT_AT:]
            fill_val = 0.0 if "_mean" in key else 1.0  # var=1 so std=1 (neutral)
            pad = torch.full((1, NUM_NEW), fill_val, dtype=tensor.dtype)
            padded[key] = torch.cat([left, pad, right], dim=1)
            print(f"  {key}: {tensor.shape} -> {padded[key].shape}")

        elif key in (
            "actor_obs_normalizer._std",
            "critic_obs_normalizer._std",
        ):
            assert tensor.shape[1] == OLD_DIM, f"{key} has shape {tensor.shape}"
            left = tensor[:, :INSERT_AT]
            right = tensor[:, INSERT_AT:]
            pad = torch.ones((1, NUM_NEW), dtype=tensor.dtype)  # std=1 (neutral)
            padded[key] = torch.cat([left, pad, right], dim=1)
            print(f"  {key}: {tensor.shape} -> {padded[key].shape}")

        elif key in (
            "actor_obs_normalizer._count",
            "critic_obs_normalizer._count",
        ):
            # Scalar, no change
            padded[key] = tensor
        else:
            # All other layers (hidden layers, biases, etc.) unchanged
            padded[key] = tensor

    checkpoint["model_state_dict"] = padded
    torch.save(checkpoint, output_path)
    print(f"\nSaved padded checkpoint to: {output_path}")


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/home/daatsi-aeres/IsaacLab/logs/rsl_rl/g1_pick/test/model_4999_perfect_pick_v3.pt"
    output_path = input_path.replace(".pt", "_padded_75dim.pt")
    print(f"Padding checkpoint: {input_path}")
    print(f"Old dim: {OLD_DIM} -> New dim: {NEW_DIM} (inserting {NUM_NEW} dims at index {INSERT_AT})\n")
    pad_checkpoint(input_path, output_path)
