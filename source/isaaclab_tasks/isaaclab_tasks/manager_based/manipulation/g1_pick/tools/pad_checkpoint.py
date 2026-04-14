"""
Pad a 66-dim obs checkpoint to work with a 75-dim obs model.

The 9 new dims (3 distractor positions) are inserted at indices 35-43
(after object_vel, before right_fingertip_pos).

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

OLD_DIM = 66
NEW_DIM = 75
INSERT_AT = 35  # after object_vel (indices 29-34), before fingertip_pos
NUM_NEW = 9

# Keys whose dim=1 needs padding from 66->75
MODEL_KEYS_TO_PAD = {
    "actor.0.weight", "critic.0.weight",
    "actor_obs_normalizer._mean", "actor_obs_normalizer._var", "actor_obs_normalizer._std",
    "critic_obs_normalizer._mean", "critic_obs_normalizer._var", "critic_obs_normalizer._std",
}


def pad_tensor_dim1(tensor, insert_at, num_new, fill_val=0.0):
    """Insert num_new columns of fill_val at position insert_at along dim=1."""
    left = tensor[:, :insert_at]
    right = tensor[:, insert_at:]
    pad = torch.full((tensor.shape[0], num_new), fill_val, dtype=tensor.dtype)
    return torch.cat([left, pad, right], dim=1)


def pad_checkpoint(input_path, output_path):
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    # --- 1. Pad model_state_dict ---
    state_dict = checkpoint["model_state_dict"]
    padded = {}
    for key, tensor in state_dict.items():
        if key in MODEL_KEYS_TO_PAD and tensor.dim() == 2 and tensor.shape[1] == OLD_DIM:
            if "_var" in key or "_std" in key:
                fill = 1.0  # neutral: var=1, std=1
            else:
                fill = 0.0  # neutral: weight=0, mean=0
            padded[key] = pad_tensor_dim1(tensor, INSERT_AT, NUM_NEW, fill)
            print(f"  model | {key}: {tensor.shape} -> {padded[key].shape}")
        else:
            padded[key] = tensor
    checkpoint["model_state_dict"] = padded

    # --- 2. Pad optimizer_state_dict (Adam exp_avg and exp_avg_sq) ---
    if "optimizer_state_dict" in checkpoint:
        opt = checkpoint["optimizer_state_dict"]
        for param_idx, param_state in opt["state"].items():
            for buf_name in ("exp_avg", "exp_avg_sq"):
                if buf_name in param_state:
                    t = param_state[buf_name]
                    if t.dim() == 2 and t.shape[1] == OLD_DIM:
                        fill = 0.0  # zero momentum for new dims = fresh start
                        param_state[buf_name] = pad_tensor_dim1(t, INSERT_AT, NUM_NEW, fill)
                        print(f"  optim | param {param_idx} {buf_name}: {t.shape} -> {param_state[buf_name].shape}")
                    elif t.dim() == 1 and t.shape[0] == OLD_DIM:
                        # 1D case (bias-shaped but matching old dim — unlikely but safe)
                        left = t[:INSERT_AT]
                        right = t[INSERT_AT:]
                        pad = torch.zeros(NUM_NEW, dtype=t.dtype)
                        param_state[buf_name] = torch.cat([left, pad, right])
                        print(f"  optim | param {param_idx} {buf_name}: {t.shape} -> {param_state[buf_name].shape}")

        # Also pad param_groups params if they store shapes (usually just indices, but check)
        checkpoint["optimizer_state_dict"] = opt
        print("  optimizer state padded")
    else:
        print("  no optimizer_state_dict found (fine for inference, will init fresh for training)")

    # --- 3. Drop the optimizer state entirely as a safer alternative ---
    # Uncomment below if optimizer padding still causes issues:
    # checkpoint.pop("optimizer_state_dict", None)
    # print("  dropped optimizer state — Adam will reinitialize from scratch")

    torch.save(checkpoint, output_path)
    print(f"\nSaved padded checkpoint to: {output_path}")


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/home/daatsi-aeres/IsaacLab/logs/rsl_rl/g1_pick/test/model_4999_perfect_pick_v3.pt"
    output_path = input_path.replace(".pt", "_padded_75dim.pt")
    print(f"Padding checkpoint: {input_path}")
    print(f"Old dim: {OLD_DIM} -> New dim: {NEW_DIM} (inserting {NUM_NEW} dims at index {INSERT_AT})\n")
    pad_checkpoint(input_path, output_path)
