"""Export ARCTIC GT motion sequences in the same npz format as demo.py.

Reads `data/arctic/data.npz`, filters by (object, action, hand-side), converts
`rot6d → axis-angle`, slices to per-sequence `nframes`, and writes .npz files
that are directly consumable by `manosim.eval_text2hoi` and
`manosim.replay_text2hoi`.

Usage (from Text2HOI repo root, in the text2hoi env):
  python scripts/export_arctic_gt.py --object box --action open --nsamples 3
  python scripts/export_arctic_gt.py --object scissors --action cut --hand-side right --nsamples 3

Output default: demo_output/arctic_gt/mano_params/gt_{action}_{object}_{side}_i{seq_idx}.npz
"""

import argparse
import os
import os.path as osp
import sys

sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))

import numpy as np
import torch

from lib.utils.rot import rot6d_to_axis_angle


def _rot6d_seq_to_axis_angle(rot6d_flat: np.ndarray) -> np.ndarray:
    """(T, 16*6) rot6d → (T, 48) axis-angle (wrist + 15 fingers)."""
    t = torch.from_numpy(rot6d_flat).float()
    aa = rot6d_to_axis_angle(t).reshape(-1, 48)
    return aa.numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-path", default="data/arctic/data.npz")
    ap.add_argument("--object", required=True, help="e.g. box, scissors, ketchup")
    ap.add_argument("--action", required=True, help="e.g. open, close, cut, grasp, lift, place")
    ap.add_argument(
        "--hand-side", choices=("both", "right", "left"), default="both",
        help="both=(is_l==1 AND is_r==1); right=(r only); left=(l only). Default: both",
    )
    ap.add_argument("--nsamples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None, help="Random seed for sampling.")
    ap.add_argument(
        "--random", action="store_true",
        help="Randomly sample nsamples instead of taking the first ones.",
    )
    ap.add_argument("--out-dir", default="demo_output/arctic_gt/mano_params")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    raw = np.load(args.data_path, allow_pickle=True)
    object_name = raw["object_name"]
    action_name = raw["action_name"]
    is_l = raw["is_lhand"]
    is_r = raw["is_rhand"]
    nframes = raw["nframes"]
    x_lhand = raw["x_lhand"]        # (N, T_max, 99)
    x_rhand = raw["x_rhand"]        # (N, T_max, 99)
    x_obj = raw["x_obj"]            # (N, T_max, 9)   trans(3)+rot6d(6)
    x_obj_angle = raw["x_obj_angle"]  # (N, T_max, 1)

    mask = (object_name == args.object) & (action_name == args.action)
    if args.hand_side == "right":
        mask &= (is_r == 1) & (is_l == 0)
    elif args.hand_side == "left":
        mask &= (is_l == 1) & (is_r == 0)
    else:  # both
        mask &= (is_l == 1) & (is_r == 1)
    idxs = np.where(mask)[0]
    print(
        f"Found {len(idxs)} matching GT sequences for "
        f"object={args.object}, action={args.action}, hand_side={args.hand_side}"
    )
    if len(idxs) == 0:
        return
    if args.random:
        rng = np.random.default_rng(args.seed)
        idxs = rng.permutation(idxs)
    n_out = min(args.nsamples, len(idxs))

    os.makedirs(args.out_dir, exist_ok=True)
    for k, i in enumerate(idxs[:n_out]):
        T = int(nframes[i])
        lhand = x_lhand[i, :T]         # (T, 99)
        rhand = x_rhand[i, :T]         # (T, 99)
        obj9 = x_obj[i, :T]            # (T, 9)
        angle = x_obj_angle[i, :T]     # (T, 1)
        obj_params = np.concatenate([obj9, angle], axis=-1).astype(np.float32)  # (T, 10)

        lhand_trans = lhand[:, :3].astype(np.float32)
        lhand_pose = _rot6d_seq_to_axis_angle(lhand[:, 3:])  # (T, 48)
        rhand_trans = rhand[:, :3].astype(np.float32)
        rhand_pose = _rot6d_seq_to_axis_angle(rhand[:, 3:])

        text = f"[GT] {args.action} {args.object} (seq={int(i)}, side={args.hand_side})"
        out_name = f"gt_{args.action}_{args.object}_{args.hand_side}_i{int(i):04d}.npz"
        out_path = osp.join(args.out_dir, out_name)
        np.savez(
            out_path,
            lhand_trans=lhand_trans,
            lhand_pose=lhand_pose,
            rhand_trans=rhand_trans,
            rhand_pose=rhand_pose,
            obj_params=obj_params,
            is_lhand=np.array(bool(is_l[i])),
            is_rhand=np.array(bool(is_r[i])),
            dataset=np.array("arctic"),
            flat_hand=np.array(False),
            fps=np.array(args.fps),
            text=np.array(text),
        )
        print(f"  [{k + 1}/{n_out}] T={T:3d}  seq_idx={int(i):4d}  → {out_path}")


if __name__ == "__main__":
    main()
