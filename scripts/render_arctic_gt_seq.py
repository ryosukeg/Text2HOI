"""Render specific ARCTIC GT sequences from data.npz to mp4.

Unlike `demo.py`, which only renders the median-length matching sequence per
(prompt), this script renders whichever indices you list.

Usage (from Text2HOI repo root, text2hoi env):
  python scripts/render_arctic_gt_seq.py --seq-idxs 40 528
  python scripts/render_arctic_gt_seq.py --seq-idxs 40 528 --out-dir demo_output/arctic_gt/videos --save-frames

Outputs:
  <out_dir>/gt_seq_<seq_idx>_<object>_<action>.mp4
  (optional) <out_dir>/frames_<seq_idx>/*.png  when --save-frames
"""

import argparse
import os
import os.path as osp
import sys

sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))

import numpy as np
import torch
import trimesh

from lib.models.mano import build_mano_aa
from lib.utils.demo_utils import proc_results
from lib.utils.file import read_json, save_frames, save_video
from lib.utils.proc import proc_long_torch_cuda, proc_torch_cuda
from lib.utils.renderer import Renderer
from lib.utils.visualize import render_videos


def _load_object_bundle(obj_root: str, object_name: str, cache: dict) -> tuple:
    """Return cached (obj_verts_cuda_scaled, obj_faces_np, obj_top_idx_cuda)."""
    if object_name in cache:
        return cache[object_name]
    mesh_path = osp.join(obj_root, object_name, "mesh.obj")
    if not osp.exists(mesh_path):
        raise FileNotFoundError(mesh_path)
    mesh = trimesh.load(mesh_path, maintain_order=True, process=False)
    obj_verts = proc_torch_cuda(mesh.vertices.copy()) / 1000.0  # arctic mm → m
    obj_faces = np.asarray(mesh.faces, dtype=np.int64)
    parts_path = osp.join(obj_root, object_name, "parts.json")
    parts = read_json(parts_path)
    obj_top_idx = (1 - np.asarray(parts))
    obj_top_idx_t = proc_long_torch_cuda(obj_top_idx.tolist())
    cache[object_name] = (obj_verts, obj_faces, obj_top_idx_t)
    return cache[object_name]


def _render_one(
    seq_idx: int,
    data: np.lib.npyio.NpzFile,
    cache: dict,
    lhand_layer,
    rhand_layer,
    renderer: Renderer,
    obj_root: str,
    out_dir: str,
    fps: int,
    save_frames_too: bool,
) -> None:
    T = int(data["nframes"][seq_idx])
    obj_name = str(data["object_name"][seq_idx])
    action = str(data["action_name"][seq_idx])
    is_l = bool(int(data["is_lhand"][seq_idx]))
    is_r = bool(int(data["is_rhand"][seq_idx]))
    print(
        f"[seq_idx={seq_idx:4d}] object={obj_name:12s} action={action:8s} "
        f"is_l={int(is_l)} is_r={int(is_r)} T={T}"
    )

    obj_verts, obj_faces, obj_top_idx_t = _load_object_bundle(obj_root, obj_name, cache)

    x_lhand = proc_torch_cuda(data["x_lhand"][seq_idx, :T])
    x_rhand = proc_torch_cuda(data["x_rhand"][seq_idx, :T])
    x_obj = proc_torch_cuda(
        np.concatenate(
            [data["x_obj"][seq_idx, :T], data["x_obj_angle"][seq_idx, :T]],
            axis=-1,
        )
    )

    obj_verts_tf, lhand_verts, lhand_faces, rhand_verts, rhand_faces = proc_results(
        x_lhand, x_rhand, x_obj,
        obj_verts, lhand_layer, rhand_layer,
        is_l, is_r,
        "arctic", obj_top_idx_t,
    )

    # Same centering as demo.py: subtract frame-0 XY centroid of the object.
    center_xy = obj_verts_tf[0, :, :2].mean(0)[None, None]
    obj_verts_tf[:, :, :2] = obj_verts_tf[:, :, :2] - center_xy
    if is_l and lhand_verts is not None:
        lhand_verts[:, :, :2] = lhand_verts[:, :, :2] - center_xy
    if is_r and rhand_verts is not None:
        rhand_verts[:, :, :2] = rhand_verts[:, :, :2] - center_xy

    video = render_videos(
        renderer, lhand_verts, lhand_faces,
        rhand_verts, rhand_faces,
        obj_verts_tf, obj_faces,
        is_l, is_r,
    )

    os.makedirs(out_dir, exist_ok=True)
    out_name = f"gt_seq_{seq_idx:04d}_{obj_name}_{action}.mp4"
    out_path = osp.join(out_dir, out_name)
    save_video(video, fps=fps, save_path=out_path)
    print(f"  → {out_path}")
    if save_frames_too:
        frames_dir = osp.join(out_dir, f"frames_seq_{seq_idx:04d}_{obj_name}_{action}")
        save_frames(video, save_folder=frames_dir)
        print(f"  → {frames_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-path", default="data/arctic/data.npz")
    ap.add_argument(
        "--obj-root",
        default="data/arctic/downloads/data/meta/object_vtemplates",
        help="ARCTIC object mesh root (contains <object>/mesh.obj).",
    )
    ap.add_argument(
        "--seq-idxs", type=int, nargs="+", required=True,
        help="One or more sequence indices in data.npz to render.",
    )
    ap.add_argument("--out-dir", default="demo_output/arctic_gt/videos")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument(
        "--save-frames", action="store_true",
        help="Also save per-frame PNGs beside the mp4.",
    )
    args = ap.parse_args()

    print(f"Loading {args.data_path} ...")
    data = np.load(args.data_path, allow_pickle=True)
    print("Building MANO layers (flat_hand=False, arctic convention) ...")
    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=False).cuda()
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=False).cuda()
    print("Building renderer (arctic_front camera) ...")
    renderer = Renderer(device="cuda", camera="arctic_front")

    cache: dict = {}
    for i in args.seq_idxs:
        _render_one(
            int(i), data, cache,
            lhand_layer, rhand_layer, renderer,
            args.obj_root, args.out_dir, args.fps, args.save_frames,
        )


if __name__ == "__main__":
    main()
