import os
import os.path as osp

import cv2
import numpy as np
import tqdm
import json
import pickle
from argparse import ArgumentParser
import trimesh

import yaml
import wandb
from easydict import EasyDict as edict
from lib.utils.proc import proc_numpy

def make_model_result_folder(root, train_type):
    model_folder = osp.join(root, "model")
    result_folder = osp.join(root, "result", train_type)
    
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    return model_folder, result_folder

def make_save_folder(save_root):
    os.makedirs(save_root, exist_ok=True)
    return save_root

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)

def update_config(config):
    parser = ArgumentParser()
    def add_argument(config, parser, parent_key=None):
        for key, value in config.items():
            if parent_key is not None:
                key = f"{parent_key}.{key}"
            if not isinstance(value, dict):
                parser.add_argument(f"--{key}", type=type(value), default=None)
            else:
                add_argument(value, parser, key)
    add_argument(config, parser)
    args, unknown_args = parser.parse_known_args()
    for unknown_arg in unknown_args:
        if "--" in unknown_arg:
            raise Exception(f"Not allowed argument! {unknown_arg}")
    for key, value in vars(args).items():
        if value is not None:
            if "." in key:
                keys = key.split(".")
                d = config
                for key in keys[:-1]:
                    d = d[key]
                d[keys[-1]] = value
            else:
                config[key] = value
    return config

def read_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def read_pkl(pkl_file):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    return data

def wandb_login(
    config, 
    config_model, 
    project_name=None, 
    model_name=None, 
    relogin=True
):
    wandb.login(relogin=relogin)
    if project_name is None:
        project_name = config.project_name
    if model_name is None:
        model_name = config_model.model_name
    wandb.init(
        project=project_name, 
        name=model_name, 
        config=config, 
    )

    return wandb

def save_video(frames, fps, save_path):
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        save_path, 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        fps, (width, height),
    )
    if frames.shape[-1] == 4:
        frames = frames[..., :3] # remove alpha channel

    if frames.max() <= 1:
        frames = (frames*255).astype(np.uint8)
    for frame in tqdm.tqdm(frames, desc="saving video"):
        writer.write(frame)
    writer.release()

def save_frames(frames, save_folder, image_ext="png"):
    os.makedirs(save_folder, exist_ok=True)

    for file_name in os.listdir(save_folder):
        if file_name.lower().endswith(f".{image_ext.lower()}"):
            os.remove(osp.join(save_folder, file_name))

    if frames.shape[-1] == 4:
        frames = frames[..., :3]  # remove alpha channel

    if frames.max() <= 1:
        frames = (frames * 255).astype(np.uint8)

    for frame_idx, frame in enumerate(tqdm.tqdm(frames, desc="saving frames")):
        cv2.imwrite(
            osp.join(save_folder, f"{frame_idx:04d}.{image_ext}"),
            frame,
        )


# Add plot angle comparison function
def save_angle_plot_realtime(gen_angles_list, save_path, gt_angles_list=None, fps=30):
    """Save an articulation angle plot with frame number on the bottom x-axis
    and real time (seconds) on the top x-axis.

    Each sequence is plotted at its own true duration without resampling.

    Parameters
    ----------
    gen_angles_list : list of array-like
        Generated angle sequences (degrees). One entry per sample.
    save_path : str
        Destination .png file path.
    gt_angles_list : list of array-like, optional
        Ground-truth angle sequences (degrees).
    fps : float
        Frames per second used to convert frame index to seconds.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(osp.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))

    # ---- GT ----
    if gt_angles_list is not None and len(gt_angles_list) > 0:
        for g in gt_angles_list:
            g = np.asarray(g, dtype=float)
            frames = np.arange(len(g))
            ax.plot(frames, g, color="lightcoral", linewidth=0.8, alpha=0.4)
        max_len_gt = max(len(np.asarray(g)) for g in gt_angles_list)
        common_frames_gt = np.arange(max_len_gt)
        gt_interp = np.full((len(gt_angles_list), max_len_gt), np.nan)
        for i, g in enumerate(gt_angles_list):
            g = np.asarray(g, dtype=float)
            gt_interp[i, :len(g)] = g
        gt_mean = np.nanmean(gt_interp, axis=0)
        ax.plot(common_frames_gt, gt_mean, color="tomato", linewidth=2.0,
                label=f"GT mean  (n={len(gt_angles_list)})")

    # ---- Generated ----
    if gen_angles_list is not None and len(gen_angles_list) > 0:
        for g in gen_angles_list:
            g = np.asarray(g, dtype=float)
            frames = np.arange(len(g))
            ax.plot(frames, g, color="lightskyblue", linewidth=1.2, alpha=0.7)
        max_len_gen = max(len(np.asarray(g)) for g in gen_angles_list)
        common_frames_gen = np.arange(max_len_gen)
        gen_interp = np.full((len(gen_angles_list), max_len_gen), np.nan)
        for i, g in enumerate(gen_angles_list):
            g = np.asarray(g, dtype=float)
            gen_interp[i, :len(g)] = g
        gen_mean = np.nanmean(gen_interp, axis=0)
        ax.plot(common_frames_gen, gen_mean, color="steelblue", linewidth=2.0,
                label=f"Generated mean  (n={len(gen_angles_list)})")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Articulation angle (deg)")
    ax.set_title("Object articulation angle  (real time)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # ---- Secondary x-axis: seconds ----
    ax_top = ax.twiny()
    ax_top.set_xlim(np.array(ax.get_xlim()) / fps)
    ax_top.set_xlabel("Time (s)")
    # Sync tick positions from bottom axis → convert to seconds
    bottom_ticks = ax.get_xticks()
    top_ticks = bottom_ticks / fps
    ax_top.set_xticks(top_ticks)
    ax_top.set_xticklabels([f"{v:.1f}" for v in top_ticks])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_angle_plot(gen_angles_list, save_path, gt_angles_list=None, norm_points=101):
    """Save a per-frame articulation angle comparison plot as a PNG.

    All sequences are resampled to a common normalised time axis [0, 100 %]
    so traces with different frame counts are directly comparable.

    Parameters
    ----------
    gen_angles_list : list of array-like
        Generated angle sequences (degrees). One entry per sample.
    save_path : str
        Destination .png file path.
    gt_angles_list : list of array-like, optional
        Ground-truth angle sequences (degrees).
    norm_points : int
        Points on the normalised time axis (default 101 → 0-100 %).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(osp.dirname(save_path), exist_ok=True)

    norm_x = np.linspace(0, 100, norm_points)

    def _resample(seq):
        seq = np.asarray(seq, dtype=float)
        src_x = np.linspace(0, 100, len(seq))
        return np.interp(norm_x, src_x, seq)

    fig, ax = plt.subplots(figsize=(10, 4))

    # ---- GT ----
    if gt_angles_list is not None and len(gt_angles_list) > 0:
        gt_resampled = np.stack([_resample(g) for g in gt_angles_list])
        for row in gt_resampled:
            ax.plot(norm_x, row, color="lightcoral", linewidth=0.8, alpha=0.3)
        gt_mean = gt_resampled.mean(0)
        gt_std  = gt_resampled.std(0)
        ax.fill_between(norm_x, gt_mean - gt_std, gt_mean + gt_std,
                        alpha=0.25, color="tomato")
        ax.plot(norm_x, gt_mean, color="tomato", linewidth=2.0,
                label=f"GT mean ± std  (n={len(gt_angles_list)})")

    # ---- Generated ----
    if gen_angles_list is not None and len(gen_angles_list) > 0:
        gen_resampled = np.stack([_resample(g) for g in gen_angles_list])
        for row in gen_resampled:
            ax.plot(norm_x, row, color="lightskyblue", linewidth=0.8, alpha=0.5)
        gen_mean = gen_resampled.mean(0)
        gen_std  = gen_resampled.std(0)
        ax.fill_between(norm_x, gen_mean - gen_std, gen_mean + gen_std,
                        alpha=0.25, color="steelblue")
        ax.plot(norm_x, gen_mean, color="steelblue", linewidth=2.0,
                label=f"Generated mean ± std  (n={len(gen_angles_list)})")

    ax.set_xlabel("Normalised time (%)")
    ax.set_ylabel("Articulation angle (deg)")
    ax.set_title("Object articulation angle  (normalised time)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_mesh_obj(
    vertices, faces, save_folder,
):
    os.makedirs(save_folder, exist_ok=True)
    faces = proc_numpy(faces)
    for obj_idx, verts in enumerate(vertices):
        verts = proc_numpy(verts)
        trimesh.Trimesh(verts, faces).export(osp.join(save_folder, f"{obj_idx:03d}.obj"))