#!/usr/bin/env python
"""
compare_angle_conditions.py
目標角度ごとの角度軌跡を1枚のグラフ（正規化時間 + 実時間）で比較し、
各条件ごとにモーション動画（最大 n_save 本）も保存する。

Usage:
    python scripts/compare_angle_conditions.py \
        dataset=arctic \
        texthom.obj_nfeats=10 \
        texthom.model_name=texthom_angle_box_v2 \
        texthom.use_angle_cond=True \
        texthom.angle_min_deg=0 \
        texthom.angle_max_deg=130 \
        'test_text=["Open box with both hands."]' \
        nsamples=5 \
        '+target_angles=[0,20,40,60,80,100]' \
        'hydra.output_subdir=null' \
        'hydra/job_logging=disabled' \
        'hydra/hydra_logging=disabled'

出力先: demo_output/compare_angle_conditions/<model_name>/
    overlay_normalized.png   正規化時間軸グラフ
    overlay_realtime.png     実時間（フレーム）軸グラフ
    target_Xdeg/motion_sN.mp4   各角度のモーション動画
    no_angle/motion_sN.mp4      角度指定なしのモーション動画
"""
import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import hydra
from omegaconf import OmegaConf
from easydict import EasyDict as edict

from lib.models.mano import build_mano_aa
from lib.utils.demo_utils import get_object_hand_info, get_valid_mask_bunch, proc_results
from lib.utils.model_utils import (
    build_refiner, build_model_and_diffusion,
    build_seq_cvae, build_mpnet, build_pointnetfeat, build_contact_estimator,
)
from lib.models.object import build_object_model
from lib.networks.clip import load_and_freeze_clip, encoded_text
from lib.utils.proc import (
    proc_obj_feat_final, proc_cond_contact_estimator,
    proc_refiner_input, proc_numpy,
)
from lib.utils.renderer import Renderer
from lib.utils.visualize import render_videos
from lib.utils.file import save_video


# ─────────────────────────────────────────────────────────────────
# ヘルパー
# ─────────────────────────────────────────────────────────────────

def _resample(traj, n=100):
    src = np.linspace(0, 1, len(traj))
    dst = np.linspace(0, 1, n)
    return np.interp(dst, src, traj)


def _label_to_folder(label):
    """ラベル文字列をフォルダ名に変換"""
    return label.replace("=", "").replace("°", "deg").replace(" ", "_")


def _plot_overlay(collected, conditions, palette, fps, max_nframes, save_dir, model_name):
    """正規化時間軸と実時間軸の2枚グラフを保存する"""
    NORM = 100

    for mode in ("normalized", "realtime"):
        fig, ax = plt.subplots(figsize=(12, 5))

        for label, target_deg in conditions:
            trajs = collected[label]
            if not trajs:
                continue
            color = palette.get(label, "black")
            max_len = max(len(tr) for tr in trajs)

            if mode == "normalized":
                resampled = np.stack([_resample(tr, NORM) for tr in trajs])
                xs_mean   = np.linspace(0, 100, NORM)
                xlabel    = "Normalised time (%)"
            else:
                resampled = np.stack([
                    np.interp(np.arange(max_len),
                              np.linspace(0, max_len - 1, len(tr)), tr)
                    for tr in trajs
                ])
                xs_mean = np.arange(max_len)
                xlabel  = f"Frame  (fps={fps})"

            for row in resampled:
                ax.plot(xs_mean, row, color=color, alpha=0.18, linewidth=0.8)

            mean_t = resampled.mean(0)
            std_t  = resampled.std(0)
            ax.fill_between(xs_mean, mean_t - std_t, mean_t + std_t,
                            alpha=0.18, color=color)
            ls = "--" if label == "角度指定なし" else "-"
            ax.plot(xs_mean, mean_t, color=color, linewidth=2.5, linestyle=ls,
                    label=f"{label}  (final={mean_t[-1]:.1f}°, n={len(trajs)})")

            if target_deg is not None:
                ax.axhline(target_deg, color=color, linestyle=":", linewidth=0.9, alpha=0.6)

        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("Articulation angle (deg)", fontsize=13)
        ax.set_title(f"Angle condition comparison  [{model_name}]", fontsize=13)
        ax.legend(fontsize=10, loc="upper left", ncol=2)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()

        path = osp.join(save_dir, f"overlay_{mode}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [saved] {path}")


# ─────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../configs", config_name="config")
@torch.no_grad()
def main(config):
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_object(config)
    config = edict(config)

    data_config   = config.dataset
    dataset_name  = data_config.name
    max_nframes   = data_config.max_nframes
    hand_nfeats   = config.texthom.hand_nfeats
    obj_nfeats    = config.texthom.obj_nfeats
    nsamples      = config.nsamples
    fps           = config.fps
    text          = list(config.test_text)

    use_angle_cond = bool(config.texthom.use_angle_cond)
    angle_min_deg  = float(config.texthom.angle_min_deg)
    angle_max_deg  = float(config.texthom.angle_max_deg)

    if "target_angles" in config:
        target_angles = [float(a) for a in config.target_angles]
    else:
        target_angles = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]

    n_save = int(getattr(config, "n_save", 3))

    print(f"[Compare] model={config.texthom.model_name}  use_angle_cond={use_angle_cond}")
    print(f"[Compare] target_angles={target_angles} + no-angle  nsamples={nsamples}  n_save={n_save}")

    root_dir = osp.join(
        osp.dirname(osp.abspath(osp.dirname(__file__))),
        "demo_output", "compare_angle_conditions",
        config.texthom.model_name,
    )
    os.makedirs(root_dir, exist_ok=True)

    # ── モデル読み込み ──
    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=data_config.flat_hand).cuda()
    rhand_layer = build_mano_aa(is_rhand=True,  flat_hand=data_config.flat_hand).cuda()
    refiner     = build_refiner(config, test=True)
    texthom, diffusion = build_model_and_diffusion(config, lhand_layer, rhand_layer, test=True)
    clip_model  = load_and_freeze_clip(config.clip.clip_version).cuda()
    mpnet       = build_mpnet(config)
    seq_cvae    = build_seq_cvae(config, test=True)
    pointnet    = build_pointnetfeat(config, test=True)
    contact_estimator = build_contact_estimator(config, test=True)
    object_model = build_object_model(data_config.data_obj_pc_path)
    renderer    = Renderer(device="cuda", camera=f"{dataset_name}_front")

    # ── 共通特徴量 ──
    is_lhand, is_rhand, \
    obj_pc_org, obj_pc_normal_org, \
    normalized_obj_pc, point_sets, \
    obj_cent, obj_scale, \
    obj_verts, obj_faces, \
    obj_top_idx, obj_pc_top_idx = get_object_hand_info(
        object_model, clip_model, text,
        data_config.obj_root, data_config, mpnet,
    )
    npts      = normalized_obj_pc.shape[1]
    enc_text  = encoded_text(clip_model, text)
    obj_feat  = pointnet(normalized_obj_pc)
    batch_num = len(text) // 64 + 1

    # ── 条件リスト ──
    conditions = [(f"target={t:.0f}°", float(t)) for t in target_angles]
    conditions.append(("角度指定なし", None))

    cmap    = plt.get_cmap("plasma", len(target_angles))
    palette = {f"target={t:.0f}°": cmap(i) for i, t in enumerate(target_angles)}
    palette["角度指定なし"] = "#7f8c8d"

    collected = {label: [] for label, _ in conditions}

    # ── 条件ごとに推論 ──
    for label, target_deg in conditions:
        if target_deg is not None and use_angle_cond:
            denom  = max(angle_max_deg - angle_min_deg, 1e-6)
            scalar = float(np.clip((target_deg - angle_min_deg) / denom, 0.0, 1.0))
            print(f"\n── {label}  (cond_scalar={scalar:.4f}) ──")
        else:
            scalar = None
            print(f"\n── {label}  (angle_cond=None) ──")

        vid_folder = osp.join(root_dir, _label_to_folder(label))
        os.makedirs(vid_folder, exist_ok=True)
        save_count = 0

        for sample_idx in range(nsamples):
            for batch_idx in range(batch_num):
                sl = slice(batch_idx * 64, (batch_idx + 1) * 64)
                enc_text_b   = enc_text[sl]
                if enc_text_b.shape[0] == 0:
                    continue
                is_lhand_b   = is_lhand[sl]
                is_rhand_b   = is_rhand[sl]
                obj_cent_b   = obj_cent[sl]
                obj_scale_b  = obj_scale[sl]
                obj_feat_b   = obj_feat[sl]
                obj_pc_org_b = obj_pc_org[sl]
                obj_pc_nor_b = obj_pc_normal_org[sl]
                norm_pc_b    = normalized_obj_pc[sl]
                obj_verts_b  = obj_verts[sl]
                obj_faces_b  = obj_faces[sl]
                ot_idx_b     = obj_top_idx[sl]    if dataset_name == "arctic" else None
                op_idx_b     = obj_pc_top_idx[sl] if dataset_name == "arctic" else None

                duration = seq_cvae.decode(enc_text_b)
                duration *= 150
                duration = duration.long()
                duration = torch.full_like(duration, max_nframes)

                valid_mask_lhand, valid_mask_rhand, valid_mask_obj = \
                    get_valid_mask_bunch(is_lhand_b, is_rhand_b, max_nframes, duration)

                obj_feat_final, est_contact_map = proc_obj_feat_final(
                    contact_estimator,
                    obj_scale_b, obj_cent_b, obj_feat_b, enc_text_b, npts,
                    config.texthom.use_obj_scale_centroid,
                    config.contact.use_scale,
                    config.texthom.use_contact_feat,
                )

                angle_cond_t = (
                    torch.full((enc_text_b.shape[0], 1), scalar,
                               device=enc_text_b.device, dtype=torch.float32)
                    if scalar is not None else None
                )

                coarse_x_lhand, coarse_x_rhand, coarse_x_obj = diffusion.sampling(
                    texthom, obj_feat_final,
                    enc_text_b, max_nframes,
                    hand_nfeats, obj_nfeats,
                    valid_mask_lhand, valid_mask_rhand, valid_mask_obj,
                    device=torch.device("cuda"),
                    angle_cond=angle_cond_t,
                )

                if est_contact_map is None:
                    cond = proc_cond_contact_estimator(
                        obj_scale_b, obj_feat_b, enc_text_b,
                        npts, config.contact.use_scale,
                    )
                    est_contact_map = contact_estimator.decode(cond)
                    est_contact_map = (est_contact_map[..., 0] > 0.5).long()

                input_lhand, input_rhand, refined_x_obj = proc_refiner_input(
                    coarse_x_lhand, coarse_x_rhand, coarse_x_obj,
                    lhand_layer, rhand_layer,
                    obj_pc_org_b, obj_pc_nor_b,
                    valid_mask_lhand, valid_mask_rhand, valid_mask_obj,
                    est_contact_map, dataset_name,
                    obj_pc_top_idx=op_idx_b,
                )

                refined_x_lhand, refined_x_rhand = refiner(
                    input_lhand, input_rhand,
                    valid_mask_lhand=valid_mask_lhand,
                    valid_mask_rhand=valid_mask_rhand,
                )

                for text_idx in range(enc_text_b.shape[0]):
                    text_dur    = duration[text_idx].item()
                    is_lhand_t  = bool(is_lhand_b[text_idx])
                    is_rhand_t  = bool(is_rhand_b[text_idx])
                    obj_verts_t = obj_verts_b[text_idx]
                    obj_faces_t = obj_faces_b[text_idx]
                    ot_idx_t    = ot_idx_b[text_idx] if ot_idx_b is not None else None

                    x_lhand_s = refined_x_lhand[text_idx, :text_dur]
                    x_rhand_s = refined_x_rhand[text_idx, :text_dur]
                    x_obj_s   = refined_x_obj[text_idx,   :text_dur]

                    # 角度軌跡
                    if obj_nfeats == 10:
                        angles_deg = np.degrees(proc_numpy(x_obj_s[:, 9]))
                        collected[label].append(angles_deg)
                        print(f"  sample={sample_idx} text={text_idx}  "
                              f"final={angles_deg[-1]:.1f}°  max={angles_deg.max():.1f}°")

                    # モーション動画（最初の n_save 本のみ）
                    if save_count < n_save:
                        obj_verts_tf, lhand_verts, lhand_faces_r, \
                        rhand_verts, rhand_faces_r = proc_results(
                            x_lhand_s, x_rhand_s, x_obj_s,
                            obj_verts_t, lhand_layer, rhand_layer,
                            is_lhand_t, is_rhand_t,
                            dataset_name, ot_idx_t,
                        )
                        center_xy = obj_verts_tf[0, :, :2].mean(0)[None, None]
                        if is_lhand_t and lhand_verts is not None:
                            lhand_verts[:, :, :2] -= center_xy
                        if is_rhand_t and rhand_verts is not None:
                            rhand_verts[:, :, :2] -= center_xy
                        obj_verts_tf[:, :, :2] -= center_xy

                        motion_video = render_videos(
                            renderer,
                            lhand_verts,  lhand_faces_r,
                            rhand_verts,  rhand_faces_r,
                            obj_verts_tf, obj_faces_t,
                            is_lhand_t, is_rhand_t,
                        )
                        vid_path = osp.join(vid_folder, f"motion_s{save_count}.mp4")
                        save_video(motion_video, fps=fps, save_path=vid_path)
                        print(f"  [video] {vid_path}")
                        save_count += 1

    # ── グラフ ──
    print("\n[Plot] グラフを生成中...")
    _plot_overlay(collected, conditions, palette, fps, max_nframes, root_dir,
                  config.texthom.model_name)

    # ── 数値まとめ ──
    print()
    print("=" * 55)
    print(f"  {'条件':<18}  {'final mean':>10}  {'final std':>9}")
    print("  " + "-" * 48)
    for label, _ in conditions:
        trajs = collected[label]
        if not trajs:
            continue
        finals = [tr[-1] for tr in trajs]
        print(f"  {label:<18}  {np.mean(finals):>9.1f}°  {np.std(finals):>8.1f}°")
    print("=" * 55)
    print(f"\n[Done] 結果を保存: {root_dir}")


if __name__ == "__main__":
    main()
