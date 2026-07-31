#!/usr/bin/env python
"""
eval_angle_control.py — 角度条件付きモデル（texthom_angle_box_v2）の
角度制御性能を定量評価するスクリプト。

指定した複数の目標角度で推論を実行し、
「目標角度 vs 実際の出力角度」をまとめたグラフと数値テーブルを出力する。

使い方:
    python scripts/eval_angle_control.py \\
        dataset=arctic \\
        texthom.obj_nfeats=10 \\
        texthom.model_name=texthom_angle_box_v2 \\
        texthom.use_angle_cond=True \\
        texthom.angle_min_deg=0 \\
        texthom.angle_max_deg=130 \\
        'test_text=["Open box with both hands."]' \\
        nsamples=5

目標角度を上書きするには:
    +target_angles="[0,20,40,60,80,100,120]"

ベースモデル（角度条件なし）との比較も行う場合:
    +compare_base=true \\
    +base_model_name=texthom_angleloss_w1

出力先: demo_output/eval_angle_control/
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
from lib.utils.demo_utils import (
    get_object_hand_info,
    get_valid_mask_bunch,
    proc_results,
)
from lib.utils.model_utils import (
    build_refiner,
    build_model_and_diffusion,
    build_seq_cvae,
    build_mpnet,
    build_pointnetfeat,
    build_contact_estimator,
)
from lib.models.object import build_object_model
from lib.networks.clip import load_and_freeze_clip, encoded_text
from lib.utils.proc import (
    proc_obj_feat_final,
    proc_cond_contact_estimator,
    proc_refiner_input,
    proc_numpy,
)


# ─────────────────────────────────────────────────────────────────────────────
# ヘルパー
# ─────────────────────────────────────────────────────────────────────────────

def _resample_to_n(traj: np.ndarray, n: int = 100) -> np.ndarray:
    """任意長の角度軌跡を n 点に正規化リサンプリング。"""
    src_x = np.linspace(0, 1, len(traj))
    dst_x = np.linspace(0, 1, n)
    return np.interp(dst_x, src_x, traj)


# ─────────────────────────────────────────────────────────────────────────────
# Target-Angle Hold Ratio (TAHR)
# ─────────────────────────────────────────────────────────────────────────────
#
# 「モーション末尾の窓で、生成角度が目標角度 ±ε° に収まっているフレームの割合」
#
#   TAHR_i(W, ε) = (1 / |W_i|) · Σ_{t ∈ W_i} 1[ |θ_pred_i(t) − θ*| ≤ ε ]
#
# 末尾窓 W は 2 種類を並列で評価する:
#   - fixed  : 末尾 terminal_frames フレーム（デフォルト 30）
#   - ratio  : 末尾 terminal_ratio %       （デフォルト 20%）
# 許容誤差 ε ∈ tolerances（デフォルト [5, 10, 15]°）
# サンプルレベル成功は TAHR_i ≥ success_ratio（デフォルト 0.5）で判定。

def _compute_hold_ratio(
    angles_deg: np.ndarray,
    target_deg: float,
    tolerances,
    terminal_frames: int,
    terminal_ratio: float,
) -> dict:
    """1 サンプルの角度軌跡に対する hold ratio を計算し、
    { (window_name, tol_deg) : ratio } の辞書を返す。"""
    n = len(angles_deg)
    windows = {}
    w_fix_len = min(terminal_frames, n)
    windows[f"last{terminal_frames}f"] = angles_deg[-w_fix_len:]
    w_rat_len = max(1, int(round(n * terminal_ratio)))
    windows[f"last{int(round(terminal_ratio * 100))}pct"] = angles_deg[-w_rat_len:]

    out = {}
    for wname, w in windows.items():
        diff = np.abs(w - target_deg)
        for tol in tolerances:
            out[(wname, float(tol))] = float(np.mean(diff <= tol))
    return out


def _aggregate_hold(results: dict, tolerances, window_names, success_ratio: float):
    """target × (window, tol) ごとに mean/std/success_rate を集計。

    Returns:
        agg[target][(wname, tol)] = {"mean":..., "std":..., "success":...}
    """
    agg = {}
    for t, rec in results.items():
        holds = rec.get("hold", [])
        agg[t] = {}
        for wname in window_names:
            for tol in tolerances:
                key = (wname, float(tol))
                vals = np.array([h[key] for h in holds]) if len(holds) else np.array([])
                if len(vals) == 0:
                    agg[t][key] = {"mean": np.nan, "std": np.nan, "success": np.nan}
                else:
                    agg[t][key] = {
                        "mean": float(vals.mean()),
                        "std":  float(vals.std()),
                        "success": float(np.mean(vals >= success_ratio)),
                    }
    return agg


def _plot_hold_ratio(
    agg: dict, tolerances, window_names, save_dir: str, success_ratio: float
):
    """target × tolerance の hold-ratio ヒートマップと success-rate ヒートマップを保存。"""
    targets = sorted(agg.keys())
    tol_arr = list(tolerances)

    for wname in window_names:
        mean_mat    = np.array([[agg[t][(wname, float(tol))]["mean"]    for tol in tol_arr] for t in targets])
        success_mat = np.array([[agg[t][(wname, float(tol))]["success"] for tol in tol_arr] for t in targets])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, mat, title, cbar_label in zip(
            axes,
            [mean_mat, success_mat],
            [f"Frame-level TAHR  (window={wname})",
             f"Sample success @ TAHR≥{success_ratio}  (window={wname})"],
            ["mean hold ratio", "success rate"],
        ):
            im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(tol_arr)))
            ax.set_xticklabels([f"±{tol:g}°" for tol in tol_arr])
            ax.set_yticks(range(len(targets)))
            ax.set_yticklabels([f"{t:.0f}°" for t in targets])
            ax.set_xlabel("Tolerance ε")
            ax.set_ylabel("Target angle")
            ax.set_title(title, fontsize=12)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat[i, j]
                    if not np.isnan(v):
                        color = "white" if v < 0.55 else "black"
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                color=color, fontsize=10)
            plt.colorbar(im, ax=ax, label=cbar_label)

        fig.tight_layout()
        path = osp.join(save_dir, f"hold_ratio_heatmap_{wname}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [saved] {path}")


def _print_hold_table(
    agg: dict, tolerances, window_names, success_ratio: float, save_dir: str
):
    """テキストテーブルを stdout と txt に出力。"""
    targets = sorted(agg.keys())
    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append(f"  Target-Angle Hold Ratio (TAHR)   success threshold = {success_ratio}")
    lines.append("=" * 90)
    for wname in window_names:
        lines.append(f"\n  [window = {wname}]")
        header = f"  {'target':>7} | " + " | ".join(
            [f"±{tol:>2g}° TAHR(mean±std)   succ" for tol in tolerances]
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for t in targets:
            cells = []
            for tol in tolerances:
                a = agg[t][(wname, float(tol))]
                cells.append(f"{a['mean']:.2f}±{a['std']:.2f}   {a['success']:.2f}")
            lines.append(f"  {t:>6.0f}° | " + " | ".join(cells))
        # overall mean
        overall = {tol: np.nanmean([agg[t][(wname, float(tol))]["success"] for t in targets])
                   for tol in tolerances}
        overall_str = "  overall success: " + "  ".join(
            [f"±{tol:g}°={overall[tol]:.2f}" for tol in tolerances]
        )
        lines.append(overall_str)
    lines.append("=" * 90)

    text = "\n".join(lines)
    print(text)
    with open(osp.join(save_dir, "hold_ratio_table.txt"), "w") as f:
        f.write(text + "\n")


def _plot_summary(results: dict, save_dir: str, angle_min: float, angle_max: float):
    """
    results: {target_deg (int/float): {"final": list[float], "max": list[float], "traj": list[np.ndarray]}}
    """
    os.makedirs(save_dir, exist_ok=True)
    targets = sorted(results.keys())
    n_targets = len(targets)

    # ── 色マップ ──
    cmap = plt.get_cmap("plasma", n_targets)
    colors = [cmap(i) for i in range(n_targets)]

    # ══════════════════════════════════════════════════
    # Figure 1: target vs actual (散布図 + エラーバー)
    # ══════════════════════════════════════════════════
    fig1, ax1 = plt.subplots(figsize=(7, 6))

    means_final = [np.mean(results[t]["final"]) for t in targets]
    stds_final  = [np.std(results[t]["final"])  for t in targets]
    means_max   = [np.mean(results[t]["max"])   for t in targets]

    ax1.errorbar(
        targets, means_final, yerr=stds_final,
        fmt="o-", capsize=6, color="steelblue", linewidth=2,
        markersize=8, label="Actual final angle (mean ± std)",
    )
    ax1.plot(
        targets, means_max,
        "s--", color="darkorange", linewidth=1.5,
        markersize=6, label="Actual max angle (mean)",
    )
    # 理想直線
    lim = [angle_min, angle_max]
    ax1.plot(lim, lim, "r--", linewidth=1.5, label="Ideal  y = x")

    ax1.set_xlabel("Target angle (deg)", fontsize=13)
    ax1.set_ylabel("Actual angle (deg)", fontsize=13)
    ax1.set_title("Angle controllability: target vs actual", fontsize=14)
    ax1.set_xlim(angle_min - 5, angle_max + 5)
    ax1.set_ylim(angle_min - 5, angle_max + 20)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    fig1.tight_layout()
    path1 = osp.join(save_dir, "target_vs_actual.png")
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)
    print(f"  [saved] {path1}")

    # ══════════════════════════════════════════════════
    # Figure 2: 目標角度ごとの角度軌跡オーバーレイ
    # ══════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    NORM = 100  # 正規化時間軸の点数

    for ci, t in enumerate(targets):
        trajs = results[t]["traj"]
        resampled = np.stack([_resample_to_n(tr, NORM) for tr in trajs])
        for row in resampled:
            ax2.plot(row, color=colors[ci], alpha=0.25, linewidth=0.9)
        mean_traj = resampled.mean(0)
        std_traj  = resampled.std(0)
        xs = np.arange(NORM)
        ax2.fill_between(
            xs, mean_traj - std_traj, mean_traj + std_traj,
            alpha=0.15, color=colors[ci],
        )
        ax2.plot(
            xs, mean_traj,
            color=colors[ci], linewidth=2.2,
            label=f"target={t:.0f}°",
        )
        # 目標ラインを破線で表示
        ax2.axhline(t, color=colors[ci], linestyle=":", linewidth=0.8, alpha=0.7)

    ax2.set_xlabel("Normalised time (%)", fontsize=13)
    ax2.set_ylabel("Articulation angle (deg)", fontsize=13)
    ax2.set_title("Angle trajectories per target angle", fontsize=14)
    ax2.legend(loc="upper left", fontsize=10, ncol=2)
    ax2.grid(True, linestyle="--", alpha=0.5)
    fig2.tight_layout()
    path2 = osp.join(save_dir, "angle_trajectories.png")
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"  [saved] {path2}")

    # ══════════════════════════════════════════════════
    # Figure 3: 残差ヒストグラム（target - actual final）
    # ══════════════════════════════════════════════════
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    all_residuals = []
    for t in targets:
        for f in results[t]["final"]:
            all_residuals.append(f - t)
    ax3.hist(all_residuals, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
    ax3.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
    ax3.axvline(np.mean(all_residuals), color="orange", linestyle="-",
                linewidth=1.5, label=f"Mean residual = {np.mean(all_residuals):.1f}°")
    ax3.set_xlabel("Actual − Target (deg)", fontsize=13)
    ax3.set_ylabel("Count", fontsize=13)
    ax3.set_title("Angle prediction residuals (all targets)", fontsize=14)
    ax3.legend(fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)
    fig3.tight_layout()
    path3 = osp.join(save_dir, "residuals.png")
    fig3.savefig(path3, dpi=150)
    plt.close(fig3)
    print(f"  [saved] {path3}")

    return means_final, stds_final


def _print_table(results: dict):
    targets = sorted(results.keys())
    print()
    print("=" * 62)
    print("  Angle Control Evaluation  —  texthom_angle_box_v2")
    print("=" * 62)
    print(f"  {'Target':>8}  {'Final mean':>10}  {'Final std':>9}  {'Max mean':>8}  {'MAE':>7}")
    print("  " + "-" * 56)
    maes = []
    for t in targets:
        finals = results[t]["final"]
        maxes  = results[t]["max"]
        mean_f = np.mean(finals)
        std_f  = np.std(finals)
        mean_m = np.mean(maxes)
        mae    = abs(mean_f - t)
        maes.append(mae)
        print(f"  {t:>7.0f}°  {mean_f:>9.1f}°  {std_f:>8.1f}°  {mean_m:>7.1f}°  {mae:>6.1f}°")
    print("  " + "-" * 56)
    print(f"  {'Overall MAE':>24}             {np.mean(maes):>6.1f}°")
    print("=" * 62)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../configs", config_name="config")
@torch.no_grad()
def main(config):
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_object(config)
    config = edict(config)

    data_config   = config.dataset
    dataset_name  = data_config.name  # "arctic"
    max_nframes   = data_config.max_nframes
    hand_nfeats   = config.texthom.hand_nfeats
    obj_nfeats    = config.texthom.obj_nfeats
    nsamples      = config.nsamples
    text          = list(config.test_text)

    # ── 角度設定 ──
    use_angle_cond = bool(config.texthom.use_angle_cond)
    angle_min_deg  = float(config.texthom.angle_min_deg)
    angle_max_deg  = float(config.texthom.angle_max_deg)
    if not use_angle_cond:
        raise ValueError(
            "このスクリプトは use_angle_cond=True のモデル用です。\n"
            "  例: texthom.use_angle_cond=True texthom.model_name=texthom_angle_box_v2"
        )

    # 目標角度リスト（CLI で +target_angles="[0,30,60,90]" と上書き可）
    if "target_angles" in config:
        target_angles = [float(a) for a in config.target_angles]
    else:
        target_angles = list(
            np.linspace(angle_min_deg, angle_max_deg, 7).round(0)
        )

    # ── TAHR 設定（CLI で +... で上書き可）──
    terminal_frames = int(config.get("terminal_frames", 30))
    terminal_ratio  = float(config.get("terminal_ratio", 0.2))
    if "tolerances" in config:
        tolerances = [float(t) for t in config.tolerances]
    else:
        tolerances = [5.0, 10.0, 15.0]
    success_ratio   = float(config.get("success_ratio", 0.5))
    window_names = [
        f"last{terminal_frames}f",
        f"last{int(round(terminal_ratio * 100))}pct",
    ]

    print(f"\n[Eval] target_angles = {[f'{a:.0f}' for a in target_angles]}")
    print(f"[Eval] nsamples per target = {nsamples}")
    print(f"[Eval] TAHR window     = last {terminal_frames} frames / last {int(terminal_ratio*100)}%")
    print(f"[Eval] TAHR tolerances = {tolerances}  |  success threshold = {success_ratio}\n")

    # ── 出力フォルダ ──
    save_dir = osp.join(
        osp.dirname(osp.abspath(osp.dirname(__file__))),
        "demo_output", "eval_angle_control",
        config.texthom.model_name,
    )
    os.makedirs(save_dir, exist_ok=True)

    # ── モデル読み込み ──
    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=data_config.flat_hand).cuda()
    rhand_layer = build_mano_aa(is_rhand=True,  flat_hand=data_config.flat_hand).cuda()

    refiner    = build_refiner(config, test=True)
    texthom, diffusion = build_model_and_diffusion(
        config, lhand_layer, rhand_layer, test=True
    )
    clip_model      = load_and_freeze_clip(config.clip.clip_version).cuda()
    mpnet           = build_mpnet(config)
    seq_cvae        = build_seq_cvae(config, test=True)
    pointnet        = build_pointnetfeat(config, test=True)
    contact_estimator = build_contact_estimator(config, test=True)
    object_model    = build_object_model(data_config.data_obj_pc_path)

    # ── オブジェクト・テキスト特徴量を事前計算（全ターゲット角で共有）──
    is_lhand, is_rhand, \
    obj_pc_org, obj_pc_normal_org, \
    normalized_obj_pc, point_sets, \
    obj_cent, obj_scale, \
    obj_verts, obj_faces, \
    obj_top_idx, obj_pc_top_idx = get_object_hand_info(
        object_model, clip_model, text,
        data_config.obj_root, data_config, mpnet,
    )

    npts       = normalized_obj_pc.shape[1]
    enc_text   = encoded_text(clip_model, text)
    obj_feat   = pointnet(normalized_obj_pc)
    batch_num  = len(text) // 64 + 1

    # ── 結果格納 ──
    results = {t: {"final": [], "max": [], "traj": [], "hold": []} for t in target_angles}

    # ── ターゲット角度ループ ──
    for target_deg in target_angles:
        _denom = max(angle_max_deg - angle_min_deg, 1e-6)
        angle_cond_scalar = (target_deg - angle_min_deg) / _denom
        angle_cond_scalar = float(np.clip(angle_cond_scalar, 0.0, 1.0))
        print(f"\n── target={target_deg:.0f}°  (cond_scalar={angle_cond_scalar:.4f}) ──")

        for sample_idx in range(nsamples):
            for batch_idx in range(batch_num):
                sl = slice(batch_idx * 64, (batch_idx + 1) * 64)

                enc_text_batch          = enc_text[sl]
                is_lhand_batch          = is_lhand[sl]
                is_rhand_batch          = is_rhand[sl]
                obj_cent_batch          = obj_cent[sl]
                obj_scale_batch         = obj_scale[sl]
                obj_feat_batch          = obj_feat[sl]
                obj_pc_org_batch        = obj_pc_org[sl]
                obj_pc_normal_org_batch = obj_pc_normal_org[sl]
                normalized_obj_pc_batch = normalized_obj_pc[sl]
                point_sets_batch        = point_sets[sl]
                obj_top_idx_batch       = obj_top_idx[sl] if dataset_name == "arctic" else None
                obj_pc_top_idx_batch    = obj_pc_top_idx[sl] if dataset_name == "arctic" else None

                if enc_text_batch.shape[0] == 0:
                    continue

                duration = seq_cvae.decode(enc_text_batch)
                duration *= 150
                duration = duration.long()
                # 角度条件モデルは全フレームを使う
                duration = torch.full_like(duration, max_nframes)

                valid_mask_lhand, valid_mask_rhand, valid_mask_obj = \
                    get_valid_mask_bunch(
                        is_lhand_batch, is_rhand_batch,
                        max_nframes, duration,
                    )

                obj_feat_final, est_contact_map = proc_obj_feat_final(
                    contact_estimator,
                    obj_scale_batch, obj_cent_batch,
                    obj_feat_batch, enc_text_batch, npts,
                    config.texthom.use_obj_scale_centroid,
                    config.contact.use_scale,
                    config.texthom.use_contact_feat,
                )

                angle_cond_tensor = torch.full(
                    (enc_text_batch.shape[0], 1),
                    angle_cond_scalar,
                    device=enc_text_batch.device,
                    dtype=torch.float32,
                )

                coarse_x_lhand, coarse_x_rhand, coarse_x_obj = \
                    diffusion.sampling(
                        texthom, obj_feat_final,
                        enc_text_batch, max_nframes,
                        hand_nfeats, obj_nfeats,
                        valid_mask_lhand, valid_mask_rhand, valid_mask_obj,
                        device=torch.device("cuda"),
                        angle_cond=angle_cond_tensor,
                    )

                if est_contact_map is None:
                    condition = proc_cond_contact_estimator(
                        obj_scale_batch, obj_feat_batch, enc_text_batch,
                        npts, config.contact.use_scale,
                    )
                    est_contact_map = contact_estimator.decode(condition)
                    est_contact_map = (est_contact_map[..., 0] > 0.5).long()

                _, _, refined_x_obj = proc_refiner_input(
                    coarse_x_lhand, coarse_x_rhand, coarse_x_obj,
                    lhand_layer, rhand_layer,
                    obj_pc_org_batch, obj_pc_normal_org_batch,
                    valid_mask_lhand, valid_mask_rhand, valid_mask_obj,
                    est_contact_map, dataset_name,
                    obj_pc_top_idx=obj_pc_top_idx_batch,
                )

                for text_idx in range(enc_text_batch.shape[0]):
                    text_dur = duration[text_idx].item()
                    x_obj_s  = refined_x_obj[text_idx, :text_dur]

                    if obj_nfeats == 10:
                        angles_rad = proc_numpy(x_obj_s[:, 9])
                        angles_deg = np.degrees(angles_rad)
                    else:
                        # angle channel がない場合はスキップ
                        print(f"  [Warning] obj_nfeats={obj_nfeats} != 10, angle unavailable")
                        continue

                    final_angle = float(angles_deg[-1])
                    max_angle   = float(angles_deg.max())

                    hold_dict = _compute_hold_ratio(
                        angles_deg, target_deg,
                        tolerances, terminal_frames, terminal_ratio,
                    )

                    results[target_deg]["final"].append(final_angle)
                    results[target_deg]["max"].append(max_angle)
                    results[target_deg]["traj"].append(angles_deg)
                    results[target_deg]["hold"].append(hold_dict)

                    # 代表 tolerance (=最初) の hold ratio を進捗表示
                    _first_key = (window_names[0], float(tolerances[0]))
                    print(
                        f"  sample={sample_idx} text={text_idx}  "
                        f"final={final_angle:.1f}°  max={max_angle:.1f}°  "
                        f"TAHR[{window_names[0]},±{tolerances[0]:g}°]={hold_dict[_first_key]:.2f}"
                    )

    # ── 集計・出力 ──
    _print_table(results)
    _plot_summary(results, save_dir, angle_min_deg, angle_max_deg)

    # ── TAHR 集計・出力 ──
    agg = _aggregate_hold(results, tolerances, window_names, success_ratio)
    _print_hold_table(agg, tolerances, window_names, success_ratio, save_dir)
    _plot_hold_ratio(agg, tolerances, window_names, save_dir, success_ratio)

    print(f"\n[Done] 評価結果を保存しました: {save_dir}")


if __name__ == "__main__":
    main()
