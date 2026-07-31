#!/usr/bin/env bash
# eval_angle_control.sh
# texthom_angle_box_v2 の角度制御性能を評価する。
#
# 使い方:
#   bash scripts/eval_angle_control.sh
#
# 出力先: demo_output/eval_angle_control/texthom_angle_box_v2/
#   - target_vs_actual.png  : 目標角度 vs 実際の出力角度（散布図）
#   - angle_trajectories.png: 目標角度ごとの角度軌跡オーバーレイ
#   - residuals.png         : 残差ヒストグラム

set -e
cd "$(dirname "$0")/.."

MODEL=texthom_angle_box_v2
NSAMPLES=5                              # 目標角度ごとのサンプル数
TEXT="[Open box with both hands.]"
# 0°〜120° を7段階でスイープ（モデルの訓練範囲 0〜130°）
TARGET_ANGLES="[0,20,40,60,80,100,120]"

# ── TAHR 設定 ──
# 末尾窓: 最後 TERMINAL_FRAMES フレーム、および 末尾 TERMINAL_RATIO%
TERMINAL_FRAMES=30
TERMINAL_RATIO=0.2
TOLERANCES="[5,10,15]"                  # ±° の許容誤差
SUCCESS_RATIO=0.5                       # サンプル成功判定の TAHR しきい値

echo "======================================================"
echo "  Angle Control Evaluation: ${MODEL}"
echo "  nsamples per target: ${NSAMPLES}"
echo "  target angles:       ${TARGET_ANGLES}"
echo "  TAHR window:         last ${TERMINAL_FRAMES}f / last ${TERMINAL_RATIO}"
echo "  TAHR tolerances:     ${TOLERANCES}"
echo "  Success threshold:   ${SUCCESS_RATIO}"
echo "======================================================"

WANDB_MODE=disabled python scripts/eval_angle_control.py \
    dataset=arctic \
    texthom.obj_nfeats=10 \
    texthom.model_name=${MODEL} \
    texthom.use_angle_cond=True \
    texthom.angle_min_deg=0 \
    texthom.angle_max_deg=130 \
    "+target_angles=${TARGET_ANGLES}" \
    "+test_text=${TEXT}" \
    "+terminal_frames=${TERMINAL_FRAMES}" \
    "+terminal_ratio=${TERMINAL_RATIO}" \
    "+tolerances=${TOLERANCES}" \
    "+success_ratio=${SUCCESS_RATIO}" \
    nsamples=${NSAMPLES} \
    hydra.output_subdir=null \
    "hydra/job_logging=disabled" \
    "hydra/hydra_logging=disabled"

echo ""
echo "======================================================"
echo "  完了。以下に結果が保存されました:"
echo "  demo_output/eval_angle_control/${MODEL}/"
echo "  - target_vs_actual.png    目標 vs 実際の散布図"
echo "  - angle_trajectories.png  各ターゲット角の軌跡"
echo "  - residuals.png           残差ヒストグラム"
echo "  - hold_ratio_heatmap_*.png  TAHR ヒートマップ (window別)"
echo "  - hold_ratio_table.txt      TAHR テキストテーブル"
echo "======================================================"
