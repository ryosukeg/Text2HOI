#!/usr/bin/env bash
# demo_angle_all.sh
# texthom_angle_all（全物体角度制御モデル）を使って
# 各物体ごとにモーション生成・グラフ出力を行う。
#
# 出力先: demo_output/eval_angle_all/<object>/
#   overlay_normalized.png  正規化時間軸グラフ
#   overlay_realtime.png    実時間（フレーム）軸グラフ
#   target_Xdeg/motion_sN.mp4  各角度のモーション動画（3本）
#   角度指定なし/motion_sN.mp4 制御なしのモーション動画（3本）
#
# 使い方: bash scripts/demo_angle_all.sh

set -e
cd "$(dirname "$0")/.."

MODEL=texthom_angle_all
NSAMPLES=5     # 角度グラフ用サンプル数
N_SAVE=3       # 保存するモーション動画本数
COMMON="dataset=arctic \
    texthom.obj_nfeats=10 \
    texthom.model_name=${MODEL} \
    texthom.use_angle_cond=True \
    texthom.angle_min_deg=0 \
    texthom.angle_max_deg=220 \
    nsamples=${NSAMPLES} \
    +n_save=${N_SAVE} \
    hydra.output_subdir=null \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled"

run_demo() {
    local OBJ="$1"
    local TEXT="$2"
    local ANGLES="$3"
    echo ""
    echo "======================================================"
    echo "  物体: ${OBJ}  text: ${TEXT}"
    echo "  target_angles: ${ANGLES}"
    echo "======================================================"
    WANDB_MODE=disabled python scripts/compare_angle_conditions.py \
        ${COMMON} \
        "+test_text=[${TEXT}]" \
        "+target_angles=${ANGLES}" \
        "+save_subdir=eval_angle_all/${OBJ}"
}

# ── 各物体のデモ ──
# 物体ごとにデータの最大角度を考慮してターゲット範囲を設定
# （全モデル訓練範囲は 0〜220°）

run_demo "box"          "Open box with both hands."          "[0,30,60,90,120]"
run_demo "laptop"       "Open laptop with both hands."       "[0,30,60,90,120]"
run_demo "scissors"     "Cut scissors with right hand."      "[0,15,30,45,60]"
run_demo "ketchup"      "Open ketchup with both hands."      "[0,40,80,120,160,200]"
run_demo "microwave"    "Open microwave with both hands."    "[0,25,50,75,100]"
run_demo "notebook"     "Open notebook with both hands."     "[0,40,80,120,160]"
run_demo "waffleiron"   "Open waffleiron with both hands."   "[0,40,80,120,160]"
run_demo "phone"        "Open phone with both hands."        "[0,40,80,120,160]"

echo ""
echo "======================================================"
echo "  完了！ 結果は demo_output/eval_angle_all/ 以下に保存されました"
echo ""
echo "  demo_output/eval_angle_all/"
for OBJ in box laptop scissors ketchup microwave notebook waffleiron phone; do
    echo "    ${OBJ}/"
    echo "      overlay_normalized.png"
    echo "      overlay_realtime.png"
    echo "      target_Xdeg/motion_s{0..2}.mp4"
    echo "      角度指定なし/motion_s{0..2}.mp4"
done
echo "======================================================"
