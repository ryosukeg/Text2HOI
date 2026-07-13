#!/bin/bash
# 旧モデル vs 角度条件モデル（30°/60°/90°）の比較推論スクリプト
# 使い方: bash scripts/compare_angle.sh
# 前提: checkpoints/arctic/texthom_angle_box.pth が存在すること

set -e
TEXT="[Open a box with both hands.]"
NSAMPLES=4
COMMON="dataset=arctic +nsamples=${NSAMPLES} texthom.obj_nfeats=10 \
    hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled"

echo "=== [1/4] 旧モデル（角度条件なし）==="
WANDB_MODE=disabled python demo/demo.py ${COMMON} \
    "+test_text=${TEXT}"
mkdir -p demo_output/compare/old_model
cp demo_output/arctic/motion/generated_b0_t0_s*.mp4 demo_output/compare/old_model/ 2>/dev/null || true
cp demo_output/arctic/angle_plots/angle_comparison_b0_t0.png \
   demo_output/compare/old_model/angle_comparison.png 2>/dev/null || true

for DEG in 30 60 90; do
    echo "=== [角度条件モデル] target_angle_deg=${DEG}° ==="
    WANDB_MODE=disabled python demo/demo.py ${COMMON} \
        "+test_text=${TEXT}" \
        "+target_angle_deg=${DEG}" \
        texthom.model_name=texthom_angle_box \
        texthom.use_angle_cond=True \
        "texthom.weight_path=checkpoints/arctic/texthom_angle_box.pth"
    mkdir -p demo_output/compare/angle_${DEG}deg
    cp demo_output/arctic/motion/generated_b0_t0_s*.mp4 \
       demo_output/compare/angle_${DEG}deg/ 2>/dev/null || true
    cp demo_output/arctic/angle_plots/angle_comparison_b0_t0.png \
       demo_output/compare/angle_${DEG}deg/angle_comparison.png 2>/dev/null || true
    echo "  → saved to demo_output/compare/angle_${DEG}deg/"
done

echo ""
echo "=== 完了 ==="
echo "比較フォルダ構成:"
echo "  demo_output/compare/old_model/           : 旧モデル（角度条件なし）"
echo "  demo_output/compare/angle_30deg/         : 新モデル target=30°"
echo "  demo_output/compare/angle_60deg/         : 新モデル target=60°"
echo "  demo_output/compare/angle_90deg/         : 新モデル target=90°"

echo ""
echo "=== [後処理打ち切り] 旧モデル + truncate ==="
for DEG in 30 60 90; do
    echo "--- truncate_at_angle_deg=${DEG}° ---"
    WANDB_MODE=disabled python demo/demo.py ${COMMON} \
        "+test_text=${TEXT}" \
        "+truncate_at_angle_deg=${DEG}"
    mkdir -p demo_output/compare/posthoc_${DEG}deg
    cp demo_output/arctic/motion/generated_b0_t0_s*.mp4 \
       demo_output/compare/posthoc_${DEG}deg/ 2>/dev/null || true
    cp demo_output/arctic/angle_plots/angle_comparison_b0_t0.png \
       demo_output/compare/posthoc_${DEG}deg/angle_comparison.png 2>/dev/null || true
    echo "  → saved to demo_output/compare/posthoc_${DEG}deg/"
done

echo ""
echo "=== 全比較フォルダ ==="
echo "  old_model/      : 旧モデル（条件なし）"
echo "  angle_{N}deg/   : 新モデル（角度条件学習）target=N°"
echo "  posthoc_{N}deg/ : 旧モデル + 後処理打ち切り target=N°"
