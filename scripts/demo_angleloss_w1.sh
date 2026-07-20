#!/usr/bin/env bash
# Demo script for texthom_angleloss_w1:
#   - 10 generated samples per motion (for angle graph comparison)
#   - 3 target motions: box open, ketchup open, scissors cut
#   - Videos saved for all 3 motions
# Output: demo_output/arctic_texthom_angleloss_w1/

set -e
cd "$(dirname "$0")/.."

MODEL=texthom_angleloss_w1

echo "=== [1/3] Open box with both hands ==="
python demo/demo.py \
    dataset=arctic \
    texthom.obj_nfeats=10 \
    texthom.model_name=${MODEL} \
    'test_text=["Open box with both hands."]' \
    nsamples=10

echo "=== [2/3] Open ketchup with both hands ==="
python demo/demo.py \
    dataset=arctic \
    texthom.obj_nfeats=10 \
    texthom.model_name=${MODEL} \
    'test_text=["Open ketchup with both hands."]' \
    nsamples=10

echo "=== [3/3] Cut scissors with both hands ==="
python demo/demo.py \
    dataset=arctic \
    texthom.obj_nfeats=10 \
    texthom.model_name=${MODEL} \
    'test_text=["Cut scissors with both hands."]' \
    nsamples=10

echo ""
echo "Done. Results saved under demo_output/arctic_${MODEL}/"
