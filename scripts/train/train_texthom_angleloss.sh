#!/usr/bin/env bash
# Train arctic texthom with an added object-angle MSE loss.
# Uses a distinct model_name so the original checkpoints (texthom.pth,
# texthom_angle.pth, ...) are NOT overwritten.
#
# use_angle_cond=False: unconditional model. The angle-loss alone pushes the
# whole predicted angle trajectory toward GT for every object / frame.

python train/train_texthom.py \
    dataset=arctic \
    texthom.obj_nfeats=10 \
    texthom.model_name=texthom_angleloss_w1 \
    texthom.lambda_angle=1.0 \
    texthom.use_angle_cond=False

# ---- lambda_angle=5.0 (stronger angle emphasis, may cause unnatural motion) ----
# python train/train_texthom.py \
#     dataset=arctic \
#     texthom.obj_nfeats=10 \
#     texthom.model_name=texthom_angleloss \
#     texthom.lambda_angle=5.0 \
#     texthom.use_angle_cond=False

# ---- Optional: also condition on target angle (Method-2 style) ----
# python train/train_texthom.py \
#     dataset=arctic \
#     texthom.obj_nfeats=10 \
#     texthom.model_name=texthom_angleloss_cond \
#     texthom.lambda_angle=1.0 \
#     texthom.use_angle_cond=True \
#     texthom.angle_min_deg=0 texthom.angle_max_deg=90 \
#     texthom.angle_tol_deg=5
