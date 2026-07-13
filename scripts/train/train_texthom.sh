python train/train_texthom.py dataset=h2o dataset.augm=False texthom.iteration=90000
# python train/train_texthom.py dataset=grab
# python train/train_texthom.py dataset=arctic texthom.obj_nfeats=10

# ---- Method 2: angle-conditioned model for ARCTIC (separate checkpoint dir) ----
# Uses a distinct model_name so the original checkpoint (texthom.pth) is kept intact.
# python train/train_texthom.py dataset=arctic texthom.obj_nfeats=10 \
#     texthom.model_name=texthom_angle \
#     texthom.use_angle_cond=True \
#     texthom.angle_min_deg=0 texthom.angle_max_deg=90 \
#     texthom.angle_tol_deg=5 texthom.angle_condition_mode=first_reach