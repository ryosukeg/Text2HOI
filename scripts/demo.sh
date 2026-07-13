python demo/demo.py \
    +test_text="[Place a cappuccino with the right hand.]" \
    +nsamples=4 \
    hydra.output_subdir=null \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled

python demo/demo.py dataset=grab \
    +test_text="[Play a flute with both hands.]" \
    +nsamples=4 \
    hydra.output_subdir=null \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled

python demo/demo.py dataset=arctic \
    +test_text="[Type a labtop with both hands.]" \
    +nsamples=4 \
    texthom.obj_nfeats=10 \
    hydra.output_subdir=null \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled

# ---- Method 2: side-by-side comparison for ARCTIC ----
# Original model (existing checkpoint, unchanged):
# python demo/demo.py dataset=arctic \
#     +test_text="[Open a box with both hands.]" \
#     +nsamples=4 \
#     texthom.obj_nfeats=10 \
#     hydra.output_subdir=null \
#     hydra/job_logging=disabled \
#     hydra/hydra_logging=disabled
#
# Angle-conditioned model (new checkpoint dir, angle passed via CLI):
# python demo/demo.py dataset=arctic \
#     +test_text="[Open a box with both hands.]" \
#     +nsamples=4 \
#     +target_angle_deg=45 \
#     texthom.obj_nfeats=10 \
#     texthom.model_name=texthom_angle \
#     texthom.use_angle_cond=True \
#     texthom.weight_path=checkpoints/arctic/texthom_angle.pth \
#     hydra.output_subdir=null \
#     hydra/job_logging=disabled \
#     hydra/hydra_logging=disabled