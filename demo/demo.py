import os.path as osp
import sys
sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))

import numpy as np
import hydra
from omegaconf import OmegaConf
from easydict import EasyDict as edict
import time

import torch

from lib.models.mano import build_mano_aa
from lib.utils.renderer import Renderer
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
from lib.utils.file import (
    make_save_folder, 
    save_video, 
    save_frames,
    save_angle_plot,
    save_mesh_obj, 
)
from lib.utils.proc import (
    proc_obj_feat_final, 
    proc_cond_contact_estimator, 
    proc_refiner_input, 
    proc_numpy, 
    proc_torch_cuda, 
)
from lib.utils.visualize import render_videos

@hydra.main(version_base=None, config_path="../configs", config_name="config")
@torch.no_grad()
def main(config):
    start_time = time.time()
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_object(config)
    config = edict(config)
    data_config = config.dataset
    dataset_name = data_config.name

    save_root = f"demo_output/{dataset_name}"
    result_folder = make_save_folder(save_root)
    
    save_obj = config.save_obj
    nsamples = config.nsamples
    fps = config.fps
    max_nframes = data_config.max_nframes
    text = config.test_text
    hand_nfeats = config.texthom.hand_nfeats
    obj_nfeats = config.texthom.obj_nfeats

    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=data_config.flat_hand).cuda()
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=data_config.flat_hand).cuda()

    refiner = build_refiner(config, test=True)
    texthom, diffusion \
        = build_model_and_diffusion(config, lhand_layer, rhand_layer, test=True)
    clip_model = load_and_freeze_clip(config.clip.clip_version)
    clip_model = clip_model.cuda()
    mpnet = build_mpnet(config)
    seq_cvae = build_seq_cvae(config, test=True)
    pointnet = build_pointnetfeat(config, test=True)
    contact_estimator = build_contact_estimator(config, test=True)
    object_model = build_object_model(data_config.data_obj_pc_path)
    
    renderer = Renderer(device="cuda", camera=f"{dataset_name}_front")

    is_lhand, is_rhand, \
    obj_pc_org, obj_pc_normal_org, \
    normalized_obj_pc, point_sets, \
    obj_cent, obj_scale, \
    obj_verts, obj_faces, \
    obj_top_idx, obj_pc_top_idx \
        = get_object_hand_info(
            object_model, 
            clip_model, 
            text, 
            data_config.obj_root, 
            data_config,
            mpnet,  
        )
    
    bs, npts = normalized_obj_pc.shape[:2]
    
    enc_text = encoded_text(clip_model, text)
    obj_feat = pointnet(normalized_obj_pc)
    
    batch_num = len(text)//64 + 1

    # Pre-load GT data for ARCTIC angle comparison and GT rendering
    gt_data = None
    if dataset_name == "arctic":
        _raw = np.load(data_config.data_path, allow_pickle=True)
        gt_data = {k: _raw[k] for k in (
            "x_lhand", "x_rhand", "x_obj", "x_obj_angle",
            "object_name", "action_name", "nframes",
            "is_lhand", "is_rhand",
        )}

    # Accumulate generated angles across samples for multi-sample comparison plot
    gen_angles_all = {}   # key: (batch_idx, text_idx) → list of angle arrays
    gt_angles_cache = {}  # key: (batch_idx, text_idx) → (gt_angles_list, gt_idxs)

    for sample_idx in range(nsamples):
        for batch_idx in range(batch_num):
            ecn_text_batch = enc_text[batch_idx*64:(batch_idx+1)*64]
            is_lhand_batch = is_lhand[batch_idx*64:(batch_idx+1)*64]
            is_rhand_batch = is_rhand[batch_idx*64:(batch_idx+1)*64]
            obj_cent_batch = obj_cent[batch_idx*64:(batch_idx+1)*64]
            obj_scale_batch = obj_scale[batch_idx*64:(batch_idx+1)*64]
            obj_feat_batch = obj_feat[batch_idx*64:(batch_idx+1)*64]
            enc_text_batch = enc_text[batch_idx*64:(batch_idx+1)*64]
            obj_pc_org_batch = obj_pc_org[batch_idx*64:(batch_idx+1)*64]
            obj_pc_normal_org_batch = obj_pc_normal_org[batch_idx*64:(batch_idx+1)*64]
            normalized_obj_pc_batch = normalized_obj_pc[batch_idx*64:(batch_idx+1)*64]
            point_sets_batch = point_sets[batch_idx*64:(batch_idx+1)*64]
            obj_verts_batch = obj_verts[batch_idx*64:(batch_idx+1)*64]
            obj_faces_batch = obj_faces[batch_idx*64:(batch_idx+1)*64]
            
            if dataset_name == "arctic":
                obj_top_idx_batch = obj_top_idx[batch_idx*64:(batch_idx+1)*64]
                obj_pc_top_idx_batch = obj_pc_top_idx[batch_idx*64:(batch_idx+1)*64]
            else:
                obj_top_idx_batch = None
                obj_pc_top_idx_batch = None
                
            duration = seq_cvae.decode(ecn_text_batch)
            duration *= 150
            duration = duration.long()
            valid_mask_lhand, valid_mask_rhand, valid_mask_obj \
                = get_valid_mask_bunch(
                    is_lhand_batch, is_rhand_batch, 
                    max_nframes, duration
                )
            obj_feat_final, est_contact_map = proc_obj_feat_final(
                contact_estimator,  
                obj_scale_batch, obj_cent_batch, 
                obj_feat_batch, enc_text_batch, npts, 
                config.texthom.use_obj_scale_centroid, 
                config.contact.use_scale, 
                config.texthom.use_contact_feat, 
            )
            coarse_x_lhand, coarse_x_rhand, coarse_x_obj \
                = diffusion.sampling(
                    texthom, obj_feat_final, 
                    enc_text_batch, max_nframes, 
                    hand_nfeats, obj_nfeats, 
                    valid_mask_lhand, 
                    valid_mask_rhand, 
                    valid_mask_obj, 
                    device=torch.device("cuda")
                )
            
            if est_contact_map is None:
                condition = proc_cond_contact_estimator(
                    obj_scale_batch, obj_feat_batch, enc_text_batch, 
                    npts, config.contact.use_scale
                )
                est_contact_map = contact_estimator.decode(condition)
                est_contact_map = (est_contact_map[..., 0] > 0.5).long()
            
            input_lhand, input_rhand, refined_x_obj, \
                = proc_refiner_input(
                    coarse_x_lhand, coarse_x_rhand, coarse_x_obj, 
                    lhand_layer, rhand_layer, obj_pc_org_batch, obj_pc_normal_org_batch, 
                    valid_mask_lhand, valid_mask_rhand, valid_mask_obj, 
                    est_contact_map, dataset_name, obj_pc_top_idx=obj_pc_top_idx_batch
                )
            
            refined_x_lhand, refined_x_rhand \
                = refiner(
                    input_lhand, input_rhand,  
                    valid_mask_lhand=valid_mask_lhand, 
                    valid_mask_rhand=valid_mask_rhand, 
                )

            for text_idx in range(normalized_obj_pc_batch.shape[0]):
                print(f"Batch #{batch_idx} Samples #{sample_idx} Text #{text_idx}")
                print(duration[text_idx])
                
                is_lhand_text = is_lhand_batch[text_idx]
                is_rhand_text = is_rhand_batch[text_idx]
                obj_verts_text = obj_verts_batch[text_idx]
                obj_faces_text = obj_faces_batch[text_idx]

                if dataset_name == "arctic":
                    obj_top_idx_text = obj_top_idx_batch[text_idx]
                else:
                    obj_top_idx_text = None
                
                text_duration = duration[text_idx].item()
                
                refined_x_lhand_sampled = refined_x_lhand[text_idx][:text_duration]
                refined_x_rhand_sampled = refined_x_rhand[text_idx][:text_duration]
                refined_x_obj_sampled = refined_x_obj[text_idx][:text_duration]
                
                refined_obj_verts_tf, refined_lhand_verts, lhand_faces, \
                refined_rhand_verts, rhand_faces = \
                    proc_results(
                        refined_x_lhand_sampled, refined_x_rhand_sampled, refined_x_obj_sampled, 
                        obj_verts_text, lhand_layer, rhand_layer, 
                        is_lhand_text, is_rhand_text, 
                        dataset_name, obj_top_idx_text
                    )
                
                if is_lhand_text:
                    refined_lhand_verts[:, :, :2] = refined_lhand_verts[:, :, :2] - refined_obj_verts_tf[0, :, :2].mean(0)[None, None]
                
                if is_rhand_text:
                    refined_rhand_verts[:, :, :2] = refined_rhand_verts[:, :, :2] - refined_obj_verts_tf[0, :, :2].mean(0)[None, None]
                
                refined_obj_verts_tf[:, :, :2] = refined_obj_verts_tf[:, :, :2] - refined_obj_verts_tf[0, :, :2].mean(0)[None, None]
                
                motion_video = render_videos(
                    renderer, refined_lhand_verts, lhand_faces, 
                    refined_rhand_verts, rhand_faces, 
                    refined_obj_verts_tf, obj_faces_text, 
                    is_lhand_text, is_rhand_text, 
                )
                
                save_video(
                    motion_video, fps=fps, 
                    save_path=osp.join(
                        result_folder, 
                        "motion", 
                        f"generated_b{batch_idx}_t{text_idx}_s{sample_idx}.mp4"
                    )
                )

                save_frames(
                    motion_video,
                    save_folder=osp.join(
                        result_folder,
                        "motion_frames",
                        f"generated_b{batch_idx}_t{text_idx}_s{sample_idx}",
                    ),
                )

                if dataset_name == "arctic" and obj_nfeats == 10:
                    angles_rad = proc_numpy(refined_x_obj_sampled[:, 9])
                    angles_deg = np.degrees(angles_rad)

                    key = (batch_idx, text_idx)
                    gen_angles_all.setdefault(key, []).append(angles_deg)

                    # --- Collect matching GT sequences (only once per key) ---
                    if key not in gt_angles_cache:
                        gt_angles_list = None
                        gt_idxs = np.array([], dtype=int)
                        if gt_data is not None:
                            from constants.arctic_constants import arctic_obj_name
                            text_str = text[batch_idx * 64 + text_idx].lower()
                            action_str = text_str.split()[0]
                            obj_name_gt = None
                            for _n in arctic_obj_name:
                                if _n in text_str:
                                    obj_name_gt = _n
                                    break
                            il_gt = int(is_lhand_text)
                            ir_gt = int(is_rhand_text)
                            if obj_name_gt is not None:
                                gt_mask = (
                                    (gt_data["object_name"] == obj_name_gt)
                                    & (gt_data["action_name"] == action_str)
                                    & (gt_data["is_lhand"] == il_gt)
                                    & (gt_data["is_rhand"] == ir_gt)
                                )
                                gt_idxs = np.where(gt_mask)[0]
                                if len(gt_idxs) > 0:
                                    gt_angles_list = [
                                        np.degrees(gt_data["x_obj_angle"][i, :gt_data["nframes"][i], 0])
                                        for i in gt_idxs
                                    ]
                        gt_angles_cache[key] = (gt_angles_list, gt_idxs)
                    else:
                        gt_angles_list, gt_idxs = gt_angles_cache[key]

                    # --- Angle comparison plot (updated with all samples so far) ---
                    save_angle_plot(
                        gen_angles_all[key],
                        save_path=osp.join(
                            result_folder,
                            "angle_plots",
                            f"angle_comparison_b{batch_idx}_t{text_idx}.png",
                        ),
                        gt_angles_list=gt_angles_list,
                    )

                    # --- Render representative GT sequence (only once per key) ---
                    if sample_idx == 0 and len(gt_idxs) > 0:
                        gt_nframes_list = [gt_data["nframes"][i] for i in gt_idxs]
                        med_i = gt_idxs[
                            np.argsort(gt_nframes_list)[len(gt_nframes_list) // 2]
                        ]
                        gt_nf = min(int(gt_data["nframes"][med_i]), max_nframes)

                        x_lhand_gt = proc_torch_cuda(
                            gt_data["x_lhand"][med_i, :gt_nf]
                        )
                        x_rhand_gt = proc_torch_cuda(
                            gt_data["x_rhand"][med_i, :gt_nf]
                        )
                        x_obj_gt = proc_torch_cuda(
                            np.concatenate([
                                gt_data["x_obj"][med_i, :gt_nf],
                                gt_data["x_obj_angle"][med_i, :gt_nf],
                            ], axis=-1)
                        )

                        gt_obj_verts_tf, gt_lhand_verts, gt_lhand_faces_r, \
                        gt_rhand_verts, gt_rhand_faces_r = \
                            proc_results(
                                x_lhand_gt, x_rhand_gt, x_obj_gt,
                                obj_verts_text, lhand_layer, rhand_layer,
                                is_lhand_text, is_rhand_text,
                                dataset_name, obj_top_idx_text,
                            )

                        if is_lhand_text and gt_lhand_verts is not None:
                            gt_lhand_verts[:, :, :2] -= \
                                gt_obj_verts_tf[0, :, :2].mean(0)[None, None]
                        if is_rhand_text and gt_rhand_verts is not None:
                            gt_rhand_verts[:, :, :2] -= \
                                gt_obj_verts_tf[0, :, :2].mean(0)[None, None]
                        gt_obj_verts_tf[:, :, :2] -= \
                            gt_obj_verts_tf[0, :, :2].mean(0)[None, None]

                        gt_video = render_videos(
                            renderer,
                            gt_lhand_verts, gt_lhand_faces_r,
                            gt_rhand_verts, gt_rhand_faces_r,
                            gt_obj_verts_tf, obj_faces_text,
                            is_lhand_text, is_rhand_text,
                        )
                        save_video(
                            gt_video, fps=fps,
                            save_path=osp.join(
                                result_folder,
                                "motion",
                                f"gt_b{batch_idx}_t{text_idx}.mp4",
                            ),
                        )
                        save_frames(
                            gt_video,
                            save_folder=osp.join(
                                result_folder,
                                "gt_frames",
                                f"gt_b{batch_idx}_t{text_idx}",
                            ),
                        )

                if save_obj:
                    save_mesh_obj(
                        refined_obj_verts_tf,
                        obj_faces_text, 
                        save_folder=osp.join(
                            result_folder, 
                            "obj_file", 
                            f"batch{batch_idx}_{text_idx}_sample{sample_idx}", 
                            "object", 
                        ),
                    )
                    if refined_lhand_verts is not None:
                        save_mesh_obj(
                            refined_lhand_verts,
                            lhand_faces, 
                            save_folder=osp.join(
                                result_folder, 
                                "obj_file", 
                                f"batch{batch_idx}_{text_idx}_sample{sample_idx}", 
                                "lhand", 
                            ),
                        )
                    if refined_rhand_verts is not None:
                        save_mesh_obj(
                            refined_rhand_verts,
                            rhand_faces, 
                            save_folder=osp.join(
                                result_folder, 
                                "obj_file", 
                                f"batch{batch_idx}_{text_idx}_sample{sample_idx}", 
                                "rhand", 
                            ),
                        )
    
    print(time.time() - start_time)

if __name__ == "__main__":
    main()