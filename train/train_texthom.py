import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))

import tqdm
import numpy as np
import hydra
from omegaconf import OmegaConf
from easydict import EasyDict as edict

import torch
import torch.optim as optim

from lib.models.mano import build_mano_aa, right_hand_mean, left_hand_mean
from lib.networks.clip import load_and_freeze_clip, encoded_text
from lib.datasets.datasets import get_dataloader
from lib.utils.model_utils import (
    build_model_and_diffusion, 
    build_seq_cvae, 
    build_pointnetfeat, 
    build_contact_estimator, 
)
from lib.models.object import build_object_model
from lib.utils.renderer import Renderer, Renderer_penet, Renderer_contact
from lib.utils.metric import AverageMeter
from lib.utils.file import (
    make_model_result_folder, 
    wandb_login, 
    save_video, 
)
from lib.utils.eval import (
    get_object_hand_info, 
    get_valid_mask_bunch, 
    proc_results, 
)
from lib.utils.proc import (
    proc_obj_feat_final, 
    proc_obj_feat_final_train, 
)
from lib.utils.visualize import render_videos


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_object(config)
    config = edict(config)
    data_config = config.dataset
    dataset_name = data_config.name
    
    wandb = wandb_login(config, config.texthom, relogin=False)

    save_demo = config.save_demo
    
    model_name = config.texthom.model_name
    best_model_name = model_name+"_best"
    save_root = config.texthom.save_root
    model_folder, result_folder = make_model_result_folder(save_root, train_type="train")
    
    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=data_config.flat_hand).cuda()
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=data_config.flat_hand).cuda()
    # ---- Method 2: inject angle-condition dataset kwargs (arctic only) ----
    use_angle_cond = bool(getattr(config.texthom, "use_angle_cond", False))
    if dataset_name == "arctic" and use_angle_cond:
        data_config.use_angle_cond = True
        data_config.angle_min_deg = float(config.texthom.angle_min_deg)
        data_config.angle_max_deg = float(config.texthom.angle_max_deg)
        data_config.angle_tol_deg = float(config.texthom.angle_tol_deg)
        data_config.angle_condition_mode = str(config.texthom.angle_condition_mode)
        print(f"[Method2] angle-cond training enabled: "
              f"[{data_config.angle_min_deg}, {data_config.angle_max_deg}] deg, "
              f"tol={data_config.angle_tol_deg}, mode={data_config.angle_condition_mode}")
    # ---- optional per-object / per-action filter (e.g. dataset.filter_objects=box) ----
    if hasattr(data_config, "filter_objects") and data_config.filter_objects:
        print(f"[Filter] objects={data_config.filter_objects}")
    if hasattr(data_config, "filter_actions") and data_config.filter_actions:
        print(f"[Filter] actions={data_config.filter_actions}")
    dataloader = get_dataloader("Motion"+dataset_name, config, data_config)
    texthom, diffusion \
        = build_model_and_diffusion(config, lhand_layer, rhand_layer)
    
    seq_cvae = build_seq_cvae(config, test=True)
    pointnet = build_pointnetfeat(config, test=True)
    object_model = build_object_model(data_config.data_obj_pc_path)
    dump_data = torch.randn([1, 1024, 3]).cuda()
    pointnet(dump_data)
    contact_estimator = build_contact_estimator(config, test=True)
    optimizer = optim.AdamW(texthom.parameters(), lr=config.texthom.lr)
    
    clip_model = load_and_freeze_clip(config.clip.clip_version)
    clip_model = clip_model.cuda()
    
    renderer = Renderer(device="cuda", camera=f"{dataset_name}_front")
    
    cur_loss = 9999
    best_loss = 9999
    best_epoch = 0
    
    lambda_simple = config.texthom.lambda_simple
    lambda_dist = config.texthom.lambda_dist
    lambda_ro = config.texthom.lambda_ro
    lambda_angle = float(getattr(config.texthom, "lambda_angle", 0.0))
    loss_lambda_dict = {
        "lambda_simple": lambda_simple, 
        "lambda_dist": lambda_dist, 
        "lambda_ro": lambda_ro, 
        "lambda_angle": lambda_angle, 
    }
    hand_nfeats = config.texthom.hand_nfeats
    obj_nfeats = config.texthom.obj_nfeats
    
    nepoch = config.texthom.iteration / (data_config.data_num/data_config.text_num)
    nepoch = int(np.ceil(nepoch / 50.0) * 50)
    with tqdm.tqdm(range(nepoch)) as pbar:
        for epoch in pbar:
            texthom.train()
            loss_meter = AverageMeter()
            loss_simple_meter = AverageMeter()
            loss_dist_meter = AverageMeter()
            loss_rot_meter = AverageMeter()
            loss_angle_meter = AverageMeter()
            target_angle_meter = AverageMeter()
            for item in dataloader:
                if dataset_name == "arctic":
                    obj_pc_top_idx = item["obj_pc_top_idx"].cuda()
                else:
                    obj_pc_top_idx = None
                x_lhand = item["x_lhand"].cuda()
                x_rhand = item["x_rhand"].cuda()
                x_obj = item["x_obj"].cuda()
                obj_pc_org = item["obj_pc"].cuda()
                normalized_obj_pc = item["normalized_obj_pc"].cuda()
                obj_scale = item["obj_scale"].cuda()
                obj_cent = item["obj_cent"].cuda()
                ldist_map = item["ldist_map"].cuda()
                rdist_map = item["rdist_map"].cuda()
                cov_map = item["cov_map"].cuda()
                
                valid_mask_lhand = item["valid_mask_lhand"].cuda()
                valid_mask_rhand = item["valid_mask_rhand"].cuda()
                valid_mask_obj = item["valid_mask_obj"].cuda()

                max_valid_nframes = int(valid_mask_obj.sum(dim=1).max().item())
                if max_valid_nframes < x_obj.shape[1]:
                    x_lhand = x_lhand[:, :max_valid_nframes]
                    x_rhand = x_rhand[:, :max_valid_nframes]
                    x_obj = x_obj[:, :max_valid_nframes]
                    ldist_map = ldist_map[:, :max_valid_nframes]
                    rdist_map = rdist_map[:, :max_valid_nframes]
                    valid_mask_lhand = valid_mask_lhand[:, :max_valid_nframes]
                    valid_mask_rhand = valid_mask_rhand[:, :max_valid_nframes]
                    valid_mask_obj = valid_mask_obj[:, :max_valid_nframes]

                text = item["text"]
                enc_text = encoded_text(clip_model, text)

                bs = x_obj.shape[0]

                angle_cond_batch = None
                if use_angle_cond and "angle_cond" in item:
                    angle_cond_batch = item["angle_cond"].cuda().float()
                    if "target_angle_deg" in item:
                        target_angle_meter.update(
                            float(item["target_angle_deg"].float().mean().item()), bs
                        )

                with torch.no_grad():
                    obj_feat = pointnet(normalized_obj_pc)
                obj_feat_final = proc_obj_feat_final_train(
                    cov_map, obj_scale, obj_cent, obj_feat, 
                    config.texthom.use_obj_scale_centroid, 
                    config.texthom.use_contact_feat, 
                )
                _, _, _, losses_dict \
                    = diffusion(
                        texthom, x_lhand, x_rhand, 
                        x_obj, obj_feat_final, 
                        enc_text=enc_text, get_losses=True, 
                        valid_mask_lhand=valid_mask_lhand, 
                        valid_mask_rhand=valid_mask_rhand, 
                        valid_mask_obj=valid_mask_obj, 
                        ldist_map=ldist_map, 
                        rdist_map=rdist_map, 
                        obj_verts_org=obj_pc_org, 
                        loss_lambda_dict=loss_lambda_dict, 
                        dataset_name=dataset_name,
                        obj_pc_top_idx=obj_pc_top_idx,
                        angle_cond=angle_cond_batch,
                    )
                simple_loss = losses_dict["simple_loss"]
                dist_map_loss = losses_dict["dist_map_loss"]
                ro_loss = losses_dict["ro_loss"]
                angle_loss = losses_dict.get("angle_loss", torch.zeros((), device=simple_loss.device))
                
                losses = lambda_simple*simple_loss \
                       + lambda_dist*dist_map_loss \
                       + lambda_ro*ro_loss \
                       + lambda_angle*angle_loss

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                loss_meter.update(losses.item(), bs)
                loss_simple_meter.update(simple_loss.item(), bs)
                loss_dist_meter.update(dist_map_loss.item(), bs)
                loss_rot_meter.update(ro_loss.item(), bs)
                loss_angle_meter.update(float(angle_loss.item()), bs)
            
            cur_loss = loss_meter.avg
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_epoch = epoch+1
                torch.save(
                    {
                        "model": texthom.state_dict(), 
                        "epoch": epoch, 
                        "loss": cur_loss, 
                    },
                    osp.join(model_folder, best_model_name+".pth"), 
                )
            wandb.log(
                {
                    "loss": cur_loss,
                    "simple_loss": loss_simple_meter.avg,
                    "dist_map_loss": loss_dist_meter.avg,
                    "ro_loss": loss_rot_meter.avg,
                    "angle_loss": loss_angle_meter.avg,
                    **({"target_angle_deg": target_angle_meter.avg} if use_angle_cond else {}),
                }
            )
            pbar.set_description(f"{model_name} | Best loss: {best_loss:.4f} ({best_epoch}), Cur loss: {cur_loss:.4f}"
                + (f", angle_loss: {loss_angle_meter.avg:.4f}" if lambda_angle > 0 else "")
                + (f", angle_avg: {target_angle_meter.avg:.2f}" if use_angle_cond else ""))
            
            if (epoch+1)%config.save_pth_freq==0:
                torch.save(
                    {
                        "model": texthom.state_dict(), 
                        "epoch": epoch, 
                        "loss": cur_loss, 
                    },
                    osp.join(model_folder, model_name+f"_{epoch+1}.pth"), 
                )
            
            if dataset_name == "h2o":
                texthom.eval()
                if save_demo and (epoch+1)%config.result_show_freq==0:
                    with torch.no_grad():
                        max_nframes = data_config.max_nframes
                        text = config.text
                        
                        is_lhand, is_rhand, \
                        obj_pc_org, _, \
                        normlized_obj_pc, point_sets, \
                        obj_cent, obj_scale, \
                        obj_verts, obj_faces, \
                        obj_top_idx, _ \
                            = get_object_hand_info(
                                object_model, clip_model, text, 
                                data_config.obj_root, data_config
                            )
                        bs, npts = obj_pc_org.shape[:2]
                        
                        enc_text = encoded_text(clip_model, text)
                        
                        obj_feat = pointnet(normlized_obj_pc)
                        obj_feat_final, est_contact_map = proc_obj_feat_final(
                            contact_estimator,  
                            obj_scale, obj_cent, 
                            obj_feat, enc_text, npts, 
                            config.texthom.use_obj_scale_centroid, config.contact.use_scale, config.texthom.use_contact_feat
                        )
                        duration = seq_cvae.decode(enc_text)
                        duration *= 150
                        duration = duration.long()
                        valid_mask_lhand, valid_mask_rhand, valid_mask_obj \
                                = get_valid_mask_bunch(is_lhand, is_rhand, max_nframes, duration)
                        coarse_x_lhand, coarse_x_rhand, coarse_x_obj \
                            = diffusion.sampling(
                                texthom, obj_feat_final, 
                                enc_text, max_nframes, 
                                hand_nfeats, obj_nfeats, 
                                valid_mask_lhand, 
                                valid_mask_rhand, 
                                valid_mask_obj, 
                                device=torch.device("cuda")
                            )
                        
                        for text_idx in range(len(text)):
                            text_duration = duration[text_idx].item()
                            obj_verts_text = obj_verts[text_idx]
                            obj_faces_text = obj_faces[text_idx]
                            point_set_text = point_sets[text_idx]
                            est_contact_map_text = est_contact_map[text_idx]
                            is_lhand_text = is_lhand[text_idx]
                            is_rhand_text = is_rhand[text_idx]
                            if dataset_name == "arctic":
                                obj_top_idx_text = obj_top_idx[text_idx]
                            else:
                                obj_top_idx_text = None
                            
                            coarse_x_lhand_sampled = coarse_x_lhand[text_idx][:text_duration]
                            coarse_x_rhand_sampled = coarse_x_rhand[text_idx][:text_duration]
                            coarse_x_obj_sampled = coarse_x_obj[text_idx][:text_duration]
                            
                            coarse_obj_verts_tf, coarse_lhand_verts, lhand_faces, \
                            coarse_rhand_verts, rhand_faces = \
                                proc_results(
                                    coarse_x_lhand_sampled, coarse_x_rhand_sampled, coarse_x_obj_sampled, 
                                    obj_verts_text, lhand_layer, rhand_layer, 
                                    is_lhand_text, is_rhand_text, 
                                    dataset_name, obj_top_idx_text, 
                                )
                                
                            merged_video = render_videos(
                                renderer, 
                                coarse_lhand_verts, lhand_faces, 
                                coarse_rhand_verts, rhand_faces, 
                                coarse_obj_verts_tf, obj_faces_text, 
                                is_lhand_text, is_rhand_text, 
                            )
                            save_video(
                                merged_video, fps=30, 
                                save_path=osp.join(
                                    result_folder, 
                                    f"{text[text_idx]}_{epoch+1}.mp4"
                                )
                            )
                        
                        # Color, Depth video
                        # colors = renderer.render_video(
                        #     vertices=vis_vertices, 
                        #     faces=vis_faces, 
                        #     mesh_kind=mesh_kind,
                        # )
                        # colors_tr = colors.transpose(0, 3, 1, 2)*255 # to T, C, H, W shape
                        # colors_tr = colors_tr.astype(np.uint8)
                        # colors_tr = colors_tr[:, :3]
                        # wandb.log({
                        #     "video_fps4": wandb.Video(colors_tr, fps=4, format="mp4")
                        # })
                        # wandb.log({
                        #     "video_fps30": wandb.Video(colors_tr, fps=30, format="mp4")
                        # })

    torch.save(
        {
            "model": texthom.state_dict(), 
            "epoch": nepoch, 
            "loss": cur_loss, 
        },
        osp.join(model_folder, model_name+".pth"), 
    )
    wandb.finish()


if __name__ == "__main__":
    main()