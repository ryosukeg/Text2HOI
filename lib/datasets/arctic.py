import time
import numpy as np
import json

from torch.utils.data import Dataset

from lib.models.object import build_object_model
from lib.utils.frame import get_valid_mask
from lib.utils.augm import (
    augmentation, 
    augmentation_joints, 
    get_augm_rot, 
    get_augm_scale, 
)
from lib.utils.proc_arctic import process_text
from lib.utils.proc import (
    get_contact_map, 
    pc_normalize, 
    process_dist_map, 
    select_from_groups, 
)


class SequenceARCTIC(Dataset): # point encoder
    def __init__(
        self, 
        data_path, 
        data_obj_pc_path, 
        text_json, 
        max_nframes, 
        data_ratio=1.0, 
        augm=False, 
        **kwargs
    ):
        super().__init__()

        self.data_path = data_path
        self.data_obj_pc_path = data_obj_pc_path
        self.max_nframes = max_nframes
        self.data_ratio = data_ratio
        self.augm = augm

        start_time = time.time()
        print("Start to read data arctic!!!")
        with np.load(data_path, allow_pickle=True) as data:
            self.object_name = data["object_name"]
            self.is_lhand = data["is_lhand"]
            self.is_rhand = data["is_rhand"]
            self.action_name = data["action_name"]
            self.nframes = data["nframes"]
        with open(text_json, "r") as f:
            self.text_description = json.load(f)

        self.object_model = build_object_model(data_obj_pc_path)

        print("Finish to read data arctic!!!", f"{time.time()-start_time:.2f}s")
        print(f"length of data: {self.__len__()}")

    def __len__(self):
        return int(len(self.nframes)*self.data_ratio)
    
    def __getitem__(self, index):
        item = {}
        
        nframes = self.nframes[index]
        if nframes > self.max_nframes:
            nframes = self.max_nframes
        seq_time = np.array([nframes/150], dtype=np.float32)
        if self.augm:
            augm_scale = 1-(2*np.random.rand()*0.05-0.05)
            seq_time *= augm_scale
            if seq_time > 1:
                seq_time = np.array([1.0], dtype=np.float32)
        item["seq_time"] = seq_time
            
        object_name = self.object_name[index]
        action_name = self.action_name[index]
        is_lhand = self.is_lhand[index]
        is_rhand = self.is_rhand[index]
        
        text = process_text(
            action_name, 
            object_name, 
            is_lhand, is_rhand,
            self.text_description, 
        )
        item["text"] = text
        return item
    
    
class ContactARCTIC(Dataset): # point encoder
    def __init__(
        self, 
        data_path, 
        data_obj_pc_path, 
        text_json, 
        max_nframes, 
        data_ratio=1.0, 
        augm=False, 
        **kwargs
    ):
        super().__init__()

        self.data_path = data_path
        self.data_obj_pc_path = data_obj_pc_path
        self.max_nframes = max_nframes
        self.data_ratio = data_ratio
        self.augm = augm

        start_time = time.time()
        print("Start to read data arctic!!!")
        with np.load(data_path, allow_pickle=True) as data:
            self.object_name = data["object_name"]
            self.lcov_idx = data["lcov_idx"] # left contact object verts idx
            self.rcov_idx = data["rcov_idx"] # right contact object verts idx
            self.is_lhand = data["is_lhand"]
            self.is_rhand = data["is_rhand"]
            self.action_name = data["action_name"]
        with open(text_json, "r") as f:
            self.text_description = json.load(f)

        self.object_model = build_object_model(data_obj_pc_path)

        print("Finish to read data arctic!!!", f"{time.time()-start_time:.2f}s")
        print(f"length of data: {self.__len__()}")

    def __len__(self):
        return int(len(self.action_name)*self.data_ratio)
    
    def __getitem__(self, index):
        item = {}
        
        is_lhand = self.is_lhand[index]
        is_rhand = self.is_rhand[index]
        item["is_lhand"] = is_lhand
        item["is_rhand"] = is_rhand
        
        action_name = self.action_name[index]
        object_name = self.object_name[index]
        item["action_name"] = action_name
        item["object_name"] = object_name
        
        text = process_text(
            action_name, 
            object_name, 
            is_lhand, is_rhand,
            self.text_description, 
        )
        item["text"] = text

        _, obj_pc, _, _, _ = self.object_model(object_name)

        if self.augm:
            aug_scale = get_augm_scale(0.2).numpy()
            obj_pc = obj_pc*aug_scale
            aug_rotmat = get_augm_rot(15, 15, 15).numpy()
            obj_pc = np.einsum("ij,kj->ki", aug_rotmat, obj_pc)
        normalized_obj_pc, _, obj_norm_scale = pc_normalize(obj_pc, return_params=True)
        item["normalized_obj_pc"] = normalized_obj_pc
        item["obj_scale"] = obj_norm_scale
        
        lcov_idx = self.lcov_idx[index]
        rcov_idx = self.rcov_idx[index]
        lcov_map = get_contact_map(lcov_idx, 1024, is_lhand)
        rcov_map = get_contact_map(rcov_idx, 1024, is_rhand)
        cov_map = (lcov_map+rcov_map)>0
        item["cov_map"] = cov_map.astype(np.float32)
        return item
    

class MotionARCTIC(Dataset):
    def __init__(
        self, 
        data_path, 
        data_obj_pc_path, 
        text_json, 
        max_nframes, 
        data_ratio=1.0, 
        augm=False, 
        use_angle_cond=False,
        angle_min_deg=0.0,
        angle_max_deg=90.0,
        angle_tol_deg=5.0,
        angle_condition_mode="first_reach",
        **kwargs
    ):
        super().__init__()

        self.data_path = data_path
        self.data_obj_pc_path = data_obj_pc_path
        self.max_nframes = max_nframes
        self.data_ratio = data_ratio
        self.augm = augm
        self.use_angle_cond = bool(use_angle_cond)
        self.angle_min_deg = float(angle_min_deg)
        self.angle_max_deg = float(angle_max_deg)
        self.angle_tol_deg = float(angle_tol_deg)
        self.angle_condition_mode = str(angle_condition_mode)

        # optional per-object / per-action filtering (comma-separated strings)
        _fo = kwargs.get("filter_objects", None)
        self._filter_objects = [s.strip() for s in _fo.split(",")] if _fo else None
        _fa = kwargs.get("filter_actions", None)
        self._filter_actions = [s.strip() for s in _fa.split(",")] if _fa else None

        start_time = time.time()
        print("Start to read data arctic!!!")
        with np.load(data_path, allow_pickle=True) as data:
            object_name_all = data["object_name"]
            action_name_all = data["action_name"]
            # build index mask
            idx_mask = np.ones(len(object_name_all), dtype=bool)
            if self._filter_objects:
                idx_mask &= np.isin(object_name_all, self._filter_objects)
            if self._filter_actions:
                idx_mask &= np.isin(action_name_all, self._filter_actions)
            sel = np.where(idx_mask)[0]
            print(f"Filter: objects={self._filter_objects}, actions={self._filter_actions} "
                  f"→ {len(sel)}/{len(object_name_all)} sequences kept")
            self.object_name = object_name_all[sel]
            self.x_lhand = data["x_lhand"][sel]
            self.x_rhand = data["x_rhand"][sel]
            self.x_obj = data["x_obj"][sel]
            self.x_obj_angle = data["x_obj_angle"][sel]
            self.lhand_org = data["lhand_org"][sel]
            self.rhand_org = data["rhand_org"][sel]
            self.lcf_idx = data["lcf_idx"][sel] # left hand contact frame idx
            self.lcov_idx = data["lcov_idx"][sel] # left contact object verts idx
            self.lchj_idx = data["lchj_idx"][sel] # left contact hand joints idx
            self.ldist_value = data["ldist_value"][sel]
            self.rcf_idx = data["rcf_idx"][sel] # right hand contact frame idx
            self.rcov_idx = data["rcov_idx"][sel] # right contact object verts idx
            self.rchj_idx = data["rchj_idx"][sel] # right contact hand joints idx
            self.rdist_value = data["rdist_value"][sel]
            self.is_lhand = data["is_lhand"][sel]
            self.is_rhand = data["is_rhand"][sel]
            self.action_name = action_name_all[sel]
            self.nframes = data["nframes"][sel]
        with open(text_json, "r") as f:
            self.text_description = json.load(f)

        self.object_model = build_object_model(data_obj_pc_path)

        print("Finish to read data arctic!!!", f"{time.time()-start_time:.2f}s")

    def __len__(self):
        return int(len(self.action_name)*self.data_ratio)
    
    def __getitem__(self, index):
        item = {}

        nframes = self.nframes[index]
        is_lhand = self.is_lhand[index]
        is_rhand = self.is_rhand[index]
        
        item["is_lhand"] = is_lhand
        item["is_rhand"] = is_rhand

        if self.use_angle_cond:
            orig_nframes = int(nframes)
            init_frame = 0
            angle_seq_rad = self.x_obj_angle[index][:orig_nframes, 0].astype(np.float32)
            tol_rad = float(np.deg2rad(self.angle_tol_deg))
            end_frame = None
            target_deg = None
            for _ in range(10):
                cand_deg = float(np.random.uniform(self.angle_min_deg, self.angle_max_deg))
                cand_rad = float(np.deg2rad(cand_deg))
                reach_idx = np.where(np.abs(angle_seq_rad - cand_rad) <= tol_rad)[0]
                if len(reach_idx) > 0:
                    target_deg = cand_deg
                    end_frame = int(reach_idx[0])
                    break
            if end_frame is None:
                cand_deg = float(np.random.uniform(self.angle_min_deg, self.angle_max_deg))
                cand_rad = float(np.deg2rad(cand_deg))
                if orig_nframes > 0:
                    end_frame = int(np.argmin(np.abs(angle_seq_rad - cand_rad)))
                else:
                    end_frame = 0
                target_deg = cand_deg
            cut_nframes = max(end_frame + 1, 1)
            # ---- stretch cut sequence to max_nframes so length is always fixed ----
            # This decouples sequence length from target angle.
            self._retime_cut = cut_nframes          # stored for use below
            nframes = self.max_nframes              # valid frames = full length after stretch
            denom = max(self.angle_max_deg - self.angle_min_deg, 1e-6)
            angle_norm = (target_deg - self.angle_min_deg) / denom
            item["target_angle_deg"] = np.float32(target_deg)
            item["angle_cond"] = np.array([angle_norm], dtype=np.float32)
        else:
            self._retime_cut = None
            if nframes > self.max_nframes:
                init_frame = np.random.randint(0, nframes-self.max_nframes)
                nframes = self.max_nframes
            else:
                init_frame = 0
            item["target_angle_deg"] = np.float32(-1.0)
            item["angle_cond"] = np.zeros(1, dtype=np.float32)
        
        item["nframes"] = nframes
        x_obj = self.x_obj[index][init_frame:init_frame+self.max_nframes]
        if self.augm:
            x_obj[:nframes], aug_rotmat, aug_trans = augmentation(x_obj[:nframes])
        x_obj_angle = self.x_obj_angle[index][init_frame:init_frame+self.max_nframes]
        x_obj = np.concatenate([x_obj, x_obj_angle], axis=1)

        # ---- retime: stretch cut portion to max_nframes ----
        if self._retime_cut is not None and self._retime_cut < self.max_nframes:
            cut = self._retime_cut
            t_src = np.linspace(0, cut - 1, cut)
            t_dst = np.linspace(0, cut - 1, self.max_nframes)
            x_obj = np.stack(
                [np.interp(t_dst, t_src, x_obj[:cut, d]) for d in range(x_obj.shape[1])],
                axis=1,
            ).astype(np.float32)

        item["x_obj"] = x_obj

        if is_lhand:
            x_lhand = self.x_lhand[index][init_frame:init_frame+self.max_nframes]
            if self.augm:
                lhand_org = self.lhand_org[index][init_frame:init_frame+nframes]
                x_lhand[:nframes], _, _ \
                    = augmentation(
                        x_lhand[:nframes], 
                        hand_org=lhand_org, 
                        aug_rotmat=aug_rotmat, 
                        aug_trans=aug_trans
                    )
            if self._retime_cut is not None and self._retime_cut < self.max_nframes:
                cut = self._retime_cut
                x_lhand = np.stack(
                    [np.interp(t_dst, t_src, x_lhand[:cut, d]) for d in range(x_lhand.shape[1])],
                    axis=1,
                ).astype(np.float32)
        else:
            x_lhand = np.zeros((150, 99), dtype=np.float32)
        
        item["x_lhand"] = x_lhand

        if is_rhand:
            x_rhand = self.x_rhand[index][init_frame:init_frame+self.max_nframes]
            if self.augm:
                rhand_org = self.rhand_org[index][init_frame:init_frame+nframes]
                x_rhand[:nframes], _, _ \
                    = augmentation(
                        x_rhand[:nframes], 
                        hand_org=rhand_org, 
                        aug_rotmat=aug_rotmat, 
                        aug_trans=aug_trans
                    )
            if self._retime_cut is not None and self._retime_cut < self.max_nframes:
                cut = self._retime_cut
                x_rhand = np.stack(
                    [np.interp(t_dst, t_src, x_rhand[:cut, d]) for d in range(x_rhand.shape[1])],
                    axis=1,
                ).astype(np.float32)
        else:
            x_rhand = np.zeros((150, 99), dtype=np.float32)
        item["x_rhand"] = x_rhand

        action_name = self.action_name[index]
        max_nframes = self.max_nframes
        
        valid_mask_lhand, valid_mask_rhand, valid_mask_obj = get_valid_mask(is_lhand, is_rhand, max_nframes, nframes) # max_nframes: 2x frames
        item["valid_mask_lhand"] = valid_mask_lhand
        item["valid_mask_rhand"] = valid_mask_rhand
        item["valid_mask_obj"] = valid_mask_obj

        object_name = self.object_name[index]
        item["object_name"] = object_name
        
        text = process_text(
            action_name, 
            object_name, 
            is_lhand, is_rhand,
            self.text_description, 
        )
        item["text"] = text
        
        _, obj_pc, obj_pc_normal, _, obj_pc_top_idx = self.object_model(object_name)
        
        normalized_obj_pc, obj_norm_cent, obj_norm_scale = pc_normalize(obj_pc, return_params=True)
        item["obj_pc_top_idx"] = obj_pc_top_idx
        item["obj_pc"] = obj_pc
        item["normalized_obj_pc"] = normalized_obj_pc
        item["obj_pc_normal"] = obj_pc_normal
        item["obj_cent"] = obj_norm_cent
        item["obj_scale"] = obj_norm_scale

        lcf_idx = self.lcf_idx[index] 
        lcov_idx = self.lcov_idx[index]
        lchj_idx = self.lchj_idx[index]
        ldist_value = self.ldist_value[index]
        ldist_map = process_dist_map(
            self.max_nframes, 
            init_frame, lcf_idx, 
            lcov_idx, lchj_idx, 
            ldist_value, is_lhand)
        item["ldist_map"] = ldist_map

        rcf_idx = self.rcf_idx[index]
        rcov_idx = self.rcov_idx[index]
        rchj_idx = self.rchj_idx[index]
        rdist_value = self.rdist_value[index]
        rdist_map = process_dist_map(
            self.max_nframes, 
            init_frame, rcf_idx, 
            rcov_idx, rchj_idx, 
            rdist_value, is_rhand)
        item["rdist_map"] = rdist_map
        
        lcov_map = get_contact_map(lcov_idx, 1024, is_lhand)
        rcov_map = get_contact_map(rcov_idx, 1024, is_rhand)
        cov_map = (lcov_map+rcov_map)>0
        item["cov_map"] = cov_map.astype(np.float32)
        return item