import logging
import os
import random
from torch.utils.data import Dataset
import torch
import glob
from torch.nn.utils.rnn import pad_sequence
import re
import numpy as np
import tqdm

logger = logging.getLogger(__name__)
IOU_THRESHOLD = 0.99

class BaseDataset(Dataset):

    def __init__(self):
        self.media_type = "point_cloud"
        self.anno = None
        self.attributes = None
        self.feats = None
        self.feats_edge = None
        self.img_feats = None
        self.scene_feats = None
        self.scene_img_feats = None
        self.scene_masks = None
        self.scene_foreground_ids = None
        self.feat_dim = 1024
        self.img_feat_dim = 1024
        self.max_obj_num = 100
        self.knn = 0
        self.point_cloud_type = None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
    def prepare_scene_features(self):
        if self.feats is not None:
            scan_ids = set('_'.join(x.split('_')[:2]) for x in self.feats.keys())
        else:
            scan_ids = set('_'.join(x.split('_')[:2]) for x in self.img_feats.keys())
        scene_feats = {}
        scene_img_feats = {}
        scene_masks = {}
        scene_gnn_feats = {}
        scene_foreground_ids = {}
        unwanted_words = ["wall", "ceiling", "floor", "object", "item"]
        for scan_id in tqdm.tqdm(scan_ids):
            #if scan_id != "scene0435_00":
            #    continue
            if scan_id not in self.attributes:
                continue
            scene_attr = self.attributes[scan_id]
            obj_num = scene_attr['locs'].shape[0]
            #obj_num = self.max_obj_num
            obj_ids = scene_attr['obj_ids'] if 'obj_ids' in scene_attr else [_ for _ in range(obj_num)]
            obj_labels = scene_attr['objects'] if 'objects' in scene_attr else [''] * obj_num
            #for i in range(len(obj_ids)):
            #    print(i, scene_attr["locs"][i])
            #exit()
            scene_feat = []
            scene_img_feat = []
            scene_mask = []
            for _i, _id in enumerate(obj_ids):
                if scan_id == 'scene0217_00':  # !!!!
                    _id += 31
                item_id = '_'.join([scan_id, f'{_id:02}'])
                if self.feats is None or item_id not in self.feats:
                    # scene_feat.append(torch.randn((self.feat_dim)))
                    scene_feat.append(torch.zeros(self.feat_dim))
                else:
                    scene_feat.append(self.feats[item_id])
                if self.img_feats is None or item_id not in self.img_feats:
                    # scene_img_feat.append(torch.randn((self.img_feat_dim)))
                    scene_img_feat.append(torch.zeros(self.img_feat_dim))
                else:
                    scene_img_feat.append(self.img_feats[item_id].float())
                # if scene_feat[-1] is None or any(x in obj_labels[_id] for x in unwanted_words):
                #     scene_mask.append(0)
                # else:
                scene_mask.append(1)
            filtered_objects = []

            if self.point_cloud_type == "gt":
                scene_foreground_ids[scan_id] = torch.LongTensor(obj_ids)
            else:
                # Compare each object with every other object in the list
                for _i, obj1 in enumerate(scene_attr["locs"]):
                    keep = True
                    for _j, obj2 in enumerate(scene_attr["locs"]):
                        if _i < _j:
                            box1 = construct_bbox_corners(obj1.tolist()[:3], obj1.tolist()[3:])
                            box2 = construct_bbox_corners(obj2.tolist()[:3], obj2.tolist()[3:])
                            iou = box3d_iou(box1, box2)

                            if iou > IOU_THRESHOLD:
                                keep = False
                                break
                    if keep:
                        filtered_objects.append(_i)
                scene_foreground_ids[scan_id] = torch.LongTensor(filtered_objects)
            scene_feat = torch.stack(scene_feat, dim=0)
            scene_img_feat = torch.stack(scene_img_feat, dim=0)
            scene_mask = torch.tensor(scene_mask, dtype=torch.int)
            scene_feats[scan_id] = scene_feat
            scene_img_feats[scan_id] = scene_img_feat
            scene_masks[scan_id] = scene_mask
            

            gnn_shape = 512
            gnn_feat = []
                
            if scan_id in self.feats_edge:
                gnn_feat = self.feats_edge[scan_id]
            else:
                gnn_feat = torch.zeros((len(obj_ids)*self.knn, gnn_shape))
            scene_gnn_feats[scan_id] = gnn_feat

        return scene_feats, scene_img_feats, scene_masks, scene_gnn_feats, scene_foreground_ids

    def get_anno(self, index):
        
        scene_id = self.anno[index]["scene_id"]
        #scene_id = "scene0435_00"
        if self.attributes is not None:
            scene_attr = self.attributes[scene_id]
            # obj_num = scene_attr["locs"].shape[0]
            scene_locs = scene_attr["locs"]
        else:
            scene_locs = torch.randn((1, 6))
        scene_feat = self.scene_feats[scene_id]
        if scene_feat.ndim == 1:
            scene_feat = scene_feat.unsqueeze(0)
        scene_img_feat = self.scene_img_feats[scene_id] if self.scene_img_feats is not None else torch.zeros((scene_feat.shape[0], self.img_feat_dim))
        scene_mask = self.scene_masks[scene_id] if self.scene_masks is not None else torch.ones(scene_feat.shape[0], dtype=torch.int)
        # assigned_ids = torch.randperm(self.max_obj_num)[:len(scene_locs)]
        assigned_ids = torch.randperm(len(scene_locs))
        # assigned_ids = torch.randperm(self.max_obj_num) # !!!
        scene_gnn_feat = self.scene_gnn_feats[scene_id]
        scene_foreground_ids = self.scene_foreground_ids[scene_id]
        return scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, assigned_ids, scene_gnn_feat, scene_foreground_ids
    

def update_caption(caption, assigned_ids):
    new_ids = {int(assigned_id): i for i, assigned_id in enumerate(assigned_ids)}
    id_format = "<OBJ\\d{3}>"
    for match in re.finditer(id_format, caption):
        idx = match.start()
        old_id = int(caption[idx+4:idx+7])
        new_id = int(new_ids[old_id])
        caption = caption[:idx+4] + f"{new_id:03}" + caption[idx+7:]
    return caption


def recover_caption(caption, assigned_ids):
    id_format = "<OBJ\\d{3}>"
    for match in re.finditer(id_format, caption):
        idx = match.start()
        new_id = int(caption[idx+4:idx+7])
        try:
            old_id = int(assigned_ids[new_id])
        except:
            old_id = random.randint(0, len(assigned_ids)-1)
        caption = caption[:idx+4] + f"{old_id:03}" + caption[idx+7:]
    return caption


if __name__ == "__main__":
    caption = "<OBJ001> <OBJ002>"
    assigned_ids = torch.randperm(5)
    print(assigned_ids)
    caption = update_caption(caption, assigned_ids)
    print(caption)
    caption = recover_caption(caption, assigned_ids)
    print(caption)

def get_box3d_min_max(corner):
    ''' Compute min and max coordinates for 3D bounding box
        Note: only for axis-aligned bounding boxes

    Input:
        corners: numpy array (8,3), assume up direction is Z (batch of N samples)
    Output:
        box_min_max: an array for min and max coordinates of 3D bounding box IoU

    '''

    min_coord = corner.min(axis=0)
    max_coord = corner.max(axis=0)
    x_min, x_max = min_coord[0], max_coord[0]
    y_min, y_max = min_coord[1], max_coord[1]
    z_min, z_max = min_coord[2], max_coord[2]

    return x_min, x_max, y_min, y_max, z_min, z_max


def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is Z
        corners2: numpy array (8,3), assume up direction is Z
    Output:
        iou: 3D bounding box IoU

    '''

    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = get_box3d_min_max(corners1)
    #print(x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1)
    x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_box3d_min_max(corners2)
    #print(x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2)
    xA = np.maximum(x_min_1, x_min_2)
    yA = np.maximum(y_min_1, y_min_2)
    zA = np.maximum(z_min_1, z_min_2)
    xB = np.minimum(x_max_1, x_max_2)
    yB = np.minimum(y_max_1, y_max_2)
    zB = np.minimum(z_max_1, z_max_2)
    inter_vol = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    #print("box_vol_1", box_vol_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    #print("box_vol_2", box_vol_2)
    #print("inter_vol", inter_vol)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)
    #print("iou", iou)

    return iou


def construct_bbox_corners(center, box_size):
    sx, sy, sz = box_size
    x_corners = [sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2]
    y_corners = [sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2]
    z_corners = [sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)

    return corners_3d