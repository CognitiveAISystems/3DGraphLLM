import logging
import os
import random
from torch.utils.data import Dataset
import torch
import glob
import numpy as np
import einops
import tqdm
import matplotlib.pyplot as plt
from collections import Counter

MAX_OBJ_NUM = 150
KNN = 2
NEIGHBOR_SHIFT = 0
MINIMUM_DISTANCE = 0.01 # cm
IOU_THRESHOLD = 0.99
FEATS_EDGE_DIR = "../annotations/output_vlsat"
SEGMENTOR = "mask3d"

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
    x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_box3d_min_max(corners2)
    xA = np.maximum(x_min_1, x_min_2)
    yA = np.maximum(y_min_1, y_min_2)
    zA = np.maximum(z_min_1, z_min_2)
    xB = np.minimum(x_max_1, x_max_2)
    yB = np.minimum(y_max_1, y_max_2)
    zB = np.minimum(z_max_1, z_max_2)
    inter_vol = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)

    return iou


topk_values_distr = []
topk_cats_distr = []
number_of_zero_distances = 0

for split in ["val", "train"]:
    with open(f"../scannet_{split}_scans.txt", "r") as f:
        scenes = f.readlines()
    
    if SEGMENTOR == "mask3d":
        attributes = torch.load(f"../annotations/scannet_mask3d_{split}_attributes_colors.pt")
    elif SEGMENTOR == "oneformer3d":
        attributes = torch.load(f"../annotations/oneformer3d/scannet_oneformer3d_{split}_attributes_1e-1.pt")

    gnn_dict = {}
    for scene in tqdm.tqdm(scenes):
        scene_id = scene.strip()
        #scene_id = "scene0435_00"
        try:
            scene_attr = attributes[scene_id]
        except:
            continue
        filtered_objects = []
        
        ious = []
        # Compare each object with every other object in the list
        for _i, obj1 in enumerate(scene_attr["locs"]):
            keep = True
            for _j, obj2 in enumerate(scene_attr["locs"]):
                if _i < _j:
                    box1 = construct_bbox_corners(obj1.tolist()[:3], obj1.tolist()[3:])
                    box2 = construct_bbox_corners(obj2.tolist()[:3], obj2.tolist()[3:])
                    iou = box3d_iou(box1, box2)
                    ious.append(iou)
                    if iou > IOU_THRESHOLD:
                        keep = False
                        break
            if keep:
                filtered_objects.append(_i)

        filtered_idx_to_idx = {
            i: filtered_i
            for i, filtered_i in enumerate(filtered_objects)
        }
        #scene_attr["locs"] = scene_attr["locs"][filtered_objects, :]
        #scene_attr["colors"] = scene_attr["colors"][filtered_objects, :]
        #scene_attr["objects"] = [obj for i, obj in enumerate(scene_attr["objects"]) if i in filtered_objects]

        obj_num = scene_attr["locs"].shape[0]
        if obj_num > MAX_OBJ_NUM:
            obj_num = MAX_OBJ_NUM
        scene_locs = scene_attr["locs"][:obj_num,...]
        scene_colors = scene_attr["colors"][:obj_num,...]
        obj_ids = scene_attr["obj_ids"] if "obj_ids" in scene_attr else [_i for _i in range(obj_num)]

        gnn_shape = 512
        
        gnn_feat = []

        object_indices = filtered_objects
        pairwise_locs = einops.repeat(scene_locs[:, :3], 'l d -> l 1 d') \
                        - einops.repeat(scene_locs[object_indices, :3], 'l d -> 1 l d')
        pairwise_dists = torch.sqrt(torch.sum(pairwise_locs ** 2, 2) + 1e-10)
        # mask small pairwise distances with large values 
        pairwise_dists[pairwise_dists < MINIMUM_DISTANCE] = 100.0
        if len(object_indices)>KNN:
            topk_values, topk_indices = torch.topk(pairwise_dists, KNN+NEIGHBOR_SHIFT, dim=1,  largest=False)
        else:
            print(len(object_indices), len(obj_ids))
            topk_values, topk_indices = torch.topk(pairwise_dists, len(object_indices), dim=1,  largest=False)

        if FEATS_EDGE_DIR is not None:
            vlsat_features = torch.load(os.path.join(FEATS_EDGE_DIR, scene_id + ".pt"))
            #print(list(vlsat_features.keys())[:5])
            for _i, _id1 in enumerate(obj_ids):
                if _i > obj_num:
                    continue
                for nn in range(min(KNN, len(object_indices)-1)):
                    _j = object_indices[topk_indices[_i, nn+NEIGHBOR_SHIFT]]
                    value = topk_values[_i,nn+NEIGHBOR_SHIFT]
                    topk_values_distr.append(value)
                    if value < 0.001:
                        number_of_zero_distances += 1
                    topk_cats_distr.append(scene_attr['objects'][_j])

                    #item_id = f'{scene_id}\n_{filtered_idx_to_idx[_i]}_{filtered_idx_to_idx[_j]}'
                    item_id = f'{scene_id}\n_{_i}_{_j}'
                    if item_id not in vlsat_features:  # i==j       
                        gnn_feat.append(None)
                    else:
                        gnn_feat.append(vlsat_features[item_id])
                        gnn_shape = gnn_feat[-1].shape[0]

            for _i in range(len(gnn_feat)):
                if gnn_feat[_i] is None:
                    gnn_feat[_i] = torch.zeros(gnn_shape)
        else:
            for _i, _id1 in enumerate(obj_ids):
                if _i > obj_num:
                    continue
                for nn in range(min(KNN, len(object_indices)-1)):
                    _j = object_indices[topk_indices[_i,nn+NEIGHBOR_SHIFT]]
                    value = topk_values[_i,nn+NEIGHBOR_SHIFT]
                    topk_values_distr.append(value)
                    if value < 0.001:
                        number_of_zero_distances += 1
                    topk_cats_distr.append(scene_attr['objects'][_j])
                    #item_id = f'{scene_id}\n_{_i}_{_j}'  #scannet
                    item_id = f'{scene}_{_i}_{_j}'
                    if item_id not in FEATS_EDGE:  # i==j
                        print(item_id)
                        gnn_feat.append(None)

                    else:
                        gnn_feat.append(FEATS_EDGE[item_id])
                        gnn_shape = gnn_feat[-1].shape[0]
            for _i in range(len(gnn_feat)):
                if gnn_feat[_i] is None:
                    gnn_feat[_i] = torch.zeros(gnn_shape)
        gnn_feat = torch.stack(gnn_feat, dim=0)
        gnn_dict[scene_id] = torch.clone(gnn_feat)

    cats = Counter(topk_cats_distr)
    for key, value in cats.items():
        print(key, value)
    plt.hist(topk_values_distr, bins=50, edgecolor='black')
    plt.title('Histogram of Values')
    plt.xlabel('Distance to nearest neighbor')
    plt.ylabel('Frequency')
    plt.savefig(f'histogram_nearest_neighbors_distr_1_0_25_dist_{split}_mask3d_filtered.png')
    print("Fraction of zero distances", number_of_zero_distances/len(topk_values_distr))
    print("Min of distances between neighbors", min(topk_values_distr))
    print("5 quantile of distances", np.quantile(topk_values_distr, 0.05))
    #exit()
    torch.save(gnn_dict, f"../annotations/scannet_{SEGMENTOR}_{split}_gnn_feats_{KNN}.pt")