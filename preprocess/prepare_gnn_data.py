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
KNN = 5
NEIGHBOR_SHIFT = 0
MINIMUM_DISTANCE = 0.01 # cm
SEGMENTOR = "gt"


topk_values_distr = []
topk_cats_distr = []
number_of_zero_distances = 0

for split in ["train"]:
    with open(f"../scannet_{split}_scans.txt", "r") as f:
        scenes = f.readlines()

    if SEGMENTOR == "mask3d":
        attributes = torch.load(f"../annotations/scannet_mask3d_{split}_attributes.pt")
    elif SEGMENTOR == "oneformer3d":
        attributes = torch.load(f"../annotations/oneformer3d/scannet_oneformer3d_{split}_attributes_1e-1.pt")
    else:
        attributes = torch.load(f"../annotations/scannet_gt_{split}_attributes.pt")
        FEATS_EDGE_DIR = None
        FEATS_EDGE = torch.load(f"../annotations/scannet_gnn_{split}_feats.pt")
    
    gnn_dict = {}
    for scene in tqdm.tqdm(scenes):
        scene_id = scene.strip()

        try:
            scene_attr = attributes[scene_id]
        except:
            continue

        obj_num = scene_attr["locs"].shape[0]
        if obj_num > MAX_OBJ_NUM:
            obj_num = MAX_OBJ_NUM
        scene_locs = scene_attr["locs"][:obj_num,...]
        scene_colors = scene_attr["colors"][:obj_num,...]
        obj_ids = scene_attr["obj_ids"] if "obj_ids" in scene_attr else [_i for _i in range(obj_num)]

        gnn_shape = 512
        
        gnn_feat = []

        object_indices = [
            i 
            for i in range(obj_num)
            #if not scene_attr['objects'][i] in ['wall', 'floor', 'ceiling']
        ]
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
                    item_id = f'{scene_id}\n_{_i}_{_j}'  #scannet
                    #item_id = f'{scene}_{_i}_{_j}'
                    if item_id not in FEATS_EDGE:  # i==j
                        print(item_id)
                        gnn_feat.append(None)

                    else:
                        gnn_feat.append(FEATS_EDGE[item_id])
                        gnn_shape = gnn_feat[-1].shape[0]
            for _i in range(len(gnn_feat)):
                if gnn_feat[_i] is None:
                    gnn_feat[_i] = torch.zeros(gnn_shape)
        try:
            gnn_feat = torch.stack(gnn_feat, dim=0)
            #print(gnn_feat)
            gnn_dict[scene_id] = torch.clone(gnn_feat)
        except:
            gnn_dict[scene_id] = torch.zeros(gnn_shape)
        #exit()
    cats = Counter(topk_cats_distr)
    for key, value in cats.items():
        print(key, value)
    plt.hist(topk_values_distr, bins=50, edgecolor='black')
    plt.title('Histogram of Values')
    plt.xlabel('Distance to nearest neighbor')
    plt.ylabel('Frequency')
    plt.savefig(f'histogram_nearest_neighbors_distr_1_0_25_dist_{split}_oneformer3d.png')
    print("Fraction of zero distances", number_of_zero_distances/len(topk_values_distr))
    print("Min of distances between neighbors", min(topk_values_distr))
    print("5 quantile of distances", np.quantile(topk_values_distr, 0.05))
    #exit()
    torch.save(gnn_dict, f"../annotations/scannet_{SEGMENTOR}_{split}_gnn_feats_{KNN}.pt")