import copy
import torch
import tqdm

import numpy as np
from torch.nn.utils.rnn import pad_sequence

from utils.config import Config
from models.graph3dllm import Graph3DLLM

IOU_THRESHOLD = 0.99

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

class Model_3DGraphLLM:
    def __init__(self, config, pretrained_path):
        self.config = copy.deepcopy(config)
        self.config.pretrained_path = pretrained_path
        self.model = Graph3DLLM(config=config)
        self.model = self.model.to(torch.device(self.config.device))

        checkpoint = torch.load(self.config.pretrained_path, map_location="cpu")
        state_dict = checkpoint["model"]
        keys_to_delete = []
        for name, param  in state_dict.items():
            if name not in self.model.state_dict():
                continue
            if param.size() != self.model.state_dict()[name].size():
                keys_to_delete.append(name)
        for key in keys_to_delete:
            del state_dict[key]
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(msg)
        print(f"Loaded checkpoint from {self.config.pretrained_path}.")
    
    def process(self, scene_feat, scene_img_feat, scene_locs, scene_mask, user_query, obj_ids, assigned_ids, scene_gnn_feats, foreground_ids):
        """
        Process the input features and query to return a response.
        
        Parameters:
            point_cloud_features (np.array): Point cloud features (e.g., shape [num_points, feature_dim])
            image_features (np.array): Image features (e.g., shape [num_images, feature_dim])
            user_query (str): Text query from the user.
            
        Returns:
            str: Response from the model.
        """
        # Example: In a real model, you'd process features and query here.
        print(f"Processing query: '{user_query}'.")
        
        response = self.model.evaluate(
            scene_feat,
            scene_img_feat,
            scene_locs,
            scene_mask,
            [user_query],
            obj_ids,
            assigned_ids,
            scene_gnn_feats,
            foreground_ids
        )
        # You'd return the model's real prediction here
        return response

    def prepare_scene_features(self, scene_id):
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
            if scan_id != scene_id:
                continue
            if scan_id not in self.attributes:
                continue
            scene_attr = self.attributes[scan_id]
            # obj_num = scene_attr['locs'].shape[0]
            obj_num = self.max_obj_num
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

    def get_anno(self, scene_id):
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
        # assigned_ids = torch.randperm(len(scene_locs))
        assigned_ids = torch.randperm(self.max_obj_num) # !!!
        scene_gnn_feat = self.scene_gnn_feats[scene_id]
        scene_foreground_ids = self.scene_foreground_ids[scene_id]
        return scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, assigned_ids, scene_gnn_feat, scene_foreground_ids
    
def main(config):
    # Define scene and load its features
    scene_id = 'scene0435_00'  # Example scene identifier
    pretrained_path = "outputs/3dgraphllm/ckpt_01_51426.pth"
    # Initialize the model
    model = Model_3DGraphLLM(config, pretrained_path)
    model.dataset_name = list(model.config.val_file_dict.keys())[0]
    model.feat_dim = model.config.model.input_dim
    model.img_feat_dim = model.config.model.img_input_dim
    model.max_obj_num = model.config.model.max_obj_num
    model.knn = model.config.model.knn
    model.point_cloud_type = model.config.val_file_dict[model.dataset_name][4]

    feat_file, img_feat_file, attribute_file, feats_gnn_file = model.config.val_file_dict[model.dataset_name][:4]
    model.attributes = torch.load(attribute_file, map_location='cpu') if attribute_file is not None else None
    model.feats = torch.load(feat_file, map_location='cpu')
    model.img_feats = torch.load(img_feat_file, map_location='cpu')
    model.feats_edge = torch.load(feats_gnn_file, map_location='cpu')
    model.scene_feats, model.scene_img_feats, model.scene_masks, model.scene_gnn_feats, model.scene_foreground_ids = model.prepare_scene_features(scene_id)
    
    scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, assigned_ids, scene_gnn_feat, scene_foreground_ids = model.get_anno(scene_id)

    scene_feat = pad_sequence((scene_feat,), batch_first=True).to(model.config.device)
    scene_img_feat = pad_sequence((scene_img_feat,), batch_first=True).to(model.config.device)
    scene_mask = pad_sequence((scene_mask,), batch_first=True).to(model.config.device)
    scene_locs = pad_sequence((scene_locs,), batch_first=True).to(model.config.device)
    assigned_ids = pad_sequence((assigned_ids,), batch_first=True).to(model.config.device)
    scene_gnn_feat = pad_sequence((scene_gnn_feat,), batch_first=True).to(model.config.device)
    scene_foreground_ids = (scene_foreground_ids,)
    obj_ids = torch.tensor([0]).to(model.config.device)

    print("=== Scene Understanding Demo ===")
    print("Type your query about the scene. Type 'exit' to quit.\n")

    while True:
        user_query = input("Your Query: ")
        
        if user_query.lower() == 'exit':
            print("Exiting the demo. Goodbye!")
            break
        # Process the query with the model
        response = model.process(scene_feat, scene_img_feat,  scene_locs, scene_mask, user_query, obj_ids, assigned_ids, scene_gnn_feat, scene_foreground_ids)
        print("Response:", response)

if __name__ == '__main__':
    config = Config.get_config()
    main(config)