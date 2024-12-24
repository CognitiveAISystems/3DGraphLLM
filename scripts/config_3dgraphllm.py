# ========================= data ==========================
anno_root = "annotations"  # annotation dir
pc_encoder = "uni3d"
segmentor = "mask3d"
version = ""

seg_feat_file = f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt"
seg_img_feat_file = f"{anno_root}/scannet_{segmentor}_videofeats.pt"
seg_val_attr_file = f"{anno_root}/scannet_{segmentor}_val_attributes.pt"
seg_val_gnn_file =  f"{anno_root}/scannet_{segmentor}_val_gnn_feats_2{version}.pt"

val_file_dict = {
    'demo': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        seg_val_gnn_file,
        f"{segmentor}",
    ],
}

# ========================= model ==========================
model = dict(
    llama_model_path="./Meta-Llama-3-8B-Instruct",
    input_dim=1024,
    img_input_dim=1024,
    attr_dim=512,
    scene_dim=256,
    pos_dim=128,
    encoder_num_layers=3,
    low_resource=False,
    system_path="prompts/system.txt",
    instruction_path="prompts/instruction.txt",
    max_txt_len=64,
    end_sym="</s>",
    role=("USER", "ASSISTANT"),
    add_scene_token=False,
    add_img_token=True,
    use_lora=True,
    train_emb=True,
    train_img_proj=True,
    no_obj=False,
    max_obj_num=100,
    bidirection=False,
    add_pos_emb=False,
    feat_fusion=False,
    fuse_with_id=False,
    use_objid=True,
    use_location_token=False,
    knn=2
)

debug=False
device = "cuda"

lora = dict(
    lora_target_modules=[
      "q_proj",
      "v_proj",
      "k_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj"
    ],
    lora_r=16,
    lora_alpha=16,
    lora_dropout=0.05
)