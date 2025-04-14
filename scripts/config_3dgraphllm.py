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
    system_path="./prompts/system.txt",
    instruction_path="./prompts/instruction.txt",
    max_txt_len=64,
    end_sym="</s>",
    role=("USER", "ASSISTANT"),
    add_scene_token=False,
    add_img_token=True,
    use_lora=True,
    train_emb=True,
    train_img_proj=True,
    no_obj=False,
    max_obj_num=150,
    bidirection=False,
    add_pos_emb=False,
    feat_fusion=False,
    fuse_with_id=False,
    use_objid=True,
    use_location_token=False,
    knn=2,
    bbox_embed=False,
    gt_pretrain=False,
    nms=True,
    nn_distance=True,
    max_knn=2
)

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

optimizer = dict(
    opt="adamW",
    lr=5e-6,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    scaler_enable=False,
    max_grad_norm=0.01,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(
        enable=False,
        module_names=["model.embed_tokens"],
        lr=[5e-4],
        wd=[0.02]
    ),
)

scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.01, warmup_epochs=0.1)

evaluate = False

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="anonym",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="3DGraphLLM",
)
dist_url = "env://"
device = "cuda"

# ========================= others ==========================
output_dir = "./llama3-8b-gt-pretrain-2-3rscan"  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 20
# eval_freq = 500
seed = 42

save_latest = False
do_save = True
auto_resume = True
pretrained_path = ""
img_projector_path = ""

debug=False