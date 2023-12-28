import sys
sys.path.append("..")
import os
import os.path as osp 
import random 
import argparse
import numpy as np 
import torch 
from torch.utils.data import DataLoader
from dataset import Endovis18Dataset, Endovis17Dataset
from segment_anything import sam_model_registry
from model import Learnable_Prototypes, Prototype_Prompt_Encoder
from utils import print_log, create_binary_masks, create_endovis_masks, eval_endovis, read_gt_endovis_masks
from model_forward import model_forward_function
from loss import DiceLoss
from pytorch_metric_learning import losses

print("======> Process Arguments")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="endovis_2018", choices=["endovis_2018", "endovis_2017"], help='specify dataset')
parser.add_argument('--fold', type=int, default=0, choices=[0,1,2,3], help='specify fold number for endovis_2017 dataset')
args = parser.parse_args()

print("======> Set Parameters for Training" )
dataset_name = args.dataset
fold = args.fold
thr = 0
seed = 666  
data_root_dir = f"../data/{dataset_name}"
batch_size = 32
vit_mode = "h"

# set seed for reproducibility 
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

print("======> Load Dataset-Specific Parameters" )
if "18" in dataset_name:
    num_tokens = 2
    val_dataset = Endovis18Dataset(data_root_dir = data_root_dir, 
                                   mode="val",
                                   vit_mode = "h")
    
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir, mode = "val")
    num_epochs = 500
    lr = 0.001
    save_dir = "./work_dirs/endovis_2018/"

elif "17" in dataset_name:
    num_tokens = 4
    val_dataset = Endovis17Dataset(data_root_dir = data_root_dir,
                                   mode = "val",
                                   fold = fold, 
                                   vit_mode = "h",
                                   version = 0)
    
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir, 
                                             mode = "val", 
                                             fold = fold)
    num_epochs = 2000
    lr = 0.0001
    save_dir = f"./work_dirs/endovis_2017/{fold}"
    
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

print("======> Load SAM" )
if vit_mode == "h":
    sam_checkpoint = "../ckp/sam/sam_vit_h_4b8939.pth"
model_type = "vit_h_no_image_encoder"
sam_prompt_encoder, sam_decoder = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_prompt_encoder.cuda()
sam_decoder.cuda()

for name, param in sam_prompt_encoder.named_parameters():
    param.requires_grad = False
for name, param in sam_decoder.named_parameters():
    param.requires_grad = True

print("======> Load Prototypes and Prototype-based Prompt Encoder" )
learnable_prototypes_model = Learnable_Prototypes(num_classes = 7, feat_dim = 256).cuda()
protoype_prompt_encoder =  Prototype_Prompt_Encoder(feat_dim = 256, 
                                                    hidden_dim_dense = 128, 
                                                    hidden_dim_sparse = 128, 
                                                    size = 64, 
                                                    num_tokens = num_tokens).cuda()
 
with open(sam_checkpoint, "rb") as f:
    state_dict = torch.load(f)
    sam_pn_embeddings_weight = {k.split("prompt_encoder.point_embeddings.")[-1]: v for k, v in state_dict.items() if k.startswith("prompt_encoder.point_embeddings") and ("0" in k or "1" in k)}
    sam_pn_embeddings_weight_ckp = {"0.weight": torch.concat([sam_pn_embeddings_weight['0.weight'] for _ in range(num_tokens)], dim=0),
                                    "1.weight": torch.concat([sam_pn_embeddings_weight['1.weight'] for _ in range(num_tokens)], dim=0)}

    protoype_prompt_encoder.pn_cls_embeddings.load_state_dict(sam_pn_embeddings_weight_ckp)

for name, param in learnable_prototypes_model.named_parameters():
    param.requires_grad = True
    
for name, param in protoype_prompt_encoder.named_parameters():
    if "pn_cls_embeddings" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True
              
print("======> Define Optmiser and Loss")
seg_loss_model = DiceLoss().cuda()
contrastive_loss_model = losses.NTXentLoss(temperature=0.07).cuda()
optimiser = torch.optim.Adam([
            {'params': learnable_prototypes_model.parameters()},
            {'params': protoype_prompt_encoder.parameters()},
            {'params': sam_decoder.parameters()}
        ], lr = lr, weight_decay = 0.0001)


print("======> Set Saving Directories and Logs")
os.makedirs(save_dir, exist_ok = True) 
log_file = osp.join(save_dir, "log.txt")
print_log(str(args), log_file)


print("======> Start Training and Validation" )
best_challenge_iou_val = -100.0

for epoch in range(num_epochs):   
    
    # choose the augmentation version to use for the current epoch 
    if epoch % 2 == 0 :
        version = 0 
    else:
        version = int((epoch % 80 + 1)/2)
    
    if "18" in dataset_name:
        train_dataset = Endovis18Dataset(data_root_dir = data_root_dir,
                                         mode="train",
                                         vit_mode = vit_mode,
                                         version = version)
        
    elif "17" in dataset_name:
        train_dataset = Endovis17Dataset(data_root_dir = data_root_dir,
                                         mode="train",
                                         fold = fold,
                                         vit_mode = vit_mode,
                                         version = version)
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    
    # training 
    protoype_prompt_encoder.train()
    sam_decoder.train()
    learnable_prototypes_model.train()

    for sam_feats, _, cls_ids, masks, class_embeddings in train_dataloader: 
        
        sam_feats = sam_feats.cuda()
        cls_ids = cls_ids.cuda()
        masks = masks.cuda()
        class_embeddings = class_embeddings.cuda()
        
        prototypes = learnable_prototypes_model()
        
        preds, _ = model_forward_function(protoype_prompt_encoder, sam_prompt_encoder, sam_decoder, sam_feats, prototypes, cls_ids)    
        
        # compute loss 
        contrastive_loss = contrastive_loss_model(prototypes, torch.tensor([i for i in range(1, prototypes.size()[0] + 1)]).cuda(), ref_emb = class_embeddings, ref_labels = cls_ids)
        seg_loss = seg_loss_model(preds, masks/255)
    
        loss = seg_loss + contrastive_loss
   
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
    # validation 
    binary_masks = dict()
    protoype_prompt_encoder.eval()
    sam_decoder.eval()
    learnable_prototypes_model.eval()

    with torch.no_grad():
        prototypes = learnable_prototypes_model()
        
        for sam_feats, mask_names, cls_ids, _, _ in val_dataloader: 
            
            sam_feats = sam_feats.cuda()
            cls_ids = cls_ids.cuda()    
            
            preds , preds_quality = model_forward_function(protoype_prompt_encoder, sam_prompt_encoder, sam_decoder, sam_feats, prototypes, cls_ids)    
 
            binary_masks = create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr)

    endovis_masks = create_endovis_masks(binary_masks, 1024, 1280)
    endovis_results = eval_endovis(endovis_masks, gt_endovis_masks)
            
    print_log(f"Validation - Epoch: {epoch}/{num_epochs-1}; IoU_Results: {endovis_results} ", log_file)
    
    if endovis_results["challengIoU"] > best_challenge_iou_val:
        best_challenge_iou_val = endovis_results["challengIoU"]
        
        torch.save({
            'prototype_prompt_encoder_state_dict': protoype_prompt_encoder.state_dict(),
            'sam_decoder_state_dict': sam_decoder.state_dict(),
            'prototypes_state_dict': learnable_prototypes_model.state_dict(),
        }, osp.join(save_dir,'model_ckp.pth'))

        print_log(f"Best Challenge IoU: {best_challenge_iou_val:.4f} at Epoch {epoch}", log_file)        
