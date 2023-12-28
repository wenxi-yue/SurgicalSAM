import sys
sys.path.append("..")
from segment_anything import sam_model_registry
import torch 
from torch.utils.data import DataLoader
from dataset import Endovis18Dataset, Endovis17Dataset
from model import Prototype_Prompt_Encoder, Learnable_Prototypes
from model_forward import model_forward_function
import argparse
from utils import read_gt_endovis_masks, create_binary_masks, create_endovis_masks, eval_endovis


print("======> Process Arguments")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="endovis_2018", choices=["endovis_2018", "endovis_2017"], help='specify dataset')
parser.add_argument('--fold', type=int, default=0, choices=[0,1,2,3], help='specify fold number for endovis_2017 dataset')
args = parser.parse_args()


print("======> Set Parameters for Inference" )
dataset_name = args.dataset
fold = args.fold
thr = 0
data_root_dir = f"../data/{dataset_name}"


print("======> Load Dataset-Specific Parameters" )
if "18" in dataset_name:
    num_tokens = 2
    dataset = Endovis18Dataset(data_root_dir = data_root_dir, 
                                mode = "val",
                                vit_mode = "h")
    surgicalSAM_ckp = f"../ckp/surgical_sam/{dataset_name}/model_ckp.pth"
    
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir,
                                            mode = "val")

elif "17" in dataset_name:
    num_tokens = 4
    dataset = Endovis17Dataset(data_root_dir = data_root_dir, 
                                mode = "val",
                                fold = fold, 
                                vit_mode = "h",
                                version = 0)
    surgicalSAM_ckp = f"../ckp/surgical_sam/{dataset_name}/fold{fold}/model_ckp.pth"
    
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir,
                                            mode = "val",
                                            fold = fold)
    
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)


print("======> Load SAM" )
sam_checkpoint = "../ckp/sam/sam_vit_h_4b8939.pth"
model_type = "vit_h_no_image_encoder"
sam_prompt_encoder, sam_decoder = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_prompt_encoder.cuda()
sam_decoder.cuda()


print("======> Load Prototypes and Prototype-based Prompt Encoder" )
# define the models
learnable_prototypes_model = Learnable_Prototypes(num_classes = 7, feat_dim = 256).cuda()
protoype_prompt_encoder =  Prototype_Prompt_Encoder(feat_dim = 256, 
                                                    hidden_dim_dense = 128, 
                                                    hidden_dim_sparse = 128, 
                                                    size = 64, 
                                                    num_tokens = num_tokens).cuda()
            
# load the weight for prototype-based prompt encoder, mask decoder, and prototypes
checkpoint = torch.load(surgicalSAM_ckp)
protoype_prompt_encoder.load_state_dict(checkpoint['prototype_prompt_encoder_state_dict'])
sam_decoder.load_state_dict(checkpoint['sam_decoder_state_dict'])
learnable_prototypes_model.load_state_dict(checkpoint['prototypes_state_dict'])

# set requires_grad to False to the whole model 
for name, param in sam_prompt_encoder.named_parameters():
    param.requires_grad = False
for name, param in sam_decoder.named_parameters():
    param.requires_grad = False
for name, param in protoype_prompt_encoder.named_parameters():
    param.requires_grad = False
for name, param in learnable_prototypes_model.named_parameters():
    param.requires_grad = False


print("======> Start Inference")
binary_masks = dict()
protoype_prompt_encoder.eval()
sam_decoder.eval()
learnable_prototypes_model.eval()

with torch.no_grad():
    prototypes = learnable_prototypes_model()

    for sam_feats, mask_names, cls_ids, _, _ in dataloader: 
        
        sam_feats = sam_feats.cuda()
        cls_ids = cls_ids.cuda()    
                
        preds , preds_quality = model_forward_function(protoype_prompt_encoder, sam_prompt_encoder, sam_decoder, sam_feats, prototypes, cls_ids)    
 
        binary_masks = create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr)

endovis_masks = create_endovis_masks(binary_masks, 1024, 1280)
endovis_results = eval_endovis(endovis_masks, gt_endovis_masks)

print(endovis_results)