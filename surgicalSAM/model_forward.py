import torch 
from einops import rearrange
from torch.nn import functional as F

# forward process of the model
def model_forward_function(prototype_prompt_encoder, 
                            sam_prompt_encoder, 
                            sam_decoder, 
                            sam_feats, 
                            prototypes, 
                            cls_ids): 
        
    sam_feats = rearrange(sam_feats, 'b h w c -> b (h w) c')

    
    dense_embeddings, sparse_embeddings = prototype_prompt_encoder(sam_feats, prototypes, cls_ids)

    pred = []
    pred_quality = []
    sam_feats = rearrange(sam_feats,'b (h w) c -> b c h w', h=64, w=64)
 
    for dense_embedding, sparse_embedding, features_per_image in zip(dense_embeddings.unsqueeze(1), sparse_embeddings.unsqueeze(1), sam_feats):    
        
        low_res_masks_per_image, mask_quality_per_image = sam_decoder(
                image_embeddings=features_per_image.unsqueeze(0),
                image_pe=sam_prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=sparse_embedding,
                dense_prompt_embeddings=dense_embedding, 
                multimask_output=False,
            )

        pred_per_image = postprocess_masks(
            low_res_masks_per_image,
            input_size=(819, 1024),
            original_size=(1024, 1280),
        )
        
        pred.append(pred_per_image)
        pred_quality.append(mask_quality_per_image)
        
    pred = torch.cat(pred,dim=0).squeeze(1)
    pred_quality = torch.cat(pred_quality,dim=0).squeeze(1)
    
    return pred, pred_quality



# taken from sam.postprocess_masks of https://github.com/facebookresearch/segment-anything
def postprocess_masks(masks, input_size, original_size):
    """
    Remove padding and upscale masks to the original image size.

    Arguments:
        masks (torch.Tensor): Batched masks from the mask_decoder,
        in BxCxHxW format.
        input_size (tuple(int, int)): The size of the image input to the
        model, in (H, W) format. Used to remove padding.
        original_size (tuple(int, int)): The original size of the image
        before resizing for input to the model, in (H, W) format.

    Returns:
        (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
        is given by original_size.
    """
    masks = F.interpolate(
        masks,
        (1024, 1024),
        mode="bilinear",
        align_corners=False,
    )
    masks = masks[..., : input_size[0], : input_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks

