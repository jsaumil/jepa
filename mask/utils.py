import torch

def apply_masks(x, masks, concat=True):
    """
    x = [b,n,d]
    masks = [b,k] contains k patches in [n]
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1,1,x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x
    
    return torch.cat(all_x, dim=0)