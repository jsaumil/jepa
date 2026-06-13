import torch
import torch.nn as nn
from vision_transformer import VisionTransformer
from mask.utils import apply_masks, get_complement_masks

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = VisionTransformer()

    def forward(self, x, masks):
        B, N, C = x.shape
        context = apply_masks(x, masks)
        complement = get_complement_masks(masks, N)
        masked_complement = apply_masks(x, [complement])
        ctxt = self.vit(context)
        tgt = self.vit(masked_complement)

        return x