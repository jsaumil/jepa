import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.patch_embed import PatchEmbed
from vit import VisionTransformer
from mask.utils import apply_masks, get_complement_masks
from predictor import VisionTransformerPredictor

class Model(nn.Module):
    def __init__(self, num_patches=16, embed_dim=768, patch_size=16, in_chans=3, img_size=224, **kwargs):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.patch = PatchEmbed(patch_size=patch_size, in_chan=in_chans, embed_dim=embed_dim)
        self.pos = nn.Parameter(
            torch.zeros(1, self.num_patches, self.embed_dim),
        )
        self.vit = VisionTransformer()
        self.predictor = VisionTransformerPredictor()

    def forward(self, x, masks, train=False):
        x = self.patch(x)
        x = x + self.pos
        B, N, C = x.shape
        context = apply_masks(x, masks)
        complement_mask = get_complement_masks(masks, N)
        target = apply_masks(x, complement_mask)
        ctxt = self.vit(context)
        tgt = torch.zeros_like(x)
        tgt = tgt + self.pos
        tgt = apply_masks(tgt, complement_mask)
        out = self.predictor(ctxt, tgt, masks, complement_mask)
        loss = None
        if train is True:
            loss = F.mse_loss(out, target)

        return out, loss