import math
from functools import partial

import torch
import torch.nn as nn

from utils.modules import Block
from utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from mask.utils import apply_masks

class VisionTransformerPredictor(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            num_frames=1,
            tubelet_size=2,
            embed_dim=768,
            predictor_embed_dim=384,
            depth=6,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            norm_layer=nn.LayerNorm,
            init_std=0.02,
            uniform_power=False,
            use_mask_tokens=False,
            num_mask_tokens=2,
            zero_init_mask_tokens=True
            ):
        super().__init__()
        # map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Mask tokens
        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList([
                nn.Parameter(torch.zero(1,1,self.predictor_embed_dim, bias=True))
                for i in range(num_mask_tokens)
            ])

        self.input_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        grid_size = self.input_size // self.patch_size
        grid_depth = self.num_frames // self.tubelet_size
        if self.is_video:
            self.num_patches = num_patches = (
                (num_frames // tubelet_size)
                * (img_size // patch_size) ** 2
            )
        else:
            self.num_patches = num_patches = (
                (img_size // patch_size) ** 2
            )

        self.uniform_power = uniform_power
        self.predictor_pos_embed = nn.Parameter(
            torch.zero(1,num_patches, predictor_embed_dim),
            requires_grad=False
        )

        # Attention Blocks
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.GELU,
                attn_drop=attn_drop_rate,
                grid_size=grid_size,
                grid_depth=grid_depth,
                norm_layer=norm_layer)
            for i in range(depth)])
        
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)


    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt, mask_index=1):
        """
        ctxt: context tokens
        tgt: target tokens
        masks_ctxt: indices of context tokens in input
        masks_tgt: indices of target tokens in input
        """
        if not isinstance(masks_ctxt, list):
            masks_ctxt = [masks_ctxt]
        if not isinstance(masks_tgt, list):
            masks_tgt = [masks_tgt]

        # Batch Size
        B = len(ctxt)//len(masks_ctxt)

        # Map context tokens to predictor dimensions
        x = self.predictor_embed(ctxt)
        _, N_ctxt, D = x.shape

        # context pos
        # tgt chahiye (image me mask k hi tgt hai)
        # 

        # add positional embedding to ctxt tokens
        # if self.predictor_pos_embed is not None:
