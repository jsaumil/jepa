import math
from functools import partial

import torch
import torch.nn as nn

from utils.patch_embed import PatchEmbed, PatchEmbed3D
from utils.modules import Block
from utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from mask.utils import apply_masks

class VisionTransformer(nn.Module):
    """Vision Transformer"""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        int_std=0.02,
        out_layers=None,
        uniform_power=False,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers

        self.input_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        grid_size = self.input_size // self.patch_size
        grid_depth = self.num_frames // self.tubelet_size

        # Tokenize pixels with convolution
        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans,
                embed_dim=embed_dim
            )
            self.num_patches = (
                (num_frames // tubelet_size)
                * (img_size // patch_size) ** 2
            )
        else:
            self.patch_embed = PatchEmbed(
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim
            )
            self.num_patches = (
                (img_size // patch_size) ** 2
            )

        # Position embedding
        self.uniform_power = uniform_power
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim),
            requires_grad=False
        )

        # Attention Blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.GELU,
                grid_size=grid_size,
                grid_depth=grid_depth,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --- initialize weights ---
        
    def forward(self, x, masks=None):
        """
        x: input image/video
        mask: indices of patch tokens to mask (remove)
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        # Tokenize input
        pos_embed = self.pos_embed
        x = self.patch_embed(x) # [B, num_patches, embed_dim]
        if pos_embed is not None:
            x += pos_embed
        B, N, D = x.shape

        # Mask away unwanted tokens
        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask=masks)
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norm(x))

        if self.out_layers is None:
            return outs
        if self.norm is not None:
            x = self.norm(x)

        return x