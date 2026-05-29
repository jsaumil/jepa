import math
from functools import partial

import torch
import torch.nn as nn

from utils.patch_embed import PatchEmbed, PatchEmbed3D
from utils.modules import Block
from utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed

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