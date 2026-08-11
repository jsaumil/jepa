import torch
import torch.nn as nn

from x_encoder import VisionTransformer
from y_encoder import Transformer
from predictor import Predictor
from utils.patch_embed import PatchEmbed3D

class DeepFake(nn.Module):
    def __init__(self, num_patchers=16,embed_dim=768,patch_size=16, **kwargs):
        super().__init__()

        self.x_encoder = VisionTransformer()
        self.y_encoder = Transformer()
        self.predictor = Predictor()
