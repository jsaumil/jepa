import torch
import torch.nn as nn
import torch.nn.functional as F
from x_encoder import VisionTransformer
from y_encoder import Transformer
from predictor import Predictor
from utils.patch_embed import PatchEmbed3D
from info_nce import InfoNCE, info_nce

class DeepFake(nn.Module):
    def __init__(self, max_k=2000,embed_dim=768,vocab_size=50244, **kwargs):
        super().__init__()

        self.x_encoder = VisionTransformer()
        self.y_encoder = Transformer()
        self.predictor = Predictor()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        )
        
        self.max_k = max_k
        self.embed_dim = embed_dim
        self.patcher = PatchEmbed3D()
        # fix the positional embeddings
        self.pos = nn.Parameter(
            torch.zeros(1,self.max_k, self.embed_dim)
        )
        self.text_pos = nn.Parameter(
            torch.zeros(1, vocab_size, embed_dim)
        )
        self.loss_fun = InfoNCE()

    def forward(self, x, query, y, train=False):
        x = self.patcher(x)
        B, N, C = x.shape
        x = x + self.pos[:, :N, :]
        x = self.x_encoder(x)
        q = self.embed(query)
        Bq, Q, Cq = q.shape
        q = q + self.text_pos[:, :Q, :]
        y_pred = self.predictor(x,q)
        loss = None
        if train:
            y = self.embed(y)
            By, Y, Cy = y.shape
            y = y + self.pos[:, :Y, :]
            y = self.y_encoder(y)
            loss = F.mse_loss(y_pred,y)

        return y_pred, loss