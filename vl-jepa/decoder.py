import torch.nn as nn


class MLPDecoder(nn.Module):
    """Simple MLP head: pooled predictor embedding [B, C] -> class logits [B, num_classes]."""

    def __init__(self, embed_dim=768, hidden_dim=512, num_classes=2, dropout=0.1):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.mlp(x)
