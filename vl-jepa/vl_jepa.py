import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DeepFake
from decoder import MLPDecoder


class VLJEPA(nn.Module):
    """Final VL-JEPA model:

        video -> x-encoder -> predictor(query) -> y_pred -> MLP decoder -> logits -> loss

    The encoder part (x-encoder + predictor + embeddings) comes pretrained from
    train.py's best.pt and is frozen. The y-encoder was only needed to train that
    encoder, so it is dropped here. Only the MLP decoder is trained.
    """

    def __init__(self, embed_dim=768, vocab_size=50244, small_model=False,
                 hidden_dim=512, num_classes=2, dropout=0.1):
        super().__init__()

        if small_model:
            embed_dim = 256
            self.encoder = DeepFake(
                embed_dim=embed_dim,
                depth=4,
                num_heads=8,
                pred_depth=2,
                pred_heads=8,
                vocab_size=vocab_size,
            )
        else:
            self.encoder = DeepFake(embed_dim=embed_dim, vocab_size=vocab_size)
        # y-encoder is not part of the final model
        del self.encoder.y_encoder

        self.decoder = MLPDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def load_encoder(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Encoder checkpoint not found: {path}")
        ckpt = torch.load(path, map_location="cpu")
        state = {
            k: v for k, v in ckpt["model_state_dict"].items()
            if not k.startswith("y_encoder.")
        }
        # strict: the encoder must match best.pt exactly, otherwise we'd be
        # training a decoder on top of randomly initialised encoder weights.
        self.encoder.load_state_dict(state, strict=True)

        bad_params = [n for n, p in self.encoder.named_parameters() if not torch.isfinite(p).all()]
        if bad_params:
            raise RuntimeError(f"Encoder checkpoint has nan/inf in params: {bad_params}")
        print(f"Loaded encoder from {path} (saved_epoch={ckpt.get('epoch')} saved_loss={ckpt.get('loss')})")

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def train(self, mode=True):
        # keep the frozen encoder in eval mode even when model.train() is called
        super().train(mode)
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        return self

    def encode(self, x, query):
        """x-encoder + predictor, pooled over tokens -> [B, C]"""
        encoder_frozen = not any(p.requires_grad for p in self.encoder.parameters())
        with torch.set_grad_enabled(torch.is_grad_enabled() and not encoder_frozen):
            return self.encoder(x, query, None, train=False).mean(dim=1)

    def forward(self, x, query, labels=None):
        y_pred = self.encode(x, query)
        logits = self.decoder(y_pred.float())
        out = {"y_pred": y_pred, "logits": logits}

        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)

        return out

    @torch.no_grad()
    def predict(self, x, query):
        """Returns (pred_label, prob_fake). label 1 = fake, 0 = real."""
        logits = self(x, query)["logits"]
        probs = logits.softmax(dim=-1)
        return probs.argmax(dim=-1), probs[:, 1]
