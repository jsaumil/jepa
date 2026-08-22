import torch
import tiktoken
class Dataloader:
    def __init__(self, B,T,text):
        self.B = B
        self.T = T

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.current_position = 0