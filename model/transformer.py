import torch
import torch.nn as nn
from .embedding import Embedding
from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm
from .linear import Linear
from .softmax import softmax

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta):
        super().__init__()

        self.num_layers = num_layers

        self.embedder = Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([TransformerBlock(d_model,num_heads, d_ff, theta, context_length) for _ in range(num_layers)])

        self.rmsnorm = RMSNorm(d_model)

        self.linear = Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedder(x)
        for trans_block in self.layers:
            x = trans_block(x)

        x = self.rmsnorm(x)
        x = self.linear(x)
        return x

    def load_weights(self, state_dict):

        self.embedder.load_state_dict({
            "weights": state_dict["token_embeddings.weight"]
        })

        for i, layer in enumerate(self.layers):
            layer.load_weights(state_dict, i)

        self.rmsnorm.load_state_dict({
            "g": state_dict["ln_final.weight"]
        })

        self.linear.load_state_dict({
            "weights": state_dict["lm_head.weight"]
        })
